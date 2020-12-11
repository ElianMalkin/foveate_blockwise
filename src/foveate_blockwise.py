#!/usr/bin/env python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from os import makedirs
from PIL import Image
import math
import numpy as np
import cv2 as cv
import sys
import getopt

def make_k(sig, ker_size):
	# Makes Gaussian Filter:
	s = ker_size
	out = np.zeros((s,s))
	for x in range(s):
		for y in range(s):
			X = x-(s-1)/2
			Y = y-(s-1)/2
			gauss = 1/(2*np.pi*sig**2) * np.exp(-(X**2 + Y**2)/(2*sig**2))
			out[x,y] = gauss
	a = np.sum(out)
	kernel = out/a
	return kernel

def separate_kernel(kernel):
	# Separates Gaussian Filter into 1-D filter:
	u, s, v = np.linalg.svd(kernel)
	first = u[:,0:1]*np.sqrt(s[0])
	second = v[0:1,:]*np.sqrt(s[0])
	first = first.flatten()
	second = second.flatten()
	return [first, second]

def calc_ker_size(sig):
	# Calculates size of Gaussian filter given its standard deviation:
	s = int(1+2*np.sqrt(-2*sig**2*np.log(0.005)))
	if s % 2 == 0:
		s += 1
	return np.int32(s)

def replication_pad(img, S):
	# Pads an image by replicating the edge pixels:
	output = cv.copyMakeBorder(img, S, S, S, S, cv.BORDER_REPLICATE)
	return output

def load_image(image_loc, image_name, W, H):
	img = cv.imread(image_loc + image_name)

	if img is None:
		sys.exit("Could not read the image.")

	if H == -1 and W == -1:
		pass
	elif W == -1:
		img = img[:H,:, :]
	elif H == -1:
		img = img[:,:W, :]
	else:
		img = img[:H,:W, :]

	return np.ascontiguousarray(img)

def make_distances(minDist, maxDist, fragmentW, fragmentH, coef = 1.4):
	# Makes list of distances denoting pooling region boundaries
	dist_l = [minDist]
	d = minDist + coef*fragmentW
	while d < maxDist:
		dist_l.append(d)
		d += coef*fragmentW

	num_regions = len(dist_l)

	# Maximized given fragment size:
	print("Number of pooling regions: " + str(num_regions))

	return (dist_l, num_regions)


def transfer_input_img(img_padded):
	img_gpu = cuda.mem_alloc(img_padded.nbytes)

	img_pinned = cuda.pagelocked_empty_like(img_padded)
	img_pinned[:] = img_padded

	start = cuda.Event()
	end = cuda.Event()
	start.record()
	cuda.memcpy_htod(img_gpu, img_pinned)
	end.record()
	end.synchronize()
	millis = start.time_till(end)
	print("Transferring to GPU: " + str(millis) + " ms")

	return img_gpu


def calc_shift(fovx, fovy, fragmentW, fragmentH, W, H):
	# Calculates tiling shift to center fragment on fixation point
	# Also calculates number of fragments to add to grid due to shift
	r_w = np.remainder(fovx, fragmentW)/fragmentW
	if r_w <= 0.5:
		shift_w = 0.5 - r_w
	else:
		shift_w = 1.5 - r_w

	r_h = np.remainder(fovy, fragmentH)/fragmentH
	if r_h <= 0.5:
		shift_h = 0.5 - r_h
	else:
		shift_h = 1.5 - r_h

	width_shift = np.int32(fragmentW * shift_w)
	height_shift = np.int32(fragmentH * shift_h)

	r_grid_w = np.remainder(W, fragmentW)/fragmentW
	if r_grid_w + shift_w <= 1:
		plus_blocks_w = 1
	else:
		plus_blocks_w = 2

	r_grid_h = np.remainder(H, fragmentH)/fragmentH
	if r_grid_h + shift_h <= 1:
		plus_blocks_h = 1
	else:
		plus_blocks_h = 2

	return (width_shift, height_shift, plus_blocks_w, plus_blocks_h)


def make_blur_grid(gridW, gridH, fragmentW, fragmentH, fovx, fovy, width_shift, height_shift, dist_l):
	# Grid that stores blurring strengths for each fragment (indices of Gaussian filters to use)
	# This function simply uses the distances list for a circularly symmetric blur_grid
	# Can be adapted to use a non-symmetric blur strength distribution
	blur_grid = np.zeros((gridH, gridW), dtype="int32")

	for i in range(gridH):
		row = int(i*fragmentH + fragmentH/2 - height_shift)

		for j in range(gridW):
			col = int(j*fragmentW + fragmentW/2 - width_shift)

			distance = math.sqrt((fovx - col)**2 + (fovy - row)**2)
			if distance > dist_l[-1]:
				grid_val = len(dist_l)
			else:
				for v in range(len(dist_l)):	
					if distance < dist_l[v]:
						grid_val = v
						break

			blur_grid[i, j] = grid_val

	blur_grid_gpu = cuda.mem_alloc(blur_grid.nbytes)
	cuda.memcpy_htod(blur_grid_gpu, blur_grid)

	return blur_grid_gpu


def make_kernel(S, fragmentW, fragmentH, num_regions):
	# CUDA kernel
	tileW = fragmentW + 2*S
	tileH = fragmentH + 2*S

	mod = SourceModule("""
	#include "stdint.h"

	const unsigned int MAX_FILTER_SIZE = 4000;

	const unsigned int NUM_REGIONS = """ + str(num_regions) + """;

	__device__ __constant__ float d_g[MAX_FILTER_SIZE];
	__device__ __constant__ int k_c[NUM_REGIONS+1];

	__global__ void convolution( const uint8_t *d_f, const unsigned int *blur_grid, const unsigned int paddedW, const unsigned int paddedH,
										  const unsigned int blockW, const unsigned int blockH, const int S, 
										  uint8_t *d_h, const unsigned int W,
										  const unsigned int fovx, const unsigned int fovy, const int width_shift, const int height_shift)
	{



		const unsigned int tileW = blockW + 2 * S;

		const int blockStartCol = blockIdx.x * blockW + S - width_shift;
		const int blockStartRow = blockIdx.y * blockH + S - height_shift;

		const unsigned int blockEndCol = blockStartCol + blockW;
		const unsigned int blockEndRow = blockStartRow + blockH;

		const int tileStartCol = blockStartCol - S;
		const unsigned int tileEndCol = blockEndCol + S;
		const unsigned int tileEndClampedCol = min( tileEndCol, paddedW );
		const int tileStartClampedCol = max(0, tileStartCol);

		const int tileStartRow = blockStartRow - S;
		const unsigned int tileEndRow = blockEndRow + S;
		const unsigned int tileEndClampedRow = min( tileEndRow, paddedH );
		const int tileStartClampedRow = max(0, tileStartRow);

		bool no_change = false;
		unsigned int kernel_offset = 0;
		unsigned int ker_size = 0;

		unsigned int grid_pos = blockIdx.y * gridDim.x + blockIdx.x;
		unsigned int k_c_ind = blur_grid[grid_pos];

		if( k_c_ind == 0 ) {
			no_change = true;
		} else {
			kernel_offset = k_c[k_c_ind-1];
			ker_size = k_c[k_c_ind] - k_c[k_c_ind-1];
		}


		
		for( unsigned int t = 0; t <= 2; t++ ) {

			if( no_change == true ) {
				unsigned int numThreads = blockDim.x;
				unsigned int iterations = ceilf(__fdividef(blockW*blockH,numThreads));
				for( unsigned int iterNo = 0; iterNo < iterations; iterNo++ ) {
					unsigned int idx = numThreads*iterNo + threadIdx.x;
					unsigned int bRow = __fdividef(idx, blockW);
					unsigned int bCol = idx - bRow*blockW;
					unsigned int iPixelPosCol = blockStartCol + bCol;
					unsigned int iPixelPosRow = blockStartRow + bRow;

					if( iPixelPosCol >= tileStartClampedCol + S && iPixelPosRow >= tileStartClampedRow + S && 
						iPixelPosCol < tileEndClampedCol - S && iPixelPosRow < tileEndClampedRow - S ) {
						unsigned int oPixelPosCol = iPixelPosCol - S; // removing the origin
						unsigned int oPixelPosRow = iPixelPosRow - S;
						unsigned int oPixelPos = oPixelPosRow * W + oPixelPosCol;

						unsigned int iPixelPos = iPixelPosRow * paddedW + iPixelPosCol;
						d_h[oPixelPos*3 + t] = d_f[iPixelPos*3 + t];
					}
				}

			} else {

				const int u = (ker_size-1)/2;

				__shared__ uint8_t sData[""" + str(tileW*tileH) + """];
				__shared__ float intermData[""" + str(fragmentW*tileH) + """];
			
				unsigned int numThreads = blockDim.x;
				unsigned int wid = ker_size-1 + blockW;
				unsigned int hig = ker_size-1 + blockH;
				unsigned int iterations = ceilf(__fdividef(wid*hig, numThreads*2));
				for( unsigned int iterNo = 0; iterNo < iterations; iterNo++ ) {
					unsigned int idx = 2*(numThreads*iterNo + threadIdx.x);

					unsigned int tRow = S-u + __fdividef(idx, wid);
					unsigned int tCol = S-u + idx - (tRow-(S-u))*wid;

					unsigned int iPixelPosCol = tileStartCol + tCol;
					unsigned int iPixelPosRow = tileStartRow + tRow;

					if( iPixelPosCol >= tileStartClampedCol && iPixelPosRow >= tileStartClampedRow &&
							iPixelPosCol < tileEndClampedCol && iPixelPosRow < tileEndClampedRow ) {
						unsigned int iPixelPos = iPixelPosRow * paddedW + iPixelPosCol;
						unsigned int tilePixelPos = tRow * tileW + tCol;
						sData[tilePixelPos] = d_f[iPixelPos*3 + t];
						sData[tilePixelPos+1] = d_f[(iPixelPos+1)*3 + t];
					}
					
				}		
					
				__syncthreads();



				wid = blockW;
				iterations = ceilf(__fdividef(wid*hig, numThreads*4));
				for( unsigned int iterNo = 0; iterNo < iterations; iterNo++ ) {

					unsigned int idx = 4*numThreads*iterNo + 4*threadIdx.x;

					if( idx < wid*hig ) {
					
						unsigned int tRow = S-u + __fdividef(idx, wid);
						unsigned int tCol = S + idx - (tRow-(S-u))*wid;
						unsigned int iPixelPosCol = tileStartCol + tCol;
						unsigned int iPixelPosRow = tileStartRow + tRow;

						if( iPixelPosCol >= tileStartClampedCol + S && iPixelPosRow >= tileStartClampedRow + (S-u) && 
								iPixelPosCol < tileEndClampedCol - S && iPixelPosRow < tileEndClampedRow ) {

							unsigned int tilePixelPos = tRow * tileW + tCol;
							unsigned int intermPixelPos = tRow * blockW + tCol - S;

							float tempSum = 0.0;
							float tempSum2 = 0.0;
							float tempSum3 = 0.0;
							float tempSum4 = 0.0;
							
							uint8_t old = sData[ tilePixelPos - u];
							uint8_t old2 = sData[ tilePixelPos - u + 1];
							uint8_t old3 = sData[ tilePixelPos - u + 2];
							
							for( int j = -u; j <= u; j++ ) {
								float filtCoef = d_g[kernel_offset + j + u];
								tempSum += old * filtCoef;
								tempSum2 += filtCoef * old2;
								tempSum3 += filtCoef * old3;
								old = old2;
								old2 = old3;
								old3 = sData[ tilePixelPos + j + 3 ];
								tempSum4 += filtCoef * old3;

								//old = sData[tilePixelPos + j + 1];
								//tempSum2 += filtCoef * old;

								//int tilePixelPosOffset = j;
								//int coefPos = ( j + u );
								//tempSum += sData[ tilePixelPos + tilePixelPosOffset ] * d_g[kernel_offset + coefPos];
							}
							intermData[intermPixelPos] = tempSum;
							intermData[intermPixelPos + 1] = tempSum2;
							intermData[intermPixelPos + 2] = tempSum3;
							intermData[intermPixelPos + 3] = tempSum4;

						}
					}
				}
				
				__syncthreads();



				hig = blockH;
				iterations = ceilf(__fdividef(wid*hig, 4*numThreads));

				for( unsigned int iterNo = 0; iterNo < iterations; iterNo++ ) {

					

					unsigned int idx = numThreads*iterNo + threadIdx.x;
					unsigned int tRow = __fdividef(idx, wid);
					unsigned int tCol = S + idx - (tRow)*wid;
					unsigned int iPixelPosCol = tileStartCol + tCol;
					unsigned int iPixelPosRow = tileStartRow + 4*tRow + S;

					if( iPixelPosCol >= tileStartClampedCol + S && iPixelPosCol < tileEndClampedCol - S &&
						iPixelPosRow >= tileStartClampedRow + S && iPixelPosRow < tileEndClampedRow - S - 3 ) {



						unsigned int oPixelPosCol = iPixelPosCol - S; // removing the origin
						unsigned int oPixelPosRow = iPixelPosRow - S;
						unsigned int oPixelPos = oPixelPosRow * W + oPixelPosCol;

						unsigned int tilePixelPos = (4*tRow + S) * blockW + tCol - S;

						float tempSum = 0.0;
						float tempSum2 = 0.0;
						float tempSum3 = 0.0;
						float tempSum4 = 0.0;

						float old = intermData[ tilePixelPos - u * blockW];
						float old2 = intermData[ tilePixelPos + (- u + 1) * blockW];
						float old3 = intermData[ tilePixelPos + (- u + 2) * blockW];


						for( int i = -u; i <= u; i++ ) {
							float filtCoef = d_g[kernel_offset + i + u];
							tempSum += old * filtCoef;
							tempSum2 += old2 * filtCoef;
							tempSum3 += old3 * filtCoef;
							old = old2;
							old2 = old3;
							old3 = intermData[ tilePixelPos + (i+3) * blockW ];
							tempSum4 += old3 * filtCoef;


							//old = intermData[ tilePixelPos + (i+1) * blockW ];
							//tempSum2 += filtCoef * old;

							//int coefPos = ( i + u );
							//int tilePixelPosOffset = i * blockW;
							//tempSum += intermData[ tilePixelPos + tilePixelPosOffset ] * d_g[kernel_offset + coefPos];
						}
						d_h[oPixelPos*3 + t] = tempSum;
						d_h[(oPixelPos + W)*3 + t] = tempSum2;
						d_h[(oPixelPos + 2*W)*3 + t] = tempSum3;
						d_h[(oPixelPos + 3*W)*3 + t] = tempSum4;
					}
				}
			}
			__syncthreads();
		}
	}

	""")

	return mod


def transfer_filters(sigma_l, mod):
	ker_l = []
	ker_cum = [0]
	for sig in sigma_l:
		ker_size = calc_ker_size(sig)
		kernel = make_k(sig, ker_size)
		oneD_kernel = list(separate_kernel(kernel)[0])
		ker_l += oneD_kernel
		ker_cum.append(ker_cum[-1] + ker_size)

	ker_l = np.float32(np.array(ker_l))
	ker_cum = np.int32(np.array(ker_cum))

	d_g_pt = mod.get_global("d_g")[0]
	k_c_pt = mod.get_global("k_c")[0]

	cuda.memcpy_htod(d_g_pt, ker_l)
	cuda.memcpy_htod(k_c_pt, ker_cum)


def run_foveation(mod, gridW, gridH, fragmentW, fragmentH, img_gpu, blur_grid_gpu, paddedW, paddedH, S, result_gpu, W, width_shift, height_shift, threadsPerBlock, fovx, fovy):
	func = mod.get_function("convolution")
	func.prepare(("P", "P", "i", "i", "i", "i", "i", "P", "i", "i", "i", "i", "i"))

	block=(threadsPerBlock, 1, 1)
	grid = (gridW, gridH)

	start=cuda.Event()
	end=cuda.Event()

	start.record() # start timing

	func.prepared_call(grid, block, img_gpu, 
				blur_grid_gpu, paddedW, paddedH, 
				np.int32(fragmentW), np.int32(fragmentH), np.int32(S), 
				result_gpu, np.int32(W), np.int32(fovx), 
				np.int32(fovy), width_shift, height_shift)

	end.record() # end timing
	end.synchronize()
	millis = start.time_till(end)

	print("Time taken: " + str(millis) + " ms")


def transfer_result(host_output, result_gpu):
	start = cuda.Event()
	end = cuda.Event()

	start.record()
	cuda.memcpy_dtoh(host_output, result_gpu)
	end.record()
	end.synchronize()
	millis = start.time_till(end)
	print("Transferring to CPU: " + str(millis) + " ms")

def make_sigmas(dist_l, maxDist):
	# Contrast Sensitivity Function parameters:
	e2 = 2.3
	alpha = 0.106
	CT_0 = 0.0133

	# Maximum equivalent eccentricity of image (human lateral FOV is 100 degrees):
	max_ecc = 100

	# Maximum cycles per degree of the mapped retina (setting at 30 for human):
	max_cyc_per_deg = 30

	sigma_l = []
	for i in range(len(dist_l)):
		# SET BLUR AT CENTER OF POOLING REGION
		if i == len(dist_l) - 1:
			pixDist = dist_l[i] + (maxDist - dist_l[i])/2
		else:
			pixDist = dist_l[i] + (dist_l[i+1] - dist_l[i])/2

		if pixDist > maxDist:
			pixDist = maxDist

		ecc = pixDist
		fc_deg = 1/(ecc/maxDist*max_ecc + e2) * (e2/alpha * np.log(1/CT_0))

		if fc_deg > max_cyc_per_deg:
			fc_deg = max_cyc_per_deg

		sig_f = 0.5 * fc_deg/max_cyc_per_deg
	 
		sigma = 1/(2*np.pi*sig_f)

		sigma_l.append(sigma)

	return sigma_l


def usage():
	print('Usage: python src/foveate_blockwise.py [options]')
	print('Real-time image foveation transform using PyCuda')
	print('Options:')
	print('-h, --help\t\t', 'Displays this help')
	print('-p, --gazePosition\t', 'Gaze position coordinates, (vertical down) then (horizontal right), (e.g. "-p 512,512"), default: center of the image')
	print('-f, --fragmentSize\t', 'Width and height of fragments for foveation, (e.g. "-f 16,16"), default: 32 x 32')
	print('-v, --visualize\t\t', 'Show foveated images')
	print('-i, --inputFile\t\t', 'Input image from "images" folder, default: "castle.jpg"')
	print('-o, --outputDir\t\t', 'Output directory and filename')

def main():
	try:
		opts, args = getopt.getopt(sys.argv[1:], 'hp:vi:o:f:', ['help', 'gazePosition =', 'visualize', 'inputFile =', 'outputDir =', 'fragmentSize ='])
	except getopt.GetoptError as err:
		print(str(err))
		usage()
		sys.exit(2)

	# Image to foveate:
	image_name = 'castle.jpg'
	image_loc = '../images/'

	# Display foveated image:
	show_img = False

	# Save foveated image
	save_img = False

	# Image dimensions (leave as -1) for full image:
	W = -1
	H = -1

	# Fixation point (in terms of image dimensions) (-1 for center):
	fovx = -1
	fovy = -1

	# Fragment size (square pieces of image):
	fragmentW = 32
	fragmentH = 32

	# Launch CUDA kernel with this many threads per block (affects execution speed):
	threadsPerBlock = 256


	for o, a in opts:
		if o in ['-h', '--help']:
			usage()
			sys.exit(2)
		if o in ['-v', '--visualize']:
			show_img = True
		if o in ['-p', '--gazePosition']:
			fovy, fovx = tuple([float(x) for x in a.split(',')])
		if o in ['-i', '--inputFile']:
			image_name = a
		if o in ['-o', '--outputDir']:
			outputDir = "../" + a
			save_img = True
		if o in ['-f', '--fragmentSize']:
			fragmentW, fragmentH = tuple([int(x) for x in a.split(',')])

	if fragmentW <= 8 and fragmentH <= 8:
		threadsPerBlock = 128

	img = load_image(image_loc, image_name, W, H)
	H = img.shape[0]
	W = img.shape[1]

	if fovx == -1 and fovy == -1:
		fovx = W/2
		fovy = H/2

	minDist = fragmentW/2
	maxDist = np.sqrt((W/2)**2 + (H/2)**2)
	dist_l, num_regions = make_distances(minDist, maxDist, fragmentW, fragmentH)

	sigma_l = make_sigmas(dist_l, maxDist)
	l_ker_size = calc_ker_size(sigma_l[-1])
	S = int((l_ker_size-1)/2)

	paddedW = np.int32(W + 2*S)
	paddedH = np.int32(H + 2*S)
	img_padded = replication_pad(img, S)
	img_gpu = transfer_input_img(img_padded)

	width_shift, height_shift, plus_blocks_w, plus_blocks_h = calc_shift(fovx, fovy, fragmentW, fragmentH, W, H)
	gridW = math.floor(W/fragmentW) + plus_blocks_w
	gridH = math.floor(H/fragmentH) + plus_blocks_h
	blur_grid_gpu = make_blur_grid(gridW, gridH, fragmentW, fragmentH, fovx, fovy, width_shift, height_shift, dist_l)
	
	result_gpu = cuda.mem_alloc(img.nbytes)
	host_output = cuda.pagelocked_empty_like(img)

	mod = make_kernel(S, fragmentW, fragmentH, num_regions)
	transfer_filters(sigma_l, mod)

	run_foveation(mod, gridW, gridH, fragmentW, fragmentH, img_gpu, blur_grid_gpu, paddedW, paddedH, S, result_gpu, W, width_shift, height_shift, threadsPerBlock, fovx, fovy)
	transfer_result(host_output, result_gpu)

	if show_img:
		Image.fromarray(host_output[:,:,::-1], 'RGB').show()

	if save_img:
		makedirs(outputDir.rpartition('/')[0], exist_ok=True)
		cv.imwrite(outputDir, host_output)

if __name__ == "__main__":
	main()