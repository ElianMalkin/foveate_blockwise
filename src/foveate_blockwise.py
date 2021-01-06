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


class FoveateBlockwise:

	def __init__(self, img_name, img_dir, frameW, frameH, fovx, 
				fovy, fragmentW=32, fragmentH=32, threadsPerBlock=256,
				e2=2.3, alpha=0.106, CT_0=0.0133, max_ecc=100, max_cyc_per_deg=30):

		self.img_name = img_name
		self.img_dir = img_dir
		self.W = frameW
		self.H = frameH
		self.fragmentW = fragmentW
		self.fragmentH = fragmentH
		self.threadsPerBlock = threadsPerBlock
		self.fovx = fovx
		self.fovy = fovy

		# Contrast Sensitivity Function parameters:
		self.e2 = e2
		self.alpha = alpha
		self.CT_0 = CT_0

		# Maximum equivalent eccentricity of image (human lateral FOV is 100 degrees):
		self.max_ecc = max_ecc

		# Maximum cycles per degree of the mapped retina (setting at 30 for human):
		self.max_cyc_per_deg = max_cyc_per_deg

	def load_image(self):
		img = cv.imread(self.img_dir + self.img_name)

		if img is None:
			sys.exit("Could not read the image.")

		if self.H == -1 and self.W == -1:
			pass
		elif self.W == -1:
			img = img[:self.H,:, :]
		elif self.H == -1:
			img = img[:,:self.W, :]
		else:
			img = img[:self.H,:self.W, :]

		self.H = img.shape[0]
		self.W = img.shape[1]

		if self.fovx == -1 and self.fovy == -1:
			self.fovx = self.W/2
			self.fovy = self.H/2

		# Distance at which blurring begins (set to first fragment after fovea fragment):
		self.minDist = self.fragmentW/2 

		# Half-diagonal of image:
		self.maxDist = np.sqrt((self.W/2)**2 + (self.H/2)**2)

		return np.ascontiguousarray(img)


	def make_distances(self, coef = 1.4):
		# Make list of distances denoting pooling region boundaries

		dist_l = [self.minDist]
		d = self.minDist + coef*self.fragmentW
		while d < self.maxDist:
			dist_l.append(d)
			d += coef*self.fragmentW

		self.num_regions = len(dist_l)

		# Number of distinct pooling regions (blurring steps) in the image:
		print("Number of pooling regions: " + str(self.num_regions))

		return dist_l


	def make_sigmas(self, dist_l):
		# Calculate standard deviations of Gaussian filters used at each pooling region

		sigma_l = []
		for i in range(len(dist_l)):
			# SET BLUR AT CENTER OF POOLING REGION
			if i == len(dist_l) - 1:
				pixDist = dist_l[i] + (self.maxDist - dist_l[i])/2
			else:
				pixDist = dist_l[i] + (dist_l[i+1] - dist_l[i])/2

			if pixDist > self.maxDist:
				pixDist = self.maxDist

			ecc = pixDist

			# cycles per degree at given pixel eccentricity:
			fc_deg = 1/(ecc/self.maxDist*self.max_ecc + self.e2) * (self.e2/self.alpha * np.log(1/self.CT_0)) 

			if fc_deg > self.max_cyc_per_deg:
				fc_deg = self.max_cyc_per_deg

			# equivalent cycles per pixel (standard deviation in frequency domain):
			sig_f = 0.5 * fc_deg/self.max_cyc_per_deg
		 
		 	# convert to pixel domain:
			sigma = 1/(2*np.pi*sig_f)

			sigma_l.append(sigma)

		l_f_size = calc_ker_size(sigma_l[-1])
		self.S = int((l_f_size-1)/2) # radius of image padding required to accommodate largest filter

		self.paddedW = np.int32(self.W + 2*self.S)
		self.paddedH = np.int32(self.H + 2*self.S)

		return sigma_l


	def calc_shift(self):
		# Calculates fragment shift to center a single fragment on the fixation point
		# Also calculates number of fragments to add to grid due to shift

		# Only shift left or up, so calculate fraction of fragment to shift by:

		r_w = np.remainder(self.fovx, self.fragmentW)/self.fragmentW
		if r_w <= 0.5:
			shift_w = 0.5 - r_w
		else:
			shift_w = 1.5 - r_w

		r_h = np.remainder(self.fovy, self.fragmentH)/self.fragmentH
		if r_h <= 0.5:
			shift_h = 0.5 - r_h
		else:
			shift_h = 1.5 - r_h

		# Shift in pixels:
		self.width_shift = np.int32(self.fragmentW * shift_w)
		self.height_shift = np.int32(self.fragmentH * shift_h)


		# Calculate number of fragments to add to the grid due to the shift:
		r_grid_w = np.remainder(self.W, self.fragmentW)/self.fragmentW
		if r_grid_w + shift_w <= 1:
			plus_frag_w = 1
		else:
			plus_frag_w = 2

		r_grid_h = np.remainder(self.H, self.fragmentH)/self.fragmentH
		if r_grid_h + shift_h <= 1:
			plus_frag_h = 1
		else:
			plus_frag_h = 2

		# Set grid dimensions:
		self.gridW = math.floor(self.W/self.fragmentW) + plus_frag_w
		self.gridH = math.floor(self.H/self.fragmentH) + plus_frag_h


	def make_blur_grid(self, dist_l):
		# Grid that stores blurring strengths for each fragment (indices into list of of Gaussian filter cumulative sizes)
		# This function simply uses the distances list for a circularly symmetric blur_grid
		# Can be adapted to use a non-symmetric blur strength distribution

		blur_grid = np.zeros((self.gridH, self.gridW), dtype="int32")

		for i in range(self.gridH):
			row = int(i*self.fragmentH + self.fragmentH/2 - self.height_shift)

			for j in range(self.gridW):
				col = int(j*self.fragmentW + self.fragmentW/2 - self.width_shift)

				distance = math.sqrt((self.fovx - col)**2 + (self.fovy - row)**2)
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


	def transfer_filters(self, sigma_l, mod):
		# Calculates Gaussian filter values and transfers them to GPU as a single list
		# Also transfers a list of their cumulative sizes for indexing

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


	def transfer_input_img(self, img_padded):
		# transfer image to be foveated to GPU

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


	def transfer_result(self, host_output, result_gpu):
		# transfer foveated image to host

		start = cuda.Event()
		end = cuda.Event()

		start.record()
		cuda.memcpy_dtoh(host_output, result_gpu)
		end.record()
		end.synchronize()
		millis = start.time_till(end)
		print("Transferring to CPU: " + str(millis) + " ms")


	def run_foveation(self, mod, img_gpu, blur_grid_gpu, result_gpu):
		func = mod.get_function("convolution")
		func.prepare(("P", "P", "P", "i", "i", "i", "i", "i", "i", "i", "i", "i", "i"))

		block=(self.threadsPerBlock, 1, 1)
		grid = (self.gridW, self.gridH)

		start=cuda.Event()
		end=cuda.Event()

		start.record() # start timing

		func.prepared_call(grid, block, img_gpu, blur_grid_gpu, result_gpu,
							self.paddedW, self.paddedH, np.int32(self.S), 
							np.int32(self.fragmentW), np.int32(self.fragmentH), 
							np.int32(self.W), np.int32(self.fovx), np.int32(self.fovy), 
							self.width_shift, self.height_shift)

		end.record() # end timing
		end.synchronize()
		millis = start.time_till(end)

		print("Time taken: " + str(millis) + " ms")


	def make_kernel(self):
		# CUDA kernel
		tileW = self.fragmentW + 2*self.S
		tileH = self.fragmentH + 2*self.S

		mod = SourceModule("""
		#include "stdint.h"

		const unsigned int MAX_FILTER_SIZE = 4000;

		const unsigned int NUM_REGIONS = """ + str(self.num_regions) + """;

		__device__ __constant__ float d_g[MAX_FILTER_SIZE];
		__device__ __constant__ int k_c[NUM_REGIONS+1];

		__global__ void convolution( const uint8_t *d_f, const unsigned int *blur_grid, uint8_t *d_h, 
									const unsigned int paddedW, const unsigned int paddedH, const int S,
									const unsigned int fragmentW, const unsigned int fragmentH, 
									const unsigned int W, const unsigned int fovx, const unsigned int fovy, 
									const int width_shift, const int height_shift)
		{



			const unsigned int tileW = fragmentW + 2 * S;

			const int blockStartCol = blockIdx.x * fragmentW + S - width_shift;
			const int blockStartRow = blockIdx.y * fragmentH + S - height_shift;

			const unsigned int blockEndCol = blockStartCol + fragmentW;
			const unsigned int blockEndRow = blockStartRow + fragmentH;

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
					unsigned int iterations = ceilf(__fdividef(fragmentW*fragmentH,numThreads));
					for( unsigned int iterNo = 0; iterNo < iterations; iterNo++ ) {
						unsigned int idx = numThreads*iterNo + threadIdx.x;
						unsigned int bRow = __fdividef(idx, fragmentW);
						unsigned int bCol = idx - bRow*fragmentW;
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
					__shared__ float intermData[""" + str(self.fragmentW*tileH) + """];
				
					unsigned int numThreads = blockDim.x;
					unsigned int wid = ker_size-1 + fragmentW;
					unsigned int hig = ker_size-1 + fragmentH;
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



					wid = fragmentW;
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
								unsigned int intermPixelPos = tRow * fragmentW + tCol - S;

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



					hig = fragmentH;
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

							unsigned int tilePixelPos = (4*tRow + S) * fragmentW + tCol - S;

							float tempSum = 0.0;
							float tempSum2 = 0.0;
							float tempSum3 = 0.0;
							float tempSum4 = 0.0;

							float old = intermData[ tilePixelPos - u * fragmentW];
							float old2 = intermData[ tilePixelPos + (- u + 1) * fragmentW];
							float old3 = intermData[ tilePixelPos + (- u + 2) * fragmentW];


							for( int i = -u; i <= u; i++ ) {
								float filtCoef = d_g[kernel_offset + i + u];
								tempSum += old * filtCoef;
								tempSum2 += old2 * filtCoef;
								tempSum3 += old3 * filtCoef;
								old = old2;
								old2 = old3;
								old3 = intermData[ tilePixelPos + (i+3) * fragmentW ];
								tempSum4 += old3 * filtCoef;


								//old = intermData[ tilePixelPos + (i+1) * fragmentW ];
								//tempSum2 += filtCoef * old;

								//int coefPos = ( i + u );
								//int tilePixelPosOffset = i * fragmentW;
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


def calc_ker_size(sig):
	# Calculates size of Gaussian filter given its standard deviation:
	s = int(1+2*np.sqrt(-2*sig**2*np.log(0.005)))
	if s % 2 == 0:
		s += 1
	return np.int32(s)

def replication_pad(img, pad_size):
	# Pads an image by replicating the edge pixels:
	S = pad_size
	output = cv.copyMakeBorder(img, S, S, S, S, cv.BORDER_REPLICATE)
	return output

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

	# Default image to foveate:
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
		threadsPerBlock = 128 # empirically faster 


	fov_inst = FoveateBlockwise(image_name, image_loc, W, H, fovx, fovy, 
								fragmentW, fragmentH, threadsPerBlock,
								max_ecc=100, max_cyc_per_deg=30)

	img = fov_inst.load_image()
	dist_l = fov_inst.make_distances()

	sigma_l = fov_inst.make_sigmas(dist_l)
	
	img_padded = replication_pad(img, fov_inst.S)
	img_gpu = fov_inst.transfer_input_img(img_padded)

	fov_inst.calc_shift()

	blur_grid_gpu = fov_inst.make_blur_grid(dist_l)

	result_gpu = cuda.mem_alloc(img.nbytes)
	host_output = cuda.pagelocked_empty_like(img)

	mod = fov_inst.make_kernel()
	fov_inst.transfer_filters(sigma_l, mod)

	fov_inst.run_foveation(mod, img_gpu, blur_grid_gpu, result_gpu)
	fov_inst.transfer_result(host_output, result_gpu)

	if show_img:
		Image.fromarray(host_output[:,:,::-1], 'RGB').show()

	if save_img:
		makedirs(outputDir.rpartition('/')[0], exist_ok=True)
		cv.imwrite(outputDir, host_output)


if __name__ == "__main__":
	main()