#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 18
#define FILTER_LENGTH 9
#define FILTER_RADIUS 1

__constant__ unsigned char c_Filter[FILTER_LENGTH];

extern "C" void setFilter(unsigned char *h_Filter)
{
    cudaMemcpyToSymbol(c_Filter, h_Filter, FILTER_LENGTH * sizeof(unsigned char));
}

__device__ int dOffset(int i, int j,int imageW) {
	return i*imageW + j;
}

__device__ int fOffset(int i, int j) {
	return i*(2*FILTER_RADIUS + 1) + j;
}

__device__ int lOffset(int i, int j) {
	return i*BLOCKSIZE_X+ j  ;
}

__global__ void filter(unsigned char* d_data,unsigned char* d_results,int imageW,int imageH)
{

	int gl_x = blockIdx.x * blockDim.x + threadIdx.x;
	int gl_y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ unsigned char s_data[BLOCKSIZE_Y + 2*FILTER_RADIUS][BLOCKSIZE_X + 2*FILTER_RADIUS];

	s_data[threadIdx.y + FILTER_RADIUS][threadIdx.x + FILTER_RADIUS] = d_data[dOffset(gl_y,gl_x,imageW)];

	/* right */
	if(threadIdx.x == 0 && gl_x != 0)
		s_data[threadIdx.y + FILTER_RADIUS][threadIdx.x] = d_data[dOffset(gl_y ,gl_x - FILTER_RADIUS,imageW)];

	/* left */
	if(threadIdx.x == blockDim.x - 1  && gl_x != imageW - 1)
		s_data[threadIdx.y + FILTER_RADIUS][threadIdx.x + 2*FILTER_RADIUS] = d_data[dOffset(gl_y ,gl_x + FILTER_RADIUS,imageW)];

	/* top */
	if(threadIdx.y == 0 && gl_y != 0)
		s_data[threadIdx.y][threadIdx.x + FILTER_RADIUS] = d_data[dOffset(gl_y - FILTER_RADIUS ,gl_x,imageW)];

 	/* bottom */
	if(threadIdx.y == blockDim.y -1 && gl_y != imageH - 1)
		s_data[threadIdx.y + 2*FILTER_RADIUS][threadIdx.x+ FILTER_RADIUS] = d_data[dOffset(gl_y+ FILTER_RADIUS ,gl_x,imageW)];

	/* top left */
	if(threadIdx.x == 0 && gl_x != 0 && threadIdx.y == 0 && gl_y!= 0)
		s_data[threadIdx.y][threadIdx.x] = d_data[dOffset(gl_y -FILTER_RADIUS,gl_x- FILTER_RADIUS,imageW)];

	/* bottom left */
	if(threadIdx.x == 0 && gl_x != 0 && threadIdx.y == blockDim.y - 1 && gl_y!= imageH - 1)
		s_data[threadIdx.y + 2*FILTER_RADIUS][threadIdx.x] = d_data[dOffset(gl_y +FILTER_RADIUS,gl_x- FILTER_RADIUS,imageW)];

	/* top right */
	if(threadIdx.x == blockDim.x - 1 && gl_x != imageW - 1 && threadIdx.y == 0 && gl_y!= 0)
		s_data[threadIdx.y][threadIdx.x+ 2*FILTER_RADIUS] = d_data[dOffset(gl_y -FILTER_RADIUS,gl_x+ FILTER_RADIUS,imageW)];

	/* bottom right*/
	if(threadIdx.x == blockDim.x - 1 && gl_x != imageW - 1 && threadIdx.y == blockDim.y - 1 && gl_y!= imageH - 1)
		s_data[threadIdx.y + 2*FILTER_RADIUS][threadIdx.x+ 2*FILTER_RADIUS] = d_data[dOffset(gl_y +FILTER_RADIUS,gl_x+ FILTER_RADIUS,imageW)];

	__syncthreads();

	int k,l;
	int outPixel = 0;
	if(gl_x < imageW && gl_y < imageH)
	{
		for(k=-1;k<=1;k++)
		{
			for(l=-1;l<=1;l++)
			{
				if ( (gl_y+k)>=0 && (gl_y+k)<imageH && (gl_x+l)>=0 && (gl_x+l)<imageW )
				{
					outPixel += s_data[threadIdx.y+FILTER_RADIUS+k][threadIdx.x+FILTER_RADIUS+l] * c_Filter[fOffset(k+1,l+1)];
				}
				else
				{
					outPixel += s_data[threadIdx.y+FILTER_RADIUS][threadIdx.x+FILTER_RADIUS] * c_Filter[fOffset(k+1,l+1)];
				}
			}
		}
		d_results[dOffset(gl_y,gl_x,imageW)] = (unsigned char)(outPixel/16);
	}	
	
	//d_results[dOffset(gl_y,gl_x,imageW)] = s_data[threadIdx.y + FILTER_RADIUS][threadIdx.x + FILTER_RADIUS];
}

void swap(unsigned char **d_data,unsigned char **d_results)
{
	unsigned char* temp = *d_data;
	*d_data = *d_results;
	*d_results = temp;
}

int main()
{
	int size,i,imageW,imageH;
	unsigned char *h_data;
	unsigned char *h_results;

	unsigned char *d_data;
	unsigned char *d_results;

	unsigned char h_filter[9];
	h_filter[0] = 1;
	h_filter[1] = 2;
	h_filter[2] = 1;
	h_filter[3] = 2;
	h_filter[4] = 4;
	h_filter[5] = 2;
	h_filter[6] = 1;
	h_filter[7] = 2;
	h_filter[8] = 1;

	imageW = 1920;
	imageH = 2520;
	size = imageW* imageH;

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	h_data =(unsigned char*)malloc(size);
	h_results =(unsigned char*)malloc(size);

	FILE* inputImage;
	inputImage = fopen("../image.raw","rb");
	fread(h_data,size,1,inputImage);
	fclose(inputImage);


	dim3 blockSize(BLOCKSIZE_X , BLOCKSIZE_Y);
	int numBlocks_X = imageW / BLOCKSIZE_X;
	int numBlocks_Y = imageH / BLOCKSIZE_Y;


	dim3 gridSize(numBlocks_X, numBlocks_Y);

	printf("blocks x %d blocks y %d\n",gridSize.x,gridSize.y );
	printf("blocks x %d blocks y %d\n",blockSize.x,blockSize.y );


	cudaEventRecord(start, 0);

	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_results, size);
	setFilter(h_filter);


	for(i = 0; i < 100; i++)
	{
		filter<<<gridSize,blockSize>>>(d_data,d_results,imageW,imageH);
		swap(&d_data,&d_results);
	}

	

	cudaMemcpy(h_results, d_results, size, cudaMemcpyDeviceToHost);
	cudaFree(d_results);
	cudaFree(d_data);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	FILE* outputImage;
	outputImage = fopen("out.raw","w+");
	fwrite(h_results,size,1,outputImage);
	fclose(outputImage);

	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);

	return 0;
}