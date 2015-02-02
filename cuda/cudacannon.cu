#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 18
#define FILTER_LENGTH 9



__constant__ unsigned char c_Filter[FILTER_LENGTH];

extern "C" void setFilter(unsigned char *h_Filter)
{
    cudaMemcpyToSymbol(c_Filter, h_Filter, FILTER_LENGTH * sizeof(unsigned char));
}

__device__ int dOffset(int x, int y,int imageW) {
	return x*imageW + y;
}

__device__ int fOffset(int x, int y,int filterW) {
	return x*filterW + y;
}

__global__ void filter(unsigned char* d_data,unsigned char* d_results,int imageW,int imageH)
{
	int k,l;
	const int gi = blockIdx.y * blockDim.y + threadIdx.y;
	const int gj = blockIdx.x * blockDim.x + threadIdx.x;

	int outPixel = 0;

	if(gi < imageH && gj < imageW)
	{
		for(k=-1;k<=1;k++)
		{
			for(l=-1;l<=1;l++)
			{
				if ( (gi+k)>=0 && (gi+k)<imageH && (gj+l)>=0 && (gj+l)<imageW )
				{
					outPixel += d_data[dOffset(gi+k,gj+l,imageW)] * c_Filter[fOffset(k+1,l+1,3)];
				}
				else
				{
					outPixel += d_data[dOffset(gi,gj,imageW)] * c_Filter[fOffset(k+1,l+1,3)];
				}
			}
		}

		d_results[dOffset(gi,gj,imageW)] = (unsigned char)(outPixel/16);
	}
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


	dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);
	int numBlocks_X = imageW / BLOCKSIZE_X;
	int numBlocks_Y = imageH / BLOCKSIZE_Y;
	printf("blocks x %d blocks y %d\n",numBlocks_X,numBlocks_Y );
	dim3 gridSize(numBlocks_X, numBlocks_Y);

	cudaEventRecord(start, 0);

	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_results, size);
	setFilter(h_filter);


	for(i = 0; i < 100; i++ )
	{
		filter<<<gridSize,blockSize>>>(d_data,d_results,imageW,imageH);
		swap(&d_data,&d_results);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(h_results, d_results, size, cudaMemcpyDeviceToHost);
	cudaFree(d_results);
	cudaFree(d_data);


	FILE* outputImage;
	outputImage = fopen("out.raw","w+");
	fwrite(h_results,size,1,outputImage);
	fclose(outputImage);

	cudaEventElapsedTime(&time, start, stop);
	printf ("Time for the kernel: %f ms\n", time);

	return 0;
} 