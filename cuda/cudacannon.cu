#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void filter(unsigned char* d_data,unsigned char* d_results,int* d_filter)
{
	int k,l;
	

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int outPixel = 0;

	if(id < 1920*2520 - 1)
	{

		for(k=-1;k<=1;k++)
		{
			for(l=-1;l<=1;l++)
			{
				if ( (id/1920-k)>=0 && (id/1920-k)<2520 && (id%1920-l)>=0 && (id%1920-l)<1920 )
				{
					outPixel += (d_data[(id/1920+k)*(1920)+(id%1920+l)]*d_filter[3*(k+1)+(l+1)]);
				}
				else
				{
					outPixel += (d_data[id]*d_filter[3*(k+1)+(l+1)]);
				}
			}
		}

		d_results[id] = (unsigned char)(outPixel/16);
		__syncthreads();
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
	int size,i;
	unsigned char *h_data;
	unsigned char *h_results;

	unsigned char *d_data;
	unsigned char *d_results;
	int *d_filter;

	int h_filter[9];
	h_filter[0] = 1;
	h_filter[1] = 2;
	h_filter[2] = 1;
	h_filter[3] = 2;
	h_filter[4] = 4;
	h_filter[5] = 2;
	h_filter[6] = 1;
	h_filter[7] = 2;
	h_filter[8] = 1;

	size = 2520*1920;

	h_data =(unsigned char*)malloc(size);
	h_results =(unsigned char*)malloc(size);

	FILE* inputImage;
	inputImage = fopen("../image.raw","rb");
	fread(h_data,size,1,inputImage);
	fclose(inputImage);

	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_results, size);
	cudaMalloc(&d_filter,9*sizeof(int));
	cudaMemcpy(d_filter, h_filter, 9*sizeof(int), cudaMemcpyHostToDevice);


	for(i = 0; i < 300; i++ )
	{
		filter<<<7560,640>>>(d_data,d_results,d_filter);
		 swap(&d_data,&d_results);
	}

	cudaMemcpy(h_results, d_results, size, cudaMemcpyDeviceToHost);

	printf("%d \n",1/1920 );
	printf("%d \n",1%1920 );
	printf("hello\n");


	FILE* outputImage;
	outputImage = fopen("out.raw","w+");
	fwrite(h_results,size,1,outputImage);
	fclose(outputImage);
}