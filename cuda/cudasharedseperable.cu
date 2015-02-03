#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sharedseperable.h"

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
	unsigned char *d_buffer;

	unsigned char h_filter[KERNEL_LENGTH];
	h_filter[0] = 1;
	h_filter[1] = 2;
	h_filter[2] = 1;

	printf("%d\n",KERNEL_LENGTH );

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



	cudaEventRecord(start, 0);

	cudaMalloc(&d_data, size);
	cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_results, size);
	cudaMalloc(&d_buffer, size);

	setConvolutionKernel(h_filter);


	for (i = 0; i < 100; i++)
    {

        convolutionRowsGPU(
            d_buffer,
            d_data,
            imageW,
            imageH
        );

        convolutionColumnsGPU(
            d_results,
            d_buffer,
            imageW,
            imageH
        );

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