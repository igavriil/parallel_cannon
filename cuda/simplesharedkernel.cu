#include "simplesharedkernel.h"

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
	return i*BLOCKDIM_X+ j  ;
}

__global__ void filter(unsigned char* d_data,unsigned char* d_results,int imageW,int imageH)
{

	int gl_x = blockIdx.x * blockDim.x + threadIdx.x;
	int gl_y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ unsigned char s_data[BLOCKDIM_Y + 2*FILTER_RADIUS][BLOCKDIM_X + 2*FILTER_RADIUS];

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