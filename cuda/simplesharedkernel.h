#ifndef SIMPLESHAREDKERNEL_H
#define SIMPLESHAREDKERNEL_H

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 18
#define FILTER_LENGTH 9
#define FILTER_RADIUS 1

extern "C" void setFilter(unsigned char *h_Filter);

__device__ int dOffset(int i, int j,int imageW);

__device__ int fOffset(int i, int j);

__device__ int lOffset(int i, int j) ;

__global__ void filter(unsigned char* d_data,unsigned char* d_results,int imageW,int imageH);	

#endif