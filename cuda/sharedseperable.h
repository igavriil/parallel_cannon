#ifndef SHAREDSEPERABLE_H
#define SHAREDSEPERABLE_H

#define KERNEL_RADIUS 1
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

extern "C" void setConvolutionKernel(unsigned char *h_Kernel);

extern "C" void convolutionRowsGPU(
    unsigned char *d_Dst,
    unsigned char *d_Src,
    int imageW,
    int imageH
);

extern "C" void convolutionColumnsGPU(
    unsigned char *d_Dst,
    unsigned char *d_Src,
    int imageW,
    int imageH
);



#endif