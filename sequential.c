#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "sequential_import_export.h"

int pixelFilter(struct image* inputImage,int i, int j);
bool imageFilter(struct image* inputImage,struct image* outputImage);



int main()
{
	
	struct image* inputImage = imageImport("waterfall_grey_1920_2520.txt");
	struct image* outputImage = initializeImage(inputImage->imageSize);

	int c;
	for(c=0;c<500;c++)
	{
		imageFilter(inputImage,outputImage);
		swapImage(&outputImage,&inputImage);
	}

	imageExport(outputImage,"test_out.txt");

}

bool imageFilter(struct image* inputImage,struct image* outputImage)
{
	int i,j;

	bool differentPixels = false;
	for(i=0; i < inputImage->imageSize->height; i++)
	{
		for(j=0; j < inputImage->imageSize->width; j++)
		{
			outputImage->imageArray[i][j] = pixelFilter(inputImage,i,j);
			if(outputImage->imageArray[i][j] != inputImage->imageArray[i][j])
			{
				differentPixels = true;
			}
		}
	}

	return differentPixels;
}


int pixelFilter(struct image* inputImage,int i, int j)
{

	int filter[3][3];
	
	filter[0][0] = 1;
	filter[0][1] = 2;
	filter[0][2] = 1;
	filter[1][0] = 2;
	filter[1][1] = 4;
	filter[1][2] = 2;
	filter[2][0] = 1;
	filter[2][1] = 2;
	filter[2][2] = 1;


	int k,l;
	int outPixel = 0;

	for(k=-1;k<=1;k++)
	{
		for(l=-1;l<=1;l++)
		{
			if ( (i-k)>=0 && (i-k)<inputImage->imageSize->height && (j-l)>=0 && (j-l)<inputImage->imageSize->width )
			{
				outPixel += (inputImage->imageArray[i-k][j-l]*filter[k+1][l+1]);
			}
			else
			{
				outPixel += (inputImage->imageArray[i][j]*filter[k+1][l+1]);
			}
		}
	}
	return (unsigned short)(outPixel/16);
}
