#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "sequential_import_export.h"

struct imageSize* findImageSize(char* fileName)
{
	FILE *filePointer;
	struct imageSize* size;
	int width, height;
	char c;

	filePointer = NULL;

	if((filePointer = fopen(fileName,"r"))!=NULL)
	{
		width = 1;
 		height = 0;
		printf("\nFile Opened Succesfull\n");
		
		do{
			c = fgetc(filePointer);
			if(c =='\t' && height == 1)
				width+=1;
			if(c == '\n')
				height+=1;
		}while(!feof(filePointer));
		printf("width: %d, height: %d\n",width,height);
		
		size = (struct imageSize*)malloc(sizeof(struct imageSize));
		size->width = width;
		size->height = height;

		fclose(filePointer);

		return size;
	}
	else
	{
		printf("\nCannot open file \n");
		printf(strerror(errno));
		printf("\n");

		return NULL;
	}
}


struct image* imageImport(char* fileName)
{
	int i,j,k,width,height,scaned;
	unsigned short value;
    FILE *filePointer;
	struct image* inputImage;
	struct imageSize* size;
	unsigned short **array;

	filePointer = NULL;
	if((size = findImageSize(fileName))!= NULL)
	{
		filePointer = fopen(fileName , "r");
		printf("\nCreating the array from the file\n");
		array = (unsigned short **)malloc((size->height)*sizeof(unsigned short *));
		for(i=0; i<(size->height); i++)
		{
			*(array+i) = (unsigned short *)malloc((size->width)*sizeof(unsigned short));
			for(j=0; j<(size->width); j++)
			{
				
				fscanf(filePointer,"%d",&scaned);
				value = (unsigned short)scaned;
				*(*(array+i)+j) = value;
			}
		}

		inputImage = (struct image*)malloc(sizeof(struct image));
		inputImage->imageArray = array;
		inputImage->imageSize = size;

		fclose(filePointer);

		return inputImage;
	}
	else
	{
		printf("\nCannot open file \n");
		printf(strerror(errno));
		printf("\n");

		return NULL;
	}
}




void imageExport(struct image* inputImage,char* fileName)
{
	int i,j;
	FILE *filePointer;

	if((filePointer = fopen(fileName, "w"))!=NULL)
	{
		for(i=0; i<inputImage->imageSize->height; i++)
		{
			for(j=0; j<inputImage->imageSize->width; j++)
			{
				if(j != inputImage->imageSize->width - 1)
					fprintf(filePointer,"%hu\t",*(*(inputImage->imageArray+i)+j) );
				else
					fprintf(filePointer,"%hu",*(*(inputImage->imageArray+i)+j));
			}
			if(i != inputImage->imageSize->height - 1)
				fprintf(filePointer,"\n");
		}
		fclose(filePointer);
	}
	else
	{
		printf("\nCannot open file \n");
		printf(strerror(errno));
		printf("\n");
	}
}


struct image* initializeImage(struct imageSize* imageSize)
{
	struct image* outImage;
	unsigned short **array;
	int i,j;


	array = (unsigned short **)malloc((imageSize->height)*sizeof(unsigned short *));
	for(i=0; i<(imageSize->height); i++)
	{
		*(array+i) = (unsigned short *)malloc((imageSize->width)*sizeof(unsigned short));
		for(j=0; j<(imageSize->width); j++)
		{
			*(*(array+i)+j) = 0;
		}
	}

	outImage = (struct image*)malloc(sizeof(struct image));
	outImage->imageArray = array;
	outImage->imageSize = imageSize;

	return outImage;
	
}

void copyImage(struct image* fromImage,struct image* toImage)
{
	int i,j;
	for(i=0; i<(fromImage->imageSize->height); i++)
	{
		for(j=0; j<(fromImage->imageSize->width); j++)
		{
			toImage->imageArray[i][j] = fromImage->imageArray[i][j];
		}
	}
}

void swapImage(struct image** fromImage,struct image** toImage)
{
	struct image* temp = *fromImage;
	*fromImage = *toImage;
	*toImage = temp;
}
