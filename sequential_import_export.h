#ifndef IMPORT_EXPORT_H_
#define IMPORT_EXPORT_H_

struct imageSize
{
	int width;
	int height;
};

struct image
{
	unsigned short** imageArray;
	struct imageSize* imageSize;
};

/**
 *	Create basic functionality for file managment
 *	Import text image into a 2 dimensional array requires first finding image size
 *	Export array to a well printed text files
 *	File names are passed as parameters in the functions
 */

struct imageSize* findImageSize(char* fileName);
struct image* imageImport(char* fileName);
void imageExport(struct image* inputImage,char* fileName);
struct image* initializeImage(struct imageSize* imageSize);
void copyImage(struct image* fromImage,struct image* toImage);
void swapImage(struct image** fromImage,struct image** toImage);



#endif