#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char *argv[])
{
	int Nx,Ny;
	int i,j,x,y;
	char buffer[13];

	if( argc != 3)
	{
		printf("expecting two arguments\n");
	}
	
	Nx = atoi(argv[1]);
	Ny = atoi(argv[2]);


	unsigned char Image[4838400];

	FILE* inputImage;
	FILE* outputImage;
 
	inputImage = fopen("../waterfall_grey_1920_2520.raw","rb");
	fread(Image,4838400,1,inputImage);

	fclose(inputImage);
	

	for(x=0;x < Nx; x++)
	{
		for(y=0;y < Ny; y++)
		{
   			snprintf(buffer, 13, "%d", Ny*x+y );
			strcat(buffer,".raw");

			outputImage = fopen(buffer,"w+");
			
			for(j =0; j<(1920/Nx)+2; j++)
			{
				putc(0,outputImage);
			}
			for(i=y*2520/Ny;i<(y+1)*2520/Ny;i++)
			{
				for(j=x*1920/Nx;j<(x+1)*1920/Nx;j++)
				{
					if(j == x*1920/Nx)
					{
						putc(0,outputImage);
					}
					putc(Image[1920*i+j],outputImage);
					if(j ==  ((x+1)*1920/Nx)-1 )
					{
						putc(0,outputImage);
					}
				}
			}
			for(j =0; j<(1920/Nx)+2; j++)
			{
				putc(0,outputImage);
			}
			fclose(outputImage);
		}
	}
}