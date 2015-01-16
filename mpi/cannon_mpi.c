#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "sequential_import_export.h"


int Nx,Ny,numtasks,totalSteps;
int width,height;

/* number of processes in each dimension */
int dims[2];


int parseCmdLineArgs(int argc, char **argv, int *dims, int myRank);
bool imageFilter(struct image* inputImage,struct image* outputImage);
int pixelFilter(struct image* inputImage,int i, int j);


int main(int argc, char* argv[]) 
{


	int rank,myGridRank,errCode, dest, source,tag,sendRequestCount,recvRequestCount,i,j;
	double* data=NULL;
	int periods[2];
	int coords[2];
	int nextCoords[2];
	char buffer[13];
	int dataSize;


	MPI_Comm comm_cart;
	MPI_Errhandler errHandler;
	MPI_Request sendRequestArr[8];
	MPI_Request recvRequestArr[8];

	int array_of_sizes[2];
	int array_of_subsizes[2];
	int array_of_starts[2];

	int sub_dims = 2;


	/* vertical (x) and horizontal (y) dimensions */


	/* dimensions for the cartesian grid */
	int ndims = 2;
	

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	/* Create a new communication and attach virtual topology (2D Cartesian) */
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm_cart);

	/* find out process rank in the grid */
	MPI_Comm_rank(comm_cart, &myGridRank);
	/* find out process coordinates in the grid */
	MPI_Cart_coords(comm_cart, myGridRank, ndims, coords);


	//creating the subarrays
	width = (int) ceil((double) (Nx/ dims[0])); // x
	height = (int) ceil((double) (Ny / dims[1])); // y

	dataSize = (width+2)*(height+2);

	data = calloc(dataSize, sizeof(unsigned char));
	results = calloc(dataSize, sizeof(unsigned char));

	FILE* inputImage;
	
	

	/* find process rank using the cartesian coordinates and assigh the appropiate
	part of image previously splitted 
	*/
	snprintf(buffer, 13, "%d", dims[1]*coords[0]+coords[1]);
	strcat(buffer,".raw");
	inputImage = fopen(buffer,"rb");
	
	

	


	fread(data,dataSize,1,inputImage);

	fclose(inputImage);

	



	//creating the subarrays






	array_of_sizes[0] = width + 2;
	array_of_sizes[1] = height + 2;


	array_of_starts[0] = 0;
	array_of_starts[1] = 0;



	/* slice to be sent to right neighbor (X-axis) */
	MPI_Datatype ROW;
	array_of_subsizes[0] = height;
	array_of_subsizes[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &ROW);
	MPI_Type_commit(&ROW);

	/* slice to be received from right neighbor (X-axis) */
	MPI_Datatype LINE;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = width;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &LINE);
	MPI_Type_commit(&LINE);

	/* slice to be sent to left neighbor (X-axis) */
	MPI_Datatype POINT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &POINT);
	MPI_Type_commit(&POINT);






	
	/* an MPI error, instead the error will be returned so we can handle it */
	MPI_Errhandler_get(comm_cart, &errHandler);
	if (errHandler == MPI_ERRORS_ARE_FATAL) {
		MPI_Errhandler_set(comm_cart, MPI_ERRORS_RETURN);
	}

	/* locate neighboring processes in the grid and initiate a send-receive
	 * operation for each neighbor found */

	/* process (x-1,y) */
	nextCoords[0] = coords[0] - 1;
	nextCoords[1] = coords[1];
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(1,1)], 1, ROW, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(1,0)], 1, ROW, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1];
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(1,width)], 1, ROW, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(1,width+1)], 1, ROW, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0];
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(1,1)], 1, LINE, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(0,1)], 1, LINE, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0];
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(height,1)], 1, LINE, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(height+1,1)], 1, LINE, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(height,width)], 1, POINT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(height+1,width+1)], 1, POINT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(1,width)], 1, POINT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(0,width+1)], 1, POINT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] - 1;
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(height,1)], 1, POINT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(height+1,0)], 1, POINT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] - 1;
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[offset(1,1)], 1, POINT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[offset(0,0)], 1, POINT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	for()
	{
		
		MPI_Startall(sendRequestCount,sendRequestArr);
		MPI_Startall(recvRequestCount,recvRequestArr);

		//calculate inner




		MPI_Waitall(recvRequestCount,recvRequestArr,MPI_STATUSES_IGNORE);


		//comptute outer







		MPI_Waitall(sendRequestCount,sendRequestArr,MPI_STATUSES_IGNORE);






	}





	MPI_Finalize();

}




int offset(int x,int y)
{


return x*(width+2)+y;


}






bool innerImageFilter(unsigned char *data,unsigned char* results)
{
	int i,j;

	bool differentPixels = false;
	for(j = 2; j < height; j++)
	{
		for(i = 2; i < width; i++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);
			if( results[offset(i,j)]!= data[offset(i,j)])
			{
				differentPixels = true;
			}
		}
	}

	return differentPixels;
}



unsigned char innerPixelFilter(unsigned char* data,int i, int j)
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
				outPixel += (data[offset((i-k),(j-l))]*filter[k+1][l+1]);
		}
	}
	return (unsigned char)(outPixel/16);
}







bool outerImageFilter(unsigned char* data, unsigned char* results,int* coords)
{


	int i,j;

	bool differentPixels = false;

	int flag[2];



//top

	if(coords[1]!=0)
	{

		i = 1;
		for(j = 1; j < width+1 ; j++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);
			
		}
	
	}

	else
	{
		i = 1;
		flag[0] = 0;
		flag[1] = -1;

		for(j = 1; j < width+1 ; j++)
		{
			results[offset(i,j)] = outerPixelFilter(data,i,j,flag);
			
		}

	}

//endTop

//startBottom


	if(coords[1]!= dims[0]-1)
	{
		i = height;

		for(j = 1; j < width+1; j++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);
			
		}
	}


	else
	{
		i = height;
		flag[0] = 0;
		flag[1] = 1;

		for(j = 1; j < width+1 ; j++)
		{
			results[offset(i,j)] = outerPixelFilter(data,i,j,flag);
			
		}

	}

//endBottom

//startLeft

	if(coords[0]!=0)
	{
		j = 1;

		for(i = 2; i < height; i++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);
			
		}
	}
	else
	{
		j = 1;
		flag[0] = 1;
		flag[1] = -1;

		for(j = 1; j < width+1 ; j++)
		{
			results[offset(i,j)] = outerPixelFilter(data,i,j,flag);
			
		}

	}

//endLeft


//startRight

	if(coords[0] != dims[1] - 1)
	{
		j = width;

		for(i = 2; i < height; i++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);
			
		}
	}

	else
	{
		j = width;
		flag[0] = 1;
		flag[1] = 1;

		for(j = 1; j < width+1 ; j++)
		{
			results[offset(i,j)] = outerPixelFilter(data,i,j,flag);
			
		}

	}



//endRight

}








unsigned char outerPixelFilter(unsigned char* data,int i, int j,int* flag)
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
				if(((flag[0]==0) && (k==flag[1])) || ((flag[0]==1) && (l==flag[1]))   )
				{
					outPixel += (data[offset((i),(j))]*filter[k+1][l+1]);	
				}

				else
				{
					outPixel += (data[offset((i-k),(j-l))]*filter[k+1][l+1]);
				}
				
			}
		}
	


	return (unsigned char)(outPixel/16);
}







int parseCmdLineArgs(int argc, char **argv, int *dims, int myRank) {
	if (argv[1] != NULL && strcmp(argv[1], "-nodes") == 0) {
		if (argv[2] != NULL && argv[3] != NULL) {
			Nx = atoi(argv[2]);
			Ny = atoi(argv[3]);
		} else {
			if (myRank == 0) {
				printf(
						"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
								" [-nodes <Nx> <Ny> -procs <i> <j> -steps <n> ]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> ]\n\n");
		}
		return 1;
	}

	/* allocate processes to each dimension. */
	if (argv[4] != NULL && strcmp(argv[4], "-procs") == 0) {
		if (argv[5] != NULL && argv[6] != NULL) {
			dims[0] = (int) atoi(argv[5]);
			dims[1] = (int) atoi(argv[6]);
		} else {
			if (myRank == 0) {
				printf(
						"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
								" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> ]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> ]\n\n");
		}
		return 1;
	}

	/* Grid of processes size must equal total number of processes */
	if (dims[0] * dims[1] != numtasks) {
		if (myRank == 0) {
			printf("\nProcessing grid size must equal total number of processes"
					" (np = i*j).\n\n");
		}
		return 1;
	}

	/* specify number of iterations */
	if (argv[7] != NULL && strcmp(argv[7], "-steps") == 0) {
		if (argv[8] != NULL) {
			totalSteps = atoi(argv[8]);
		} else {
			if (myRank == 0) {
				printf(
						"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
								" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> ]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny> <Nz> -procs <i> <j> -steps <n> ]\n\n");
		}
		return 1;
	}

	return 0;
}
