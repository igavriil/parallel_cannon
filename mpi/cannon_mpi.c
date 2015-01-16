#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* number of processes */
int numtasks
/* image-grid dimensions */
int Nx,Ny,totalSteps;
/* the part of the grid allocate in each process */
int width,height;
/* number of processes in each dimension */
int dims[2];

//////////////////////////////////////////////////////////not myRank but rank///
int parseCmdROWArgs(int argc, char **argv, int *dims, int myRank);
bool imageFilter(struct image* inputImage,struct image* outputImage);
int pixelFilter(struct image* inputImage,int i, int j);


int main(int argc, char* argv[]) 
{
	/* process rank in MPI_COMM_WORLD communicator */
	int rank;
	/* process rank in Cartesian communicator */
	int myGridRank;
	/* source and destination rank for communication*/
	int source,dest;
	int errCode, sendRequestCount,recvRequestCount,i,j;
	/* tag for messages */
	int tag = 0;
	double* data=NULL;
	int periods[2];
	int coords[2];
	int nextCoords[2];
	char buffer[13];
	/* dayasize for each process */
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
	/* start MPI */
	MPI_Init(&argc, &argv);
	/* get number of processes used */
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	/* get process rank in MPI_COMM_WORLD communicator */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* if uncorrect arguments are passed finalize MPI */
	if (parseCmdROWArgs(argc, argv, dims, myRank) == 1) {
		MPI_Finalize();
		return 1;
	}

	/* [width, depth] of problem's grid assigned to each
	 * process, calculated by the total number of nodes in each dimension divided by
	 * the number of processes assigned to that dimension.
	**/
	width = (int) ceil((double) (Nx/ dims[0]));
	height = (int) ceil((double) (Ny / dims[1]));

	/* print settings defined for the execution */
	if (myRank == 0) {
		printf("\nProblem grid: %dx%dx%d.\n", Nx, Ny, Nz);
		printf("P = %d processes.\n", p);
		printf("Sub-grid / process: %dx%dx%d.\n", width, depth, height);
		printf("\nC = %d iterations.\n", totalSteps);
		printf("Reduction performed every %d iteration(s).\n", reduceFreq);
	}

	/* There's not communication wraparound in the model. Outter image pixels are
	 * calulated in a different manner than inner so there is no cyclic boundaries
	 * in either x and y dimension
	**/
	periods[0] = 0;
	periods[1] = 0;

	/* Create a new communicator and attach virtual topology (2D Cartesian)
	 * allowing process reordering 
	**/
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm_cart);
	/* get process rank in grid - Cartesian communicator */
	MPI_Comm_rank(comm_cart, &myGridRank);
	/* get process coordinates in grid - Cartesian communicator */
	MPI_Cart_coords(comm_cart, myGridRank, ndims, coords);

	/* Calculate datasize for the process adding padding for halo points.
	 * Padding needed for the algorithm is one pixel at each outter pixel
	**/
	dataSize = (width+2)*(height+2);
	/* Allocate empty space for data to be read and output data calculated 
	 * from the filtering algorithm. Data are stored as 1D array in the 
	 * memory
	**/ 
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

	



	/* Create three datatypes by defining sections of the original array
	 * Our array is treated as a 2 dimensional array and we are slicing 
	 * rows, columns and points having as starting position the initial position (0,0)
	 * of our array
	**/
	/* define the width and height of the array represented as 2 dimensional array
	 * halo points added to this representation
	**/
	array_of_sizes[0] = width + 2;
	array_of_sizes[1] = height + 2;
	/* define the starting position - address for the slices to be cutted. We set 
	 * those to (0,0) in order to generalize this datatypes
	**/
	array_of_starts[0] = 0;
	array_of_starts[1] = 0;

	/* slicing a COLUMN from the 2 dimensional array */
	MPI_Datatype COLUMN;
	/* a column consists of 2D array with [1] column and [height] rows */
	array_of_subsizes[0] = height;
	array_of_subsizes[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &COLUMN);
	/* assing the subarray to the datatype */
	MPI_Type_commit(&COLUMN);

	/* slicing a ROW from the 2 dimensional array */
	MPI_Datatype ROW;
	/* a row consists of 2D array with [1] row and [width] columns */
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = width;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &ROW);
	/* assing the subarray to the datatype */
	MPI_Type_commit(&ROW);

	/* slicing a POINT from the 2 dimensional array */
	MPI_Datatype POINT;
	/* a point consists of 2D array with [1] column and [1] rows*/
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &POINT);
	/* assing the subarray to the datatype */
	MPI_Type_commit(&POINT);

	/* The predefined default error handler, which is
	 * MPI_ERRORS_ARE_FATAL, for a newly created communicator or
	 * for MPI_COMM_WORLD is to abort the  whole parallel program
	 * as soon as any MPI error is detected. By setting the error handler to
	 * MPI_ERRORS_RETURN  the program will no longer abort on having detected
	 * an MPI error, instead the error will be returned so we can handle it. 
	**/
	MPI_Errhandler_get(comm_cart, &errHandler);
	if (errHandler == MPI_ERRORS_ARE_FATAL) {
		MPI_Errhandler_set(comm_cart, MPI_ERRORS_RETURN);
	}
	/* Locate neighboring processes in the grid and initiate a send-receive
	 * operation for each neighbor found 
	**/

	/* Left Neighboor - process (x-1,y) in the Cartesian! topology */
	nextCoords[0] = coords[0] - 1;
	nextCoords[1] = coords[1];
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		/* take the column starting at ([1]-row,[1]-column) address and send */
		MPI_Send_init(&data[offset(1,1)], 1, COLUMN, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		/* set starting position at ([1]-row,[0]-column) and place the received column */
		MPI_Recv_init(&data[offset(1,0)], 1, COLUMN, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* Right Neighboor - process (x-1,y) in the Cartesian! topology */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1];
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		/* take the column starting at ([1]-row,[width]-column) address and send */
		MPI_Send_init(&data[offset(1,width)], 1, COLUMN, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		/* set starting position at ([1]-row,[width+1]-column) and place the received column */
		MPI_Recv_init(&data[offset(1,width+1)], 1, COLUMN, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* Top Neighboor - process (x,y-1) in the Cartesian! topology */
	nextCoords[0] = coords[0];
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		/* take the row starting at ([1]-row,[1]-column) address and send */
		MPI_Send_init(&data[offset(1,1)], 1, ROW, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		/* set starting position at ([0]-row,[1]-column) and place the received row */
		MPI_Recv_init(&data[offset(0,1)], 1, ROW, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* Bottom Neighboor - process (x,y+1) in the Cartesian! topology */
	nextCoords[0] = coords[0];
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		/* take the row starting at ([height]-row,[1]-column) address and send */
		MPI_Send_init(&data[offset(height,1)], 1, ROW, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		/* set starting position at ([height+1]-row,[1]-column) and place the received row */
		MPI_Recv_init(&data[offset(height+1,1)], 1, ROW, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* Bottom-Right Neighboor - process (x+1,y+1) in the Cartesian! topology */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		/* take the point starting at ([height]-row,[width]-column) address and send */
		MPI_Send_init(&data[offset(height,width)], 1, POINT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		/* set starting position at ([height+1]-row,[width+1]-column) and place the received point */
		MPI_Recv_init(&data[offset(height+1,width+1)], 1, POINT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* Top-Right Neighboor - process (x+1,y-1) in the Cartesian! topology */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		/* take the point starting at ([1]-row,[width]-column) address and send */
		MPI_Send_init(&data[offset(1,width)], 1, POINT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		/* set starting position at ([0]-row,[width+1]-column) and place the received point */
		MPI_Recv_init(&data[offset(0,width+1)], 1, POINT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* Bottom-Left Neighboor - process (x-1,y-1) in the Cartesian! topology */
	nextCoords[0] = coords[0] - 1;
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		/* take the point starting at ([height]-row,[1]-column) address and send */
		MPI_Send_init(&data[offset(height,1)], 1, POINT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		/* set starting position at ([height+1]-row,[0]-column) and place the received point */
		MPI_Recv_init(&data[offset(height+1,0)], 1, POINT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* Top-Left Neighboor - process (x-1,y-1) in the Cartesian! topology */
	nextCoords[0] = coords[0] - 1;
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		/* take the point starting at ([1]-row,[1]-column) address and send */
		MPI_Send_init(&data[offset(1,1)], 1, POINT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		/* set starting position at ([0]-row,[0]-column) and place the received point */
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



/* returns the offset of an element in a 2D array of dimensions (width,height) 
 * allocated in row major order
**/
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

	//begin top Left
	if(coords[1]==0 && coords[0]==0)
	{
		flag[0] = -1;
		flag[1] = -1;

		results[offset(1,1)] = cornerPixelFilter(data,1,1,flag);
	}
	//end top Left

	//begin top Right
	if(coords[1]==0 && coords[0]==dims[1] -1)
	{
		flag[0] = -1;
		flag[1] = 1;

		results[offset(1,width)] = cornerPixelFilter(data,1,width,flag);
	}
	//end top Right

	//begin bottom left
	if(coords[1]==dims[0] -1 && coords[0]==0)
	{
		flag[0] = 1;
		flag[1] = -1;

		results[offset(height,1)] = cornerPixelFilter(data,height,1,flag);
	}
	//end bottom left

	//begin bottom Right
	if(coords[1]==dims[0] -1&& coords[0]==dims[1] -1)
	{
		flag[0] = 1;
		flag[1] = 1;

		results[offset(height,width)] = cornerPixelFilter(data,height,width,flag);
	}77
	//end bottom Right






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
				
		0	}
		}
	


	return (unsigned char)(outPixel/16);
}

unsigned char cornerPixelFilter(unsigned char* data,int i, int j,int* flag)
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
				if(k==flag[0] && l==flag[1])
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






int parseCmdROWArgs(int argc, char **argv, int *dims, int myRank) {
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
