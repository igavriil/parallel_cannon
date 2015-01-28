#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

/* number of processes */
int numtasks;
/* image-grid dimensions */
int Nx,Ny;
/* the part of the grid allocate in each process */
int width,height;
/* number of processes in each dimension */
int dims[2];
/* width and height counters */
int i,j;
/* number of iterations */
int totalSteps; 
/* redunction frequence in steps */
int reduceFreq; 
/* filter array */
int filter[3][3];




int parseCmdLineArgs(int argc, char **argv, int *dims, int myRank);
int offset(int i,int j);
int innerImageFilter(unsigned char *data,unsigned char* results);
int outerImageFilter(unsigned char* data, unsigned char* results,int* coords);
unsigned char innerPixelFilter(unsigned char* data,int i, int j);
unsigned char outerPixelFilter(unsigned char* data,int i, int j,int* flag);
unsigned char cornerPixelFilter(unsigned char* data,int i, int j,int* flag);
void swapImage(unsigned char** data,unsigned char** results);
int breakf();

int main(int argc, char* argv[]) 
{
	/* process rank in MPI_COMM_WORLD communicator */
	int myRank;
	/* process rank in Cartesian communicator */
	int myGridRank;
	/* source and destination rank for communication*/
	int source,dest;
	/* tag for messages */
	int tag = 0;
	/* part of the dataset assigned for the process */
	unsigned char *data = NULL;
	/* filter calculation results for the process */
	unsigned char *results = NULL;
	/* determines whether a dimension has cyclic boundaries,
	 * meaning the 2 edge processes are connected  */
	int periods[2];
	/* process coordinates in the cartesian communicator */
	int coords[2];
	/* next process coordinates used for communication between neighboor processes*/
	int nextCoords[2];
	/* datasize for each process */
	int dataSize;
	/* computation steps */
	int steps = 0; 

	double t1,t2,Dt_local,Dt_total;

	int sf;
	int rf;

	char buffer[9];

	/* assign the filter values */
	filter[0][0] = 1;
	filter[0][1] = 2;
	filter[0][2] = 1;
	filter[1][0] = 2;
	filter[1][1] = 4;
	filter[1][2] = 2;
	filter[2][0] = 1;
	filter[2][1] = 2;
	filter[2][2] = 1;

	/* cartesian communicator */
	MPI_Comm comm_cart;
	/* Error handle for the communicator */
	MPI_Errhandler errHandler;
	/* array of identifiers for non-blocking recv and send */
	MPI_Request sendRequestArr[8],recvRequestArr[8];
  	/* offset for seeking into the file */
    MPI_Offset fileOffset;
    /* Statuses for reading file and Requests send/received */
    MPI_Status fileStatus;
	MPI_Status sendRequestStatus[8],recvRequestStatus[8];
	/* error coded returned while searching for neighboors */
	int errCode;
	/* count how many non-blocking recvs and sends have been submitted */
	int recvRequestCount = 0;
	int sendRequestCount = 0;
	/* array size for create_subarray function */
	int array_of_sizes[2];
	/* subarray size to be created from the initial array */
	int array_of_subsizes[2];
	/* address of the initial address where subarray cutting starts */
	int array_of_starts[2];
	/* dimensions of initial array and subarray to be produced */
	int sub_dims = 2;
	/* dimensions for the cartesian grid */
	int ndims = 2;
	/* start MPI */
	MPI_Init(&argc, &argv);
	/* get number of processes used */
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	/* get process rank in MPI_COMM_WORLD communicator */
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	/* if uncorrect arguments are passed finalize MPI */
	if (parseCmdLineArgs(argc, argv, dims, myRank) == 1) {
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
		printf("\nProblem grid: %dx%d.\n", Nx, Ny);
		printf("P = %d processes.\n", numtasks);
		printf("Sub-grid / process: %dx%d.\n", width, height);
		printf("Sub-grid datasize: %d\n",(width+2)*(height+2));
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

	
	/* find process rank using the cartesian coordinates and assigh the appropiate
	 * part of image previously splitted 
	**/
	MPI_File image;

	MPI_Datatype ARRAY;
    array_of_sizes[0] = height+2;
    array_of_sizes[1] = width+2;
 	array_of_starts[0] = 1;
    array_of_starts[1] = 1;
    array_of_subsizes[0] = height;
    array_of_subsizes[1] = width;
    MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &ARRAY);
    MPI_Type_commit(&ARRAY);


    MPI_Datatype FILETYPE;
    array_of_sizes[0] = Ny;
    array_of_sizes[1] = Nx;
    array_of_starts[0] = coords[1]*(height);
    array_of_starts[1] = coords[0]*(width);
    array_of_subsizes[0] = height;
    array_of_subsizes[1] = width;
    MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &FILETYPE);
    MPI_Type_commit(&FILETYPE);



	MPI_File_open(comm_cart,"../waterfall_grey_1920_2520.raw",MPI_MODE_RDWR,MPI_INFO_NULL,&image);
	fileOffset = 0;
	MPI_File_set_view(image,fileOffset,MPI_UNSIGNED_CHAR,FILETYPE,"native",MPI_INFO_NULL);
    MPI_File_read_all(image, data,1,ARRAY,&fileStatus);

    MPI_File_close(&image);

	
	/* Create three datatypes by defining sections of the original array
	 * Our array is treated as a 2 dimensional array and we are slicing 
	 * rows, columns and points having as starting position the initial position (0,0)
	 * of our array
	**/
	/* define the width and height of the array represented as 2 dimensional array
	 * halo points added to this representation
	**/
	array_of_sizes[0] = height + 2;
	array_of_sizes[1] = width + 2;
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

	t1 = MPI_Wtime();

	while (steps < totalSteps)
	{
		recvRequestCount = 0;
		sendRequestCount = 0;

		steps++;

		/* Locate neighboring processes in the grid and initiate a send-receive
		 * operation for each neighbor found 
		**/
		/* Left Neighboor - process (x-1,y) in the Cartesian! topology */
		nextCoords[0] = coords[0] - 1;
		nextCoords[1] = coords[1];
		errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
		if (errCode == MPI_SUCCESS) {
			/* take the column starting at ([1]-row,[1]-column) address and send */
			MPI_Isend(&data[offset(1,1)], 1, COLUMN, dest, tag,comm_cart,&sendRequestArr[sendRequestCount]);
			sendRequestCount++;
			source = dest;
			/* set starting position at ([1]-row,[0]-column) and place the received column */
			MPI_Irecv(&data[offset(1,0)], 1, COLUMN, source, tag,comm_cart,&recvRequestArr[recvRequestCount]);
			recvRequestCount++;

		}

		/* Right Neighboor - process (x-1,y) in the Cartesian! topology */
		nextCoords[0] = coords[0] + 1;
		nextCoords[1] = coords[1];
		errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
		if (errCode == MPI_SUCCESS) {
			/* take the column starting at ([1]-row,[width]-column) address and send */
			MPI_Isend(&data[offset(1,width)], 1, COLUMN, dest, tag,comm_cart,&sendRequestArr[sendRequestCount]);
			sendRequestCount++;
			source = dest;
			/* set starting position at ([1]-row,[width+1]-column) and place the received column */
			MPI_Irecv(&data[offset(1,width+1)], 1, COLUMN, source, tag,comm_cart,&recvRequestArr[recvRequestCount]);
			recvRequestCount++;
		}

		/* Bottom Neighboor - process (x,y+1) in the Cartesian! topology */
		nextCoords[0] = coords[0];
		nextCoords[1] = coords[1] + 1;
		errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
		if (errCode == MPI_SUCCESS) {
			/* take the row starting at ([height]-row,[1]-column) address and send */
			MPI_Isend(&data[offset(height,1)], 1, ROW, dest, tag,comm_cart,&sendRequestArr[sendRequestCount]);
			sendRequestCount++;
			source = dest;
			/* set starting position at ([height+1]-row,[1]-column) and place the received row */
			MPI_Irecv(&data[offset(height+1,1)], 1, ROW, source, tag,comm_cart,&recvRequestArr[recvRequestCount]);
			recvRequestCount++;
		}

		/* Top Neighboor - process (x,y-1) in the Cartesian! topology */
		nextCoords[0] = coords[0];
		nextCoords[1] = coords[1] - 1;
		errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
		if (errCode == MPI_SUCCESS) {
			/* take the row starting at ([1]-row,[1]-column) address and send */
			MPI_Isend(&data[offset(1,1)], 1, ROW, dest, tag,comm_cart,&sendRequestArr[sendRequestCount]);
			sendRequestCount++;
			source = dest;
			/* set starting position at ([0]-row,[1]-column) and place the received row */
			MPI_Irecv(&data[offset(0,1)], 1, ROW, source, tag,comm_cart,&recvRequestArr[recvRequestCount]);
			recvRequestCount++;
		}

		/* Bottom-Right Neighboor - process (x+1,y+1) in the Cartesian! topology */
		nextCoords[0] = coords[0] + 1;
		nextCoords[1] = coords[1] + 1;
		errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
		if (errCode == MPI_SUCCESS && 1) {
			/* take the point starting at ([height]-row,[width]-column) address and send */
			MPI_Isend(&data[offset(height,width)], 1, POINT, dest, tag,comm_cart,&sendRequestArr[sendRequestCount]);
			sendRequestCount++;
			source = dest;
			/* set starting position at ([height+1]-row,[width+1]-column) and place the received point */
			MPI_Irecv(&data[offset(height+1,width+1)], 1, POINT, source, tag,comm_cart,&recvRequestArr[recvRequestCount]);
			recvRequestCount++;
		}

		/* Top-Right Neighboor - process (x+1,y-1) in the Cartesian! topology */
		nextCoords[0] = coords[0] + 1;
		nextCoords[1] = coords[1] - 1;
		errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
		if (errCode == MPI_SUCCESS) {
			/* take the point starting at ([1]-row,[width]-column) address and send */
			MPI_Isend(&data[offset(1,width)], 1, POINT, dest, tag,comm_cart,&sendRequestArr[sendRequestCount]);
			sendRequestCount++;
			source = dest;
			/* set starting position at ([0]-row,[width+1]-column) and place the received point */
			MPI_Irecv(&data[offset(0,width+1)], 1, POINT, source, tag,comm_cart,&recvRequestArr[recvRequestCount]);
			recvRequestCount++;
		}

		/* Bottom-Left Neighboor - process (x-1,y-1) in the Cartesian! topology */
		nextCoords[0] = coords[0] - 1;
		nextCoords[1] = coords[1] + 1;
		errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
		if (errCode == MPI_SUCCESS) {
			/* take the point starting at ([height]-row,[1]-column) address and send */
			MPI_Isend(&data[offset(height,1)], 1, POINT, dest, tag,comm_cart,&sendRequestArr[sendRequestCount]);
			sendRequestCount++;
			source = dest;
			/* set starting position at ([height+1]-row,[0]-column) and place the received point */
			MPI_Irecv(&data[offset(height+1,0)], 1, POINT, source, tag,comm_cart,&recvRequestArr[recvRequestCount]);
			recvRequestCount++;
		}

		/* Top-Left Neighboor - process (x-1,y-1) in the Cartesian! topology */
		nextCoords[0] = coords[0] - 1;
		nextCoords[1] = coords[1] - 1;
		errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
		if (errCode == MPI_SUCCESS) {
			/* take the point starting at ([1]-row,[1]-column) address and send */
			MPI_Isend(&data[offset(1,1)], 1, POINT, dest, tag,comm_cart,&sendRequestArr[sendRequestCount]);
			sendRequestCount++;
			source = dest;
			/* set starting position at ([0]-row,[0]-column) and place the received point */
			MPI_Irecv(&data[offset(0,0)], 1, POINT, source, tag,comm_cart,&recvRequestArr[recvRequestCount]);
			recvRequestCount++;
		}
		
		/* calculate filter for inner data - no need for communication */
		innerImageFilter(data,results);
		

		/* before continuing to compute outer image make sure all messages 
		 * are received
		**/
		
		MPI_Waitall(recvRequestCount,recvRequestArr,MPI_STATUSES_IGNORE);
		//MPI_Testall(recvRequestCount, recvRequestArr,&rf, recvRequestStatus);
		//printf("%d\n",rf);
		

		/* calculate filter for outer with the halo points received 
		 * process coordinates are given in order to detect what part of the image
		 * the process holds
		**/
		outerImageFilter(data,results,coords);



		/* ensure all data have been sent successfully sent
		 * before the next loop iteration */
		
		MPI_Waitall(sendRequestCount,sendRequestArr,MPI_STATUSES_IGNORE);
		//MPI_Testall(sendRequestCount, sendRequestArr,&sf, recvRequestStatus);
		//printf("%d\n",sf);

		swapImage(&data,&results);

	}
	
	t2 = MPI_Wtime();

    Dt_local = t2 - t1;
	MPI_Reduce(&Dt_local, &Dt_total, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);	
	
/* Elapsed times for the slowest process */
	if (myRank == 0) {
		
		printf("Max. Total time: Dt = %.3f msec.\n\n", Dt_total * 1000);
	}



	MPI_File output;





	MPI_File_open(comm_cart,"../outgrey.raw",MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&output);
	MPI_File_set_view(output,0,MPI_UNSIGNED_CHAR,FILETYPE,"native",MPI_INFO_NULL);
    MPI_File_write_all(output, data,1,ARRAY,&fileStatus);

    MPI_File_close(&output);  



   	MPI_Finalize();


	return 0;
}



/* returns the offset of an element in a 2D array of dimensions (width,height) 
 * allocated in row major order
**/
int offset(int i,int j)
{
	return i*(width+2)+j;
}

void swapImage(unsigned char** data,unsigned char** results)
{

	/*for(i=0;i<height+2;i++)
	{
		for(j=0;j<width+2;j++)
		{
			data[offset(i,j)] = results[offset(i,j)];
		}
	}*/
	unsigned char* temp = *data;
	*data = *results;
	*results = temp;
}
int breakf()
{
	printf("break\n");
}

int innerImageFilter(unsigned char *data,unsigned char* results)
{



//#pragma omp parallel for collapse(2)
	for(i = 2; i < height; i++)
	{

		for(j = 2; j < width; j++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);
		}
	}
	return 1;
}

unsigned char innerPixelFilter(unsigned char* data,int i, int j)
{
	int k,l;
	int outPixel = 0;

	for(k=-1;k<=1;k++)
	{
		for(l=-1;l<=1;l++)
		{
				outPixel += (data[offset((i+k),(j+l))]*filter[k+1][l+1]);
		}
	}
	return (unsigned char)(outPixel/16);
}

/**
 * In order to generalize pixel filtering functions we create specific location flags
 * Location flags ( 0,-1): TOP        |     ( 1,-1):LEFT          | outerPixelFilter
 *                ( 0, 1): BOTTOM     |     ( 1, 1): RIGHT        |
 *
 *                (-1,-1): TOP-LEFT   |     ( 1,-1): BOTTOM-LEFT  | cornerPixelFilter
 *                (-1, 1): TOP-RIGHT  |     ( 1, 1): BOTTOM-RIGHT |
**/
int outerImageFilter(unsigned char* data, unsigned char* results,int* coords)
{
	int flag[2];
	/*  
	 * parse the whole TOP row excluding halo points and filter each pixel
	**/
	/* if the TOP part of the process is not a TOP part on the initial image
	 * then process the local data as usual as neighboors on TOP do exist
	**/
	if(coords[1]!=0)
	{
		i = 1; /* first row */
		for(j = 1; j < width+1; j++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);
		}
	}
	/* if the TOP part of the process is a TOP part on the initial image, pass the information 
	 * that those pixels belong to the TOP image part, and so TOP neighboors does not exist
	**/
	else
	{
		i = 1; /* first row */
		flag[0] = 0;
		flag[1] = -1;
		for(j = 1; j < width+1; j++)
		{
			results[offset(i,j)] = outerPixelFilter(data,i,j,flag);
		}
	}

	/*  
	 * parse the whole BOTTOM row excluding halo points and filter each pixel
	**/
	/* if the BOTTOM part of the process is not a BOTTOM part on the initial image
	 * then process the local data as usual as neighboors on BOTTOM do exist
	**/
	if(coords[1]!= dims[0]-1)
	{
		i = height; /* [height] row */
		for(j = 1; j < width+1; j++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);

		}
	}
	/* if the BOTTOM part of the process is a BOOTOM part on the initial image, pass the information 
	 * that those pixels belong to the BOTTOM image part, and so TOP neighboors does not exist
	**/
	else
	{
		i = height; /* [height] row */
		flag[0] = 0;
		flag[1] = 1;
		for(j = 1; j < width+1 ; j++)
		{
			results[offset(i,j)] = outerPixelFilter(data,i,j,flag);
		}
	}

	/*  
	 * parse the whole LEFT column excluding halo points and points submitted by the top 
	 * and bottom row and filter each pixel
	**/
	/* if the LEFT part of the process is not a LEFT part on the initial image
	 * then process the local data as usual as neighboors on LEFT do exist
	**/
	if(coords[0]!=0)
	{
		j = 1; /* first column */
		for(i = 2; i < height; i++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);

		}
	}
	/* if the LEFT part of the process is a LEFT part on the initial image, pass the information 
	 * that those pixels belong to the LEFT image part, and so LEFT neighboors does not exist
	**/
	else
	{
		j = 1; /* first column */
		flag[0] = 1;
		flag[1] = -1;
		for(i = 2; i < height; i++)
		{
			results[offset(i,j)] = outerPixelFilter(data,i,j,flag);
		}
	}

	/*  
	 * parse the whole RIGHT column excluding halo points and points submitted by the top 
	 * and bottom row and filter each pixel
	**/
	/* if the RIGHT part of the process is not a RIGHT part on the initial image
	 * then process the local data as usual as neighboors on RIGHT do exist
	**/
	if(coords[0] != dims[1] - 1) 	
	{
		j = width; /* [width] column */
		for(i = 2; i < height; i++)
		{
			results[offset(i,j)] = innerPixelFilter(data,i,j);

		}
	}
	/* if the RIGHT part of the process is a RIGHT part on the initial image, pass the information 
	 * that those pixels belong to the RIGHT image part, and so RIGHT neighboors does not exist
	**/
	else
	{
		j = width; /* [width] column */
		flag[0] = 1;
		flag[1] = 1;
		for(i = 2; i < height; i++)
		{
			results[offset(i,j)] = outerPixelFilter(data,i,j,flag);
		}
	}
	/*
	 * !Override! pixel values for corners created by the previous TOP and BOTTOM filters
	**/
	/* 
	 * For the TOP-LEFT process of the image and the top-left pixel (excluding halo point)
	 * override the result using the corner specific function
	**/
	if(coords[1]==0 && coords[0]==0)
	{
		flag[0] = -1;
		flag[1] = -1;

		results[offset(1,1)] = cornerPixelFilter(data,1,1,flag);
	}
	/* 
	 * For the TOP-RIGHT process of the image and the top-right pixel (excluding halo point)
	 * override the result using the corner specific function
	**/
	
	if(coords[1]==0 && coords[0]==dims[1] -1)
	{
		flag[0] = -1;
		flag[1] = 1;

		results[offset(1,width)] = cornerPixelFilter(data,1,width,flag);
	}
	/* 
	 * For the BOTTOM-LEFT process of the image and the bottom-left pixel (excluding halo point)
	 * override the result using the corner specific function
	**/
	if(coords[1]==dims[0] -1 && coords[0]==0)
	{
		flag[0] = 1;
		flag[1] = -1;

		results[offset(height,1)] = cornerPixelFilter(data,height,1,flag);
	}
	/* 
	 * For the BOTTOM-right process of the image and the bottom-right pixel (excluding halo point)
	 * override the result using the corner specific function
	**/
	if(coords[1]==dims[0] -1&& coords[0]==dims[1] -1)
	{
		flag[0] = 1;
		flag[1] = 1;

		results[offset(height,width)] = cornerPixelFilter(data,height,width,flag);
	}

	return 1;
}



unsigned char outerPixelFilter(unsigned char* data,int i, int j,int* flag)
{
	int k,l;
	int outPixel = 0;

	for(k=-1;k<=1;k++)
	{
		for(l=-1;l<=1;l++)
		{
			if( ((flag[0]==0) && (k==flag[1])) || ((flag[0]==1) && (l==flag[1]))   )
			{
				outPixel += (data[offset(i,j)]*filter[k+1][l+1]);	
			}
			else
			{
				outPixel += (data[offset((i+k),(j+l))]*filter[k+1][l+1]);
			}
		}
	}
	return (unsigned char)(outPixel/16);
}

unsigned char cornerPixelFilter(unsigned char* data,int i, int j,int* flag)
{
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
				outPixel += (data[offset((i+k),(j+l))]*filter[k+1][l+1]);
			}
		}
	}
	return (unsigned char)(outPixel/16);
}



int parseCmdLineArgs(int argc, char **argv, int *dims, int myRank) {
	/*
	Nx = 1920;
	Ny = 2520;
	dims[0]=2;
	dims[1]=2;
	totalSteps = 4;
	reduceFreq = 10;


	return 0;
	*/
	
	if (argv[1] != NULL && strcmp(argv[1], "-nodes") == 0) {
		if (argv[2] != NULL && argv[3] != NULL) {
			Nx = atoi(argv[2]);
			Ny = atoi(argv[3]);
		} else {
			if (myRank == 0) {
				printf(
						"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
								" [-nodes <Nx> <Ny> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny>  -procs <i> <j> -steps <n> -reduce <f>]\n\n");
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
								" [-nodes <Nx> <Ny> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny>  -procs <i> <j> -steps <n> -reduce <f>]\n\n");
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
								" [-nodes <Nx> <Ny> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny>  -procs <i> <j> -steps <n> -reduce <f>]\n\n");
		}
		return 1;
	}

	if (argv[9] != NULL && strcmp(argv[9], "-reduce") == 0) {
		if (argv[10] != NULL) {
			reduceFreq = (int) atoi(argv[10]);
		} else {
			if (myRank == 0) {
				printf(
						"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
								" [-nodes <Nx> <Ny> -procs <i> <j> -steps <n> -reduce <f>]\n\n");
			}
			return 1;
		}
	} else {
		if (myRank == 0) {
			printf(
					"\nSpecify grid of nodes, grid of processes, number of iterations and reduction frequency"
							" [-nodes <Nx> <Ny>  -procs <i> <j> -steps <n> -reduce <f>]\n\n");
		}
		return 1;
	}
	
	return 0;
	
}
