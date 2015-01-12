#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int Nx,Ny,numtasks,totalSteps;


int parseCmdLineArgs(int argc, char **argv, int *dims, int myRank);


int main(int argc, char* argv[]) 
{


	int width,height,rank,myGridRank,errCode, dest, source,tag,sendRequestCount,recvRequestCount;
	double* data=NULL;
	int periods[2];
	int coords[2];
	int nextCoords[2];

	MPI_Comm comm_cart;
	MPI_Errhandler errHandler;
	MPI_Request sendRequestArr[8];
	MPI_Request recvRequestArr[8];

	int array_of_sizes[2];
	int array_of_subsizes[2];
	int array_of_starts[2];

	int sub_dims = 2;

	array_of_sizes[0] = width + 2;
	array_of_sizes[1] = height + 2;

	/* vertical (x) and horizontal (y) dimensions */

	/* slice to be sent to right neighbor (X-axis) */
	MPI_Datatype SEND_TO_RIGHT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = height;
	array_of_starts[0] = width;
	array_of_starts[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_RIGHT);
	MPI_Type_commit(&SEND_TO_RIGHT);

	/* slice to be received from right neighbor (X-axis) */
	MPI_Datatype RCV_FROM_RIGHT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = height;
	array_of_starts[0] = width + 1;
	array_of_starts[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_RIGHT);
	MPI_Type_commit(&RCV_FROM_RIGHT);

	/* slice to be sent to left neighbor (X-axis) */
	MPI_Datatype SEND_TO_LEFT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = height;
	array_of_starts[0] = 1;
	array_of_starts[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_LEFT);
	MPI_Type_commit(&SEND_TO_LEFT);

	/* slice to be received from left neighbor (X-axis) */
	MPI_Datatype RCV_FROM_LEFT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = height;
	array_of_starts[0] = 0;
	array_of_starts[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_LEFT);
	MPI_Type_commit(&RCV_FROM_LEFT);


	/* slice to be sent to top neighbor (Y-axis) */
	MPI_Datatype SEND_TO_TOP;
	array_of_subsizes[0] = width;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = 1;
	array_of_starts[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_TOP);
	MPI_Type_commit(&SEND_TO_TOP);

	/* slice to be received from top neighbor (Y-axis) */
	MPI_Datatype RCV_FROM_TOP;
	array_of_subsizes[0] = width;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = 1;
	array_of_starts[1] = 0;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_TOP);
	MPI_Type_commit(&RCV_FROM_TOP);

	/* slice to be sent to bottom, neighbor (Y-axis) */
	MPI_Datatype SEND_TO_BOTTOM;
	array_of_subsizes[0] = width;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = 1;
	array_of_starts[1] = height;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_BOTTOM);
	MPI_Type_commit(&SEND_TO_BOTTOM);

	/* slice to be received from bottom neighbor (Y-axis) */
	MPI_Datatype RCV_FROM_BOTTOM;
	array_of_subsizes[0] = width;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = height + 1;
	array_of_starts[1] = 0;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_BOTTOM);
	MPI_Type_commit(&RCV_FROM_BOTTOM);

	/* diagonal dimensions (xy) */

	/* slice to be sent to top-right neighbor (XY-axis) */
	MPI_Datatype SEND_TO_TOP_RIGHT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = width;
	array_of_starts[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_TOP_RIGHT);
	MPI_Type_commit(&SEND_TO_TOP_RIGHT);

	/* slice to be received from top-right neighbor (XY-axis) */
	MPI_Datatype RCV_FROM_TOP_RIGHT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = width + 1;
	array_of_starts[1] = 0;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_TOP_RIGHT);
	MPI_Type_commit(&RCV_FROM_TOP_RIGHT);

	/* slice to be sent to top-left neighbor (XY-axis) */
	MPI_Datatype SEND_TO_TOP_LEFT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = 1;
	array_of_starts[1] = 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_TOP_LEFT);
	MPI_Type_commit(&SEND_TO_TOP_LEFT);

	/* slice to be received from top-left neighbor (XY-axis) */
	MPI_Datatype RCV_FROM_TOP_LEFT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = 0;
	array_of_starts[1] = 0;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_TOP_LEFT);
	MPI_Type_commit(&RCV_FROM_TOP_LEFT);

	/* slice to be sent to bottom-rigth neighbor (XY-axis) */
	MPI_Datatype SEND_TO_BOTTOM_RIGHT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = width;
	array_of_starts[1] = height;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_BOTTOM_RIGHT);
	MPI_Type_commit(&SEND_TO_BOTTOM_RIGHT);

	/* slice to be received from bottom-right neighbor (XY-axis) */
	MPI_Datatype RCV_FROM_BOTTOM_RIGHT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = width + 1;
	array_of_starts[1] = height + 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_BOTTOM_RIGHT);
	MPI_Type_commit(&RCV_FROM_BOTTOM_RIGHT);

	/* slice to be sent to bottom-left neighbor (XY-axis) */
	MPI_Datatype SEND_TO_BOTTOM_LEFT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = 1;
	array_of_starts[1] = height;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_BOTTOM_LEFT);
	MPI_Type_commit(&SEND_TO_BOTTOM_LEFT);

	/* slice to be received from bottom-left neighbor (XY-axis) */
	MPI_Datatype RCV_FROM_BOTTOM_LEFT;
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = 1;
	array_of_starts[0] = 0;
	array_of_starts[1] = height + 1;
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
			array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &RCV_FROM_BOTTOM_LEFT);
	MPI_Type_commit(&RCV_FROM_BOTTOM_LEFT);



	/* dimensions for the cartesian grid */
	int ndims = 2;
	/* number of processes in each dimension */
	int dims[2];


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	/* Create a new communication and attach virtual topology (2D Cartesian) */
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm_cart);

	/* find out process rank in the grid */
	MPI_Comm_rank(comm_cart, &myGridRank);
	/* find out process coordinates in the grid */
	MPI_Cart_coords(comm_cart, myGridRank, ndims, coords);


	
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
		MPI_Send_init(&data[0], 1, SEND_TO_LEFT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[0], 1, RCV_FROM_LEFT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1];
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[0], 1, SEND_TO_RIGHT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[0], 1, RCV_FROM_RIGHT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0];
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[0], 1, SEND_TO_TOP, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[0], 1, RCV_FROM_TOP, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0];
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[0], 1, SEND_TO_BOTTOM, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[0], 1, RCV_FROM_BOTTOM, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[0], 1, SEND_TO_BOTTOM_RIGHT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[0], 1, RCV_FROM_BOTTOM_RIGHT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] + 1;
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[0], 1, SEND_TO_TOP_RIGHT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[0], 1, RCV_FROM_TOP_RIGHT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] - 1;
	nextCoords[1] = coords[1] + 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[0], 1, SEND_TO_BOTTOM_LEFT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[0], 1, RCV_FROM_BOTTOM_LEFT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}

	/* process (x-1,y) */
	nextCoords[0] = coords[0] - 1;
	nextCoords[1] = coords[1] - 1;
	errCode = MPI_Cart_rank(comm_cart, nextCoords, &dest);
	if (errCode == MPI_SUCCESS) {
		MPI_Send_init(&data[0], 1, SEND_TO_TOP_LEFT, dest, tag,
				comm_cart, &sendRequestArr[sendRequestCount]);
		sendRequestCount++;
		source = dest;
		MPI_Recv_init(&data[0], 1, RCV_FROM_TOP_LEFT, source, tag,
				comm_cart, &recvRequestArr[recvRequestCount]);
		recvRequestCount++;
	}


	MPI_Finalize();

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
