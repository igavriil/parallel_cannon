#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);


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





	MPI_Finalize();

}
