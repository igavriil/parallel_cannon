#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>





int main()
{
	

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	
	
	/*
	 * Datatype definition START
	 * */
	
	int array_of_sizes[2];
	int array_of_subsizes[2];
	int array_of_starts[2];
	
	int sub_dims = 2;
	
	MPI_Datatype SEND_TO_RIGHT;
	
	
	array_of_sizes[0] = width;
	array_of_sizes[1] = height;
	
	array_of_subsizes[0] = 1;
	array_of_subsizes[1] = height;
	
	array_of_starts[0] = width-1;
	array_of_starts[1] = 0;
	
	
	MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
				array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_RIGHT);
	
	MPI_Type_commit(&SEND_TO_RIGHT);

	
	
	
	MPI_Datatype SEND_TO_LEFT;
		
		
		array_of_sizes[0] = width;
		array_of_sizes[1] = height;
		
		array_of_subsizes[0] = 1;
		array_of_subsizes[1] = height;
		
		array_of_starts[0] = 0;
		array_of_starts[1] = 0;
		
		
		MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
					array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_LEFT);

		MPI_Type_commit(&SEND_TO_LEFT);



		MPI_Datatype SEND_TO_TOP;
			
			
			array_of_sizes[0] = width;
			array_of_sizes[1] = height;
			
			array_of_subsizes[0] = height;
			array_of_subsizes[1] = 1;
			
			array_of_starts[0] = 0;
			array_of_starts[1] = 0;
			
			
			MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
						array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_TOP);

			MPI_Type_commit(&SEND_TO_TOP);

	

			MPI_Datatype SEND_TO_BOTTOM;
				
				
				array_of_sizes[0] = width;
				array_of_sizes[1] = height;
				
				array_of_subsizes[0] = height;
				array_of_subsizes[1] = 1;
				
				array_of_starts[0] = 0;
				array_of_starts[1] = height-1;
				
				
				MPI_Type_create_subarray(sub_dims, array_of_sizes, array_of_subsizes,
							array_of_starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &SEND_TO_BOTTOM);

				MPI_Type_commit(&SEND_TO_BOTTOM);

	
	

	/*
	 * Datatype definition END
	 * */
	
	
	
	
				
				
	
	
	
	
	MPI_Finalize();
	
}
