# parallel_cannon

This algorithm is about filtering an image with a specific filter. Each pixel of the output image is calculated as the sum of each neighboor multiplied by the coefficient defined in the filter array. Changing the filter array different results in the output image can be extracted.

##### Sequential Algorithm
We define two structures for holding the image with information related to the dimentions of it: the input image and the output image. After applying the filter to each pixel this structures are swap (pointer swaping) in order to repeat the process in the generated image. The number of iterations is defined as the "hardness" of the applying filter.

##### Parallel Algorithm using MPI
* Each process takes a fraction of the image. Let assume that the image is splitted into square parts and distributed in each process (this is the most scalable choice for the data distribution)
* Create new datatypes in order to define the parts of the image each process send/received to/from another process. These datatypes also include specific information about the positioning of these datatypes(arrays actually) in the local array that each process holds.
Creating datatypes prevent from unneeded copying data to buffers.
* Create a 2D-cartesian communicator which provide an easy way to find each process's neighboors, but also optimizes the placement of the machines topological.
* Initiate the communication (asynchronous/non-blocking) for the processes and proceed to computer the inner parts of the image filter for each process. Upon finishing ensure that all communication has finished and proceed to compute outer parts(halo points). Filter computation may optionally be performed as Multithreaded, using OpenMP functionallity.
* Repeat 
