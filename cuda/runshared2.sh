#!/bin/bash
rm out.raw
nvcc -I./ -o cudasep cudasharedseperable.cu sharedseperable.cu
./cudasep
