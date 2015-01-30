#!/bin/bash
rm out.raw
nvcc -o cudacannon cudacannon.cu
./cudacannon
