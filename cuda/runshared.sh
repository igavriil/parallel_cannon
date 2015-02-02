#!/bin/bash
rm out.raw
nvcc -o cudasharedcannon cudasharedcannon.cu
./cudasharedcannon
