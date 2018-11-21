#!/bin/bash
./run.sh generic_model_gpu$3.py $1 random-independent $2 && \
./run.sh generic_model_gpu$3.py $1 borders-independent $2 && \
./run.sh generic_model_gpu$3.py $1 kcenters-independent $2 && \
./run.sh generic_model_gpu$3.py $1 centers-independent $2 && \
./run.sh generic_model_gpu$3.py $1 spanning-independent $2 
