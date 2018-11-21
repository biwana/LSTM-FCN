#!/bin/bash
./run.sh generate_dataset_flexible_generic_fold.py $1 random independent $2 $3 $4 $5 && \
./run.sh generate_dataset_flexible_generic_fold.py $1 centers independent $2 $3 $4 $5 && \
./run.sh generate_dataset_flexible_generic_fold.py $1 kcenters independent $2 $3 $4 $5 && \
./run.sh generate_dataset_flexible_generic_fold.py $1 borders independent $2 $3 $4 $5 && \
./run.sh generate_dataset_flexible_generic_fold.py $1 spanning independent $2 $3 $4 $5
