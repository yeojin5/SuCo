#!/bin/bash

#gdb --args ./suco 

DATASET="openai1m"
DATA_DIM=1536
SUBSPACE_NUM=8
SUBSPACE_DIM=$((DATA_DIM/SUBSPACE_NUM))
K_SIZE=$1
ALPHA=$2 # candidate ratio
BETA=$3 # collision ratio
TOPSUBSPACE=$4

#gdb --args 
./suco  --dataset-path ./dataset/${DATASET}/${DATASET}_base.fbin \
	--query-path ./dataset/${DATASET}/${DATASET}_query.fbin \
	--groundtruth-path ./dataset/${DATASET}/${DATASET}_gt_K${K_SIZE}.fbin \
	--dataset-size 1000000 \
	--query-size 1000 \
	--k-size ${K_SIZE} \
	--data-dimensionality ${DATA_DIM} \
	--subspace-dimensionality $SUBSPACE_DIM \
	--subspace-num $SUBSPACE_NUM \
	--candidate-ratio $ALPHA \
	--collision-ratio $BETA \
	--kmeans-num-centroid 50 \
	--kmeans-num-iters 2 \
	--index-path ./index/${DATASET}/${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}_K${K_SIZE}.bin \
	--load-index ./index/${DATASET}/${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}_K${K_SIZE}.bin \
	--top-n-subspaces ${TOPSUBSPACE} \
	| tee ./result/${DATASET}/${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}_K${K_SIZE}_simple_${TOPSUBSPACE}.txt
	# --load-index ./index/${DATASET}/${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}_K${K_SIZE}.bin
echo "save [${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}_K${K_SIZE}.txt]"
