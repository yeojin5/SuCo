#!/bin/bash


DATASET="sift10m"
DATA_DIM=128
SUBSPACE_NUM=8
SUBSPACE_DIM=$((DATA_DIM/SUBSPACE_NUM))
K_SIZE=$1
ALPHA=$2 # candidate ratio
BETA=$3 # collision ratio

#gdb -ex run -ex bt --args 
./suco  --dataset-path ./dataset/${DATASET}/${DATASET}_base.fbin \
	--query-path ./dataset/${DATASET}/${DATASET}_query.fbin \
	--groundtruth-path ./dataset/${DATASET}/${DATASET}_gt_K${K_SIZE}.fbin \
	--dataset-size 10000000 \
	--k-size ${K_SIZE} \
	--data-dimensionality $DATA_DIM \
	--subspace-dimensionality $SUBSPACE_DIM \
	--subspace-num $SUBSPACE_NUM \
	--candidate-ratio $ALPHA \
	--collision-ratio $BETA \
	--kmeans-num-centroid 50 \
	--kmeans-num-iters 2 \
	--index-path ./index/${DATASET}/${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}_K${K_SIZE}.bin \
	| tee ./result/${DATASET}/${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}_K${K_SIZE}.txt
	#--load-index ./index/${DATASET}/${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}_K${K_SIZE}.bin
echo "save [sift10m_${SUBSPACE_NUM}_${ALPHA}_${BETA}_K${K_SIZE}.txt]"
