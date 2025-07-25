#!/bin/bash


DATASET="sift10m"
DATA_DIM=128
SUBSPACE_NUM=8
SUBSPACE_DIM=$((DATA_DIM/SUBSPACE_NUM))
ALPHA=0.05 # candidate ratio
BETA=0.005 # collision ratio
K_SIZE=50

#gdb -ex run -ex bt --args 
./suco  --dataset-path ./dataset/${DATASET}/${DATASET}_base.fbin \
	--query-path ./dataset/${DATASET}/${DATASET}_query.fbin \
	--groundtruth-path ./dataset/${DATASET}/${DATASET}_gt_k${K_SIZE} \
	--dataset-size 10000000 \
	--k-size ${K_SIZE} \
	--data-dimensionality $DATA_DIM \
	--subspace-dimensionality $SUBSPACE_DIM \
	--subspace-num $SUBSPACE_NUM \
	--candidate-ratio $ALPHA \
	--collision-ratio $BETA \
	--kmeans-num-centroid 50 \
	--kmeans-num-iters 2 \
	--index-path ./index/${DATASET}/${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}.bin #\
	# > ./result/${DATASET}/${DATASET}_${SUBSPACE_NUM}_${ALPHA}_${BETA}.txt
echo "save [sift10m_${SUBSPACE_NUM}_${ALPHA}_${BETA}.txt]"

