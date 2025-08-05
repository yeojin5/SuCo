#pragma once
#include <iostream>
#include <math.h>
#include <vector>
#include "dist_calculation.h"

void recall_and_ratio(float ** &dataset, float ** &querypoints, int data_dimensionality, int ** &queryknn_results, long int ** &gt, int query_size);

void subspace_accuracy_and_contribution(
    long int ** &gt,
    int ** &queryknn_results,
    const vector<vector<vector<int>>> &subspace_candidates,
    const vector<vector<vector<int>>> &subspace_scores,
    int query_size,
    int k_size,
    int subspace_num
);

void evaluate_chosen_subspace_recall(
    long int** &gt, // Ground truth
    const std::vector<std::vector<std::vector<int>>>& subspace_candidates, // [query_id][subspace_id][candidates]
    const std::vector<std::vector<int>>& chosen_subspaces_by_query, // [query_id][chosen_subspace_ids]
    int query_size,
    int k,
    int num_subspaces
);

using namespace std;