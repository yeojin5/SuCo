#pragma once
#include <iostream>
#include <math.h>
#include <vector>
#include "dist_calculation.h"

void recall_and_ratio(float ** &dataset, float ** &querypoints, int data_dimensionality, int ** &queryknn_results, int ** &gt, int query_size);
void subspace_accuracy_and_contribution(int ** &gt, int ** &queryknn_results, const vector<vector<vector<int>>> &subspace_candidates, int query_size, int k_size, int subspace_num);


using namespace std;