#include "evaluate.h"
#include <unordered_set>

void recall_and_ratio(float ** &dataset, float ** &querypoints, int data_dimensionality, int ** &queryknn_results, long int ** &gt, int query_size) {
    int ks[6] = {1, 10, 20, 30, 40, 50};
    
    for (int k_index = 0; k_index < sizeof(ks) / sizeof(ks[0]); k_index++) {
        int retrieved_data_num = 0;

        for (int i = 0; i < query_size; i++)
        {
            for (int j = 0; j < ks[k_index]; j++)
            {
                for (int z = 0; z < ks[k_index]; z++) {
			// std::cout << "query_result["<<i<<"]["<<j <<"]"<< ", gt[" <<i<< "][" <<z<< "]" << gt[i][z] << std::endl;  
                    if (queryknn_results[i][j] == gt[i][z]) {
                        retrieved_data_num++;
                        break;
                    }
                }
            }
        }

        float ratio = 0.0f;
        for (int i = 0; i < query_size; i++)
        {
            for (int j = 0; j < ks[k_index]; j++)
            {
                // Distance between the query and its ground truth neighbor.
                float groundtruth_square_dist = euclidean_distance(querypoints[i], dataset[gt[i][j]], data_dimensionality);
                // Distance between the query and the retrieved neighbor.
                float obtained_square_dist = euclidean_distance(querypoints[i], dataset[queryknn_results[i][j]], data_dimensionality);
                if (groundtruth_square_dist == 0) {
                    ratio += 1.0f;
                } else {
                    // This ratio measures similarity. A value closer to 1 means the retrieved neighbor's distance
                    // is similar to the ground truth neighbor's distance.
                    ratio += sqrt(obtained_square_dist) / sqrt(groundtruth_square_dist);
		}
            }
        }

        float recall_value = float(retrieved_data_num) / (query_size * ks[k_index]);
        float overall_ratio = ratio / (query_size * ks[k_index]);

        cout << "When k = " << ks[k_index] << ", (recall, ratio) = (" << recall_value << ", " << overall_ratio << ")" << endl;
    }
}

void subspace_accuracy_and_contribution(
    long int ** &gt,
    int ** &queryknn_results,
    const vector<vector<vector<int>>> &subspace_candidates,
    const vector<vector<vector<int>>> &subspace_scores, // 각 candidate의 subspace별 score
    int query_size,
    int k_size,
    int subspace_num
) {
    cout << "k,subspace,accuracy,contribution" << endl;
    int ks[6] = {1, 10, 20, 30, 40, 50};

    for (int k_index = 0; k_index < sizeof(ks) / sizeof(ks[0]); k_index++) {
        int current_k = ks[k_index];

        vector<double> subspace_accuracy(subspace_num, 0.0);
        vector<double> subspace_contribution(subspace_num, 0.0);

        for (int i = 0; i < query_size; ++i) {
            unordered_set<long int> ground_truth_set(gt[i], gt[i] + current_k);

            // step 1: 최종 Top-k 결과 벡터 집합
            unordered_set<int> topk_result_set(queryknn_results[i], queryknn_results[i] + current_k);

            // step 2: 해당 벡터들의 subspace별 SC-score 분해
            unordered_map<int, vector<int>> score_map;  // candidate_id -> vector of subspace scores
            unordered_map<int, int> score_sum_map;      // candidate_id -> total SC-score

            for (int j = 0; j < subspace_num; ++j) {
                if (i < subspace_candidates.size() && j < subspace_candidates[i].size()) {
                    const auto &candidates = subspace_candidates[i][j];
                    const auto &scores = subspace_scores[i][j];
                    for (int k = 0; k < candidates.size(); ++k) {
                        int candidate_id = candidates[k];
                        int score = scores[k];

                        score_map[candidate_id].resize(subspace_num, 0);
                        score_map[candidate_id][j] += score;
                        score_sum_map[candidate_id] += score;
                    }
                }
            }

            // step 3: subspace별 기여도 계산
            for (int result_id : topk_result_set) {
                if (score_map.find(result_id) == score_map.end()) continue;
                int total_score = score_sum_map[result_id];
                if (total_score == 0) continue;

                for (int j = 0; j < subspace_num; ++j) {
                    subspace_contribution[j] += (double)score_map[result_id][j] / total_score;
                }
            }

            // step 4: subspace별 accuracy 계산
            for (int j = 0; j < subspace_num; ++j) {
                int accuracy_count = 0;
                if (i < subspace_candidates.size() && j < subspace_candidates[i].size()) {
                    const auto &candidates = subspace_candidates[i][j];
                    unordered_set<int> candidate_set(candidates.begin(), candidates.end());

                    for (int candidate_id : candidate_set) {
                        if (ground_truth_set.count(candidate_id)) {
                            accuracy_count++;
                        }
                    }
                    if (!ground_truth_set.empty()) {
                        subspace_accuracy[j] += (double)accuracy_count / ground_truth_set.size();
                    }
                }
            }
        }

        // 출력
        for (int j = 0; j < subspace_num; ++j) {
            double acc = subspace_accuracy[j] / query_size * 100.0;
            double contrib = subspace_contribution[j] / query_size * 100.0;
            cout << current_k << "," << j << "," << acc << "," << contrib << endl;
        }
    }
}

