#include "evaluate.h"
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>

void recall_and_ratio(float ** &dataset, float ** &querypoints, int data_dimensionality, int ** &queryknn_results, long int ** &gt, int query_size) {
    int ks[1] = {10};
    
    for (int k_index = 0; k_index < sizeof(ks) / sizeof(ks[0]); k_index++) {
        int retrieved_data_num = 0;

        for (int i = 0; i < query_size; i++)
        {
            for (int j = 0; j < ks[k_index]; j++)
            {
                for (int z = 0; z < ks[k_index]; z++) {
                    if (queryknn_results[i][j] == gt[i][z]) {
		    //	std::cout << "query_result["<<i<<"]["<<j <<"]"<< ", gt[" <<i<< "][" <<z<< "]" << gt[i][z] << std::endl;  
                        retrieved_data_num++;
                        break;
                    }
		    // std::cout <<  "retrieved_data_num: " << retrieved_data_num << std::endl;
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
    int ks[1] = {10};

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
                    if (ground_truth_set.size() > 0) {
                        subspace_accuracy[j] += (double)accuracy_count / ground_truth_set.size();
                    }
                }
            }
        }

        // 출력
        for (int j = 0; j < subspace_num; ++j) {
            double acc = (subspace_accuracy[j] / query_size) * 100.0;
            double contrib = (subspace_contribution[j] / query_size / current_k) * 100.0;
            cout << current_k << "," << j << "," << acc << "," << contrib << endl;
        }
    }
}

void evaluate_chosen_subspace_recall(
    long int** &gt, // Ground truth
    const std::vector<std::vector<std::vector<int>>>& subspace_candidates, // [query_id][subspace_id][candidates]
    const std::vector<std::vector<int>>& chosen_subspaces_by_query, // [query_id][chosen_subspace_ids]
    int query_size,
    int k,
    int num_subspaces
) {
    std::vector<double> subspace_recall_sum(num_subspaces, 0.0);
    std::vector<int> subspace_selection_count(num_subspaces, 0);

    for (int i = 0; i < query_size; ++i) {
        const auto& chosen_subspaces = chosen_subspaces_by_query[i];
        std::unordered_set<long int> ground_truth_set(gt[i], gt[i] + k);

        for (int chosen_subspace_id : chosen_subspaces) {
            if (chosen_subspace_id < 0 || chosen_subspace_id >= num_subspaces) {
                cerr << "Warning: Query " << i << " has invalid chosen_subspace id " << chosen_subspace_id << endl;
                continue;
            }

            const auto& candidates = subspace_candidates[i][chosen_subspace_id];
            
            int hit_count = 0;
            for (int candidate_id : candidates) {
                if (ground_truth_set.count(candidate_id)) {
                    hit_count++;
                }
            }

            double recall = (ground_truth_set.size() > 0) ? static_cast<double>(hit_count) / ground_truth_set.size() : 0.0;

            subspace_recall_sum[chosen_subspace_id] += recall;
            subspace_selection_count[chosen_subspace_id]++;
        }
    }

    cout << "\n--- Chosen Subspace Recall Evaluation ---" << endl;
    cout << "Subspace,AvgRecall,Selections" << endl;
    for (int j = 0; j < num_subspaces; ++j) {
        if (subspace_selection_count[j] > 0) {
            double avg_recall = subspace_recall_sum[j] / subspace_selection_count[j];
            cout << j << "," << avg_recall << "," << subspace_selection_count[j] << endl;
        } else {
            cout << j << ",0.0," << 0 << endl;
        }
    }
    cout << "-----------------------------------------" << endl;
}

