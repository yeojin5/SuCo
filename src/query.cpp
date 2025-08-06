#include "query.h"
#include <algorithm>
#include <vector>

void ann_query(float ** &dataset, int ** &queryknn_results, long int dataset_size, int data_dimensionality, int query_size, int k_size, float ** &querypoints, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, float * &centroids_list, int subspace_num, int subspace_dimensionality, int kmeans_num_centroid, int kmeans_dim, int collision_num, int candidate_num, int number_of_threads, long int &query_time, vector<vector<vector<int>>> &subspace_candidates, vector<vector<vector<int>>> &subspace_scores, const vector<int>& excluded_subspaces, int top_n_subspaces) {
    struct timeval start_query, end_query;
    
    progress_display pd_query(query_size);

    subspace_candidates.resize(query_size);
    subspace_scores.resize(query_size);

    vector<int> collision_count(dataset_size, 0);
    
    for (int i = 0; i < query_size; i++) {
        gettimeofday(&start_query, NULL);

        subspace_candidates[i].resize(subspace_num);
        subspace_scores[i].resize(subspace_num);

        vector<pair<float, int>> subspace_distances;
        for (int j = 0; j < subspace_num; j++) {
            if (find(excluded_subspaces.begin(), excluded_subspaces.end(), j) != excluded_subspaces.end()) {
                continue;
            }
            float min_dist = FLT_MAX;
            for (int z = 0; z < kmeans_num_centroid; z++) {
                float dist = euclidean_distance(&querypoints[i][j * subspace_dimensionality], &centroids_list[j * 2 * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            subspace_distances.push_back({min_dist, j});
        }

        sort(subspace_distances.begin(), subspace_distances.end());

        vector<int> selected_subspaces;
        for (int j = 0; j < top_n_subspaces && j < subspace_distances.size(); j++) {
            selected_subspaces.push_back(subspace_distances[j].second);
        }
//	std::cout << "[selected subspace num] ";
//	for (int j = 0; j < top_n_subspaces && j < subspace_distances.size(); j++){
//		std::cout << selected_subspaces[j]<< "," ;
//	}
//	std::cout << std::endl;

        for (int subspace_idx : selected_subspaces) {
            // first half dist
            vector<float> first_half_dists(kmeans_num_centroid);
            for (int z = 0; z < kmeans_num_centroid; z++) {
                first_half_dists[z] = euclidean_distance(&querypoints[i][subspace_idx * subspace_dimensionality], &centroids_list[subspace_idx * 2 * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // first half sort
            vector<int> first_half_idx(kmeans_num_centroid);
            iota(first_half_idx.begin(), first_half_idx.end(), 0);
            sort(first_half_idx.begin(), first_half_idx.end(), [&first_half_dists](int i1, int i2) {return first_half_dists[i1] < first_half_dists[i2];});

            // second half dist
            vector<float> second_half_dists(kmeans_num_centroid);
            for (int z = 0; z < kmeans_num_centroid; z++) {
                second_half_dists[z] = euclidean_distance(&querypoints[i][subspace_idx * subspace_dimensionality + kmeans_dim], &centroids_list[(subspace_idx * 2 + 1) * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // second half sort
            vector<int> second_half_idx(kmeans_num_centroid);
            iota(second_half_idx.begin(), second_half_idx.end(), 0);
            sort(second_half_idx.begin(), second_half_idx.end(), [&second_half_dists](int i1, int i2) {return second_half_dists[i1] < second_half_dists[i2];});


            // dynamic activate algorithm (multi thread)
            vector<pair<int, int>> retrieved_cell;
            dynamic_activate(indexes, retrieved_cell, first_half_dists, first_half_idx, second_half_dists, second_half_idx, collision_num, kmeans_num_centroid, subspace_idx);

            // count collision and store subspace candidates
            vector<int>& current_subspace_candidates = subspace_candidates[i][subspace_idx];
            vector<int>& current_subspace_scores = subspace_scores[i][subspace_idx];
            for (int z = 0; z < retrieved_cell.size(); z++) {
                auto iterator = indexes[subspace_idx].find(retrieved_cell[z]);
                if (iterator != indexes[subspace_idx].end()) {
                    current_subspace_candidates.insert(current_subspace_candidates.end(), iterator->second.begin(), iterator->second.end());
                    current_subspace_scores.insert(current_subspace_scores.end(), iterator->second.size(), 1);
                }
            }

            // Remove duplicates
            sort(current_subspace_candidates.begin(), current_subspace_candidates.end());
            current_subspace_candidates.erase(unique(current_subspace_candidates.begin(), current_subspace_candidates.end()), current_subspace_candidates.end());

            // Update collision_count using the unique candidates
            #pragma omp parallel for num_threads(number_of_threads)
            for (int z = 0; z < current_subspace_candidates.size(); z++) {
                #pragma omp atomic
                collision_count[current_subspace_candidates[z]]++;
            }
        }

        int * collision_num_array = new int[subspace_num + 1]();
        int ** local_collision_num = new int * [number_of_threads];
        for (int j = 0; j < number_of_threads; j++) {
            local_collision_num[j] = new int [subspace_num + 1]();
        }

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < dataset_size; j++) {
            int id = omp_get_thread_num();
            local_collision_num[id][collision_count[j]]++;
        }

        for (int j = 0; j < subspace_num + 1; j++) {
            for (int z = 0; z < number_of_threads; z++) {
                collision_num_array[j] += local_collision_num[z][j];
            }
        }

        // release the candidate number to include all points in last_collision_num, saving the time for checking points whose collision_num is last_collision_num
        int last_collision_num;
        int sum_candidate = 0;
        for (int j = subspace_num; j >= 0; j--) {
            if (collision_num_array[j] <= candidate_num - sum_candidate) {
                sum_candidate += collision_num_array[j];
            } else {
                last_collision_num = j;
                break;
            }
        }

        vector<int> candidate_idx;
        vector<vector<int>> local_candidate_idx(number_of_threads);

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < dataset_size; j++) {
            int id = omp_get_thread_num();
            if (collision_count[j] >= last_collision_num) {
                local_candidate_idx[id].push_back(j);
            }
        }

        for (int j = 0; j < number_of_threads; j++) {
            candidate_idx.insert(candidate_idx.end(), local_candidate_idx[j].begin(), local_candidate_idx[j].end());
        }

        vector<float> candidate_dists(candidate_idx.size());

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < candidate_idx.size(); j++) {
            // candidate_dists[j] = euclidean_distance(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
            // candidate_dists[j] = euclidean_distance_SIMD(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
            candidate_dists[j] = faiss::fvec_L2sqr_avx512(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
        }

        vector<int> candidate_sort_idx(candidate_idx.size());
        iota(candidate_sort_idx.begin(), candidate_sort_idx.end(), 0);
        if (k_size < candidate_idx.size()) {
            partial_sort(candidate_sort_idx.begin(), candidate_sort_idx.begin() + k_size, candidate_sort_idx.end(), [&candidate_dists](int i1, int i2){return candidate_dists[i1] < candidate_dists[i2];});
        } else {
            sort(candidate_sort_idx.begin(), candidate_sort_idx.end(),  [&candidate_dists](int i1, int i2){return candidate_dists[i1] < candidate_dists[i2];});
        }

        gettimeofday(&end_query, NULL);
        query_time += (1000000 * (end_query.tv_sec - start_query.tv_sec) + end_query.tv_usec - start_query.tv_usec);

        for (int j = 0; j < k_size && j < candidate_idx.size(); j++) {
            queryknn_results[i][j] = candidate_idx[candidate_sort_idx[j]];
        }

        fill(collision_count.begin(), collision_count.end(), 0);

        // cout << "Finish the " << i + 1 << "-th query." << endl;
        ++pd_query;
    }
}

void ann_query_method1(float ** &dataset, int ** &queryknn_results, long int dataset_size, int data_dimensionality, int query_size, int k_size, float ** &querypoints, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, float * &centroids_list, int subspace_num, int subspace_dimensionality, int kmeans_num_centroid, int kmeans_dim, int collision_num, int candidate_num, int number_of_threads, long int &query_time, vector<vector<vector<int>>> &subspace_candidates, vector<vector<vector<int>>> &subspace_scores, const vector<int>& excluded_subspaces, int top_n_subspaces, std::vector<std::vector<int>>& chosen_subspaces_by_query) {
	std::cout << "method 1" << std::endl;
    struct timeval start_query, end_query;
    
    progress_display pd_query(query_size);

    subspace_candidates.resize(query_size);
    subspace_scores.resize(query_size);
    chosen_subspaces_by_query.resize(query_size);

    vector<int> collision_count(dataset_size, 0);
    
    for (int i = 0; i < query_size; i++) {
	    //std::cout << "query num: " << i << std::endl;
        gettimeofday(&start_query, NULL);

        subspace_candidates[i].resize(subspace_num);
        subspace_scores[i].resize(subspace_num);

        vector<int> selected_subspaces;
        for (int j = 0; j < top_n_subspaces && j < subspace_num; j++) {
            if (find(excluded_subspaces.begin(), excluded_subspaces.end(), j) != excluded_subspaces.end()) {
                continue;
            }
            selected_subspaces.push_back(j);
        }

        chosen_subspaces_by_query[i] = selected_subspaces;
	//std::cout << "[selected subspace num] ";
	//for (int j = 0; j < top_n_subspaces && j < subspace_distances.size(); j++){
//		std::cout << selected_subspaces[j]<< "," ;
//	}
//	std::cout << std::endl;

        for (int subspace_idx : selected_subspaces) {
		//std::cout << "subspace idx: " << subspace_idx << std::endl;
            // first half dist
            vector<float> first_half_dists(kmeans_num_centroid);
            for (int z = 0; z < kmeans_num_centroid; z++) {
                first_half_dists[z] = euclidean_distance(&querypoints[i][subspace_idx * subspace_dimensionality], &centroids_list[subspace_idx * 2 * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // first half sort
            vector<int> first_half_idx(kmeans_num_centroid);
            iota(first_half_idx.begin(), first_half_idx.end(), 0);
            sort(first_half_idx.begin(), first_half_idx.end(), [&first_half_dists](int i1, int i2) {return first_half_dists[i1] < first_half_dists[i2];});

            // second half dist
            vector<float> second_half_dists(kmeans_num_centroid);
            for (int z = 0; z < kmeans_num_centroid; z++) {
                second_half_dists[z] = euclidean_distance(&querypoints[i][subspace_idx * subspace_dimensionality + kmeans_dim], &centroids_list[(subspace_idx * 2 + 1) * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // second half sort
            vector<int> second_half_idx(kmeans_num_centroid);
            iota(second_half_idx.begin(), second_half_idx.end(), 0);
            sort(second_half_idx.begin(), second_half_idx.end(), [&second_half_dists](int i1, int i2) {return second_half_dists[i1] < second_half_dists[i2];});


            // dynamic activate algorithm (multi thread)
            vector<pair<int, int>> retrieved_cell;
            dynamic_activate(indexes, retrieved_cell, first_half_dists, first_half_idx, second_half_dists, second_half_idx, collision_num, kmeans_num_centroid, subspace_idx);

            // count collision and store subspace candidates
            vector<int>& current_subspace_candidates = subspace_candidates[i][subspace_idx];
            vector<int>& current_subspace_scores = subspace_scores[i][subspace_idx];
            for (int z = 0; z < retrieved_cell.size(); z++) {
                auto iterator = indexes[subspace_idx].find(retrieved_cell[z]);
                if (iterator != indexes[subspace_idx].end()) {
                    current_subspace_candidates.insert(current_subspace_candidates.end(), iterator->second.begin(), iterator->second.end());
                    current_subspace_scores.insert(current_subspace_scores.end(), iterator->second.size(), 1);
                }
            }

            // Remove duplicates
            sort(current_subspace_candidates.begin(), current_subspace_candidates.end());
            current_subspace_candidates.erase(unique(current_subspace_candidates.begin(), current_subspace_candidates.end()), current_subspace_candidates.end());

            // Update collision_count using the unique candidates
            #pragma omp parallel for num_threads(number_of_threads)
            for (int z = 0; z < current_subspace_candidates.size(); z++) {
                #pragma omp atomic
                collision_count[current_subspace_candidates[z]]++;
            }
        }

        int * collision_num_array = new int[top_n_subspaces + 1]();
        int ** local_collision_num = new int * [number_of_threads];
        for (int j = 0; j < number_of_threads; j++) {
            local_collision_num[j] = new int [top_n_subspaces + 1]();
        }

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < dataset_size; j++) {
            int id = omp_get_thread_num();
            if (collision_count[j] > top_n_subspaces) {
                local_collision_num[id][top_n_subspaces]++;
            } else {
                local_collision_num[id][collision_count[j]]++;
            }
        }

        for (int j = 0; j < top_n_subspaces + 1; j++) {
            for (int z = 0; z < number_of_threads; z++) {
                collision_num_array[j] += local_collision_num[z][j];
            }
        }

        // release the candidate number to include all points in last_collision_num, saving the time for checking points whose collision_num is last_collision_num
        int last_collision_num = 1;
        int sum_candidate = 0;
        for (int j = top_n_subspaces; j >= 1; j--) {
            sum_candidate += collision_num_array[j];
            if (sum_candidate >= candidate_num) {
                last_collision_num = j;
                break;
            }
        }

        vector<int> candidate_idx;
        vector<vector<int>> local_candidate_idx(number_of_threads);

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < dataset_size; j++) {
            int id = omp_get_thread_num();
            if (collision_count[j] >= last_collision_num) {
                local_candidate_idx[id].push_back(j);
            }
        }

        for (int j = 0; j < number_of_threads; j++) {
            candidate_idx.insert(candidate_idx.end(), local_candidate_idx[j].begin(), local_candidate_idx[j].end());
        }

        vector<float> candidate_dists(candidate_idx.size());

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < candidate_idx.size(); j++) {
            // candidate_dists[j] = euclidean_distance(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
            // candidate_dists[j] = euclidean_distance_SIMD(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
            candidate_dists[j] = faiss::fvec_L2sqr_avx512(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
        }

        vector<int> candidate_sort_idx(candidate_idx.size());
        iota(candidate_sort_idx.begin(), candidate_sort_idx.end(), 0);
        if (k_size < candidate_idx.size()) {
            partial_sort(candidate_sort_idx.begin(), candidate_sort_idx.begin() + k_size, candidate_sort_idx.end(), [&candidate_dists](int i1, int i2){return candidate_dists[i1] < candidate_dists[i2];});
        } else {
            sort(candidate_sort_idx.begin(), candidate_sort_idx.end(),  [&candidate_dists](int i1, int i2){return candidate_dists[i1] < candidate_dists[i2];});
        }

        gettimeofday(&end_query, NULL);
        query_time += (1000000 * (end_query.tv_sec - start_query.tv_sec) + end_query.tv_usec - start_query.tv_usec);

        for (int j = 0; j < k_size && j < candidate_idx.size(); j++) {
            queryknn_results[i][j] = candidate_idx[candidate_sort_idx[j]];
        }

        fill(collision_count.begin(), collision_count.end(), 0);

        // cout << "Finish the " << i + 1 << "-th query." << endl;
        ++pd_query;
    }
}

void ann_query_method2(float ** &dataset, int ** &queryknn_results, long int dataset_size, int data_dimensionality, int query_size, int k_size, float ** &querypoints, vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, float * &centroids_list, int subspace_num, int subspace_dimensionality, int kmeans_num_centroid, int kmeans_dim, int collision_num, int candidate_num, int number_of_threads, long int &query_time, vector<vector<vector<int>>> &subspace_candidates, vector<vector<vector<int>>> &subspace_scores, const vector<int>& excluded_subspaces, int top_n_subspaces) {
    struct timeval start_query, end_query;
    
    progress_display pd_query(query_size);

    subspace_candidates.resize(query_size);
    subspace_scores.resize(query_size);

    for (int i = 0; i < query_size; i++) {
        gettimeofday(&start_query, NULL);

        subspace_candidates[i].resize(subspace_num);
        subspace_scores[i].resize(subspace_num);

        vector<pair<float, int>> subspace_distances;
        for (int j = 0; j < subspace_num; j++) {
            if (find(excluded_subspaces.begin(), excluded_subspaces.end(), j) != excluded_subspaces.end()) {
                continue;
            }
            float min_dist = FLT_MAX;
            for (int z = 0; z < kmeans_num_centroid; z++) {
                float dist = euclidean_distance(&querypoints[i][j * subspace_dimensionality], &centroids_list[j * 2 * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            subspace_distances.push_back({min_dist, j});
        }

        sort(subspace_distances.begin(), subspace_distances.end());

        vector<int> selected_subspaces;
        for (int j = 0; j < top_n_subspaces && j < subspace_distances.size(); j++) {
            selected_subspaces.push_back(subspace_distances[j].second);
        }
//	std::cout << "[selected subspace num] ";
//	for (int j = 0; j < top_n_subspaces && j < subspace_distances.size(); j++){
//		std::cout << selected_subspaces[j]<< "," ;
//	}
//	std::cout << std::endl;

        vector<float> candidate_scores(dataset_size, 0.0f);

        for (int subspace_idx : selected_subspaces) {
            // Find the distance for the current subspace
            float current_subspace_dist = 0.0f;
            for(const auto& p : subspace_distances) {
                if (p.second == subspace_idx) {
                    current_subspace_dist = p.first;
                    break;
                }
            }
            float score_to_add = 1.0f / (current_subspace_dist + 1e-6);

            // first half dist
            vector<float> first_half_dists(kmeans_num_centroid);
            for (int z = 0; z < kmeans_num_centroid; z++) {
                first_half_dists[z] = euclidean_distance(&querypoints[i][subspace_idx * subspace_dimensionality], &centroids_list[subspace_idx * 2 * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // first half sort
            vector<int> first_half_idx(kmeans_num_centroid);
            iota(first_half_idx.begin(), first_half_idx.end(), 0);
            sort(first_half_idx.begin(), first_half_idx.end(), [&first_half_dists](int i1, int i2) {return first_half_dists[i1] < first_half_dists[i2];});

            // second half dist
            vector<float> second_half_dists(kmeans_num_centroid);
            for (int z = 0; z < kmeans_num_centroid; z++) {
                second_half_dists[z] = euclidean_distance(&querypoints[i][subspace_idx * subspace_dimensionality + kmeans_dim], &centroids_list[(subspace_idx * 2 + 1) * kmeans_num_centroid * kmeans_dim + z * kmeans_dim], kmeans_dim);
            }

            // second half sort
            vector<int> second_half_idx(kmeans_num_centroid);
            iota(second_half_idx.begin(), second_half_idx.end(), 0);
            sort(second_half_idx.begin(), second_half_idx.end(), [&second_half_dists](int i1, int i2) {return second_half_dists[i1] < second_half_dists[i2];});


            // dynamic activate algorithm (multi thread)
            vector<pair<int, int>> retrieved_cell;
            dynamic_activate(indexes, retrieved_cell, first_half_dists, first_half_idx, second_half_dists, second_half_idx, collision_num, kmeans_num_centroid, subspace_idx);

            // count collision and store subspace candidates
            vector<int>& current_subspace_candidates = subspace_candidates[i][subspace_idx];
            vector<int>& current_subspace_scores = subspace_scores[i][subspace_idx];
            for (int z = 0; z < retrieved_cell.size(); z++) {
                auto iterator = indexes[subspace_idx].find(retrieved_cell[z]);
                if (iterator != indexes[subspace_idx].end()) {
                    current_subspace_candidates.insert(current_subspace_candidates.end(), iterator->second.begin(), iterator->second.end());
                    current_subspace_scores.insert(current_subspace_scores.end(), iterator->second.size(), 1);
                }
            }

            // Remove duplicates
            sort(current_subspace_candidates.begin(), current_subspace_candidates.end());
            current_subspace_candidates.erase(unique(current_subspace_candidates.begin(), current_subspace_candidates.end()), current_subspace_candidates.end());

            // Update candidate_scores using the unique candidates
            #pragma omp parallel for num_threads(number_of_threads)
            for (int z = 0; z < current_subspace_candidates.size(); z++) {
                #pragma omp atomic
                candidate_scores[current_subspace_candidates[z]] += score_to_add;
            }
        }

        vector<int> non_zero_score_indices;
        for(int j=0; j < dataset_size; ++j) {
            if (candidate_scores[j] > 0.0f) {
                non_zero_score_indices.push_back(j);
            }
        }

        int num_to_select = min((int)non_zero_score_indices.size(), candidate_num);
        
        partial_sort(non_zero_score_indices.begin(), non_zero_score_indices.begin() + num_to_select, non_zero_score_indices.end(),
                     [&candidate_scores](int i1, int i2) {
                         return candidate_scores[i1] > candidate_scores[i2]; // Higher score first
                     });

        vector<int> candidate_idx;
        if (num_to_select > 0) {
            candidate_idx.assign(non_zero_score_indices.begin(), non_zero_score_indices.begin() + num_to_select);
        }

        vector<float> candidate_dists(candidate_idx.size());

        #pragma omp parallel for num_threads(number_of_threads)
        for (int j = 0; j < candidate_idx.size(); j++) {
            // candidate_dists[j] = euclidean_distance(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
            // candidate_dists[j] = euclidean_distance_SIMD(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
            candidate_dists[j] = faiss::fvec_L2sqr_avx512(querypoints[i], dataset[candidate_idx[j]], data_dimensionality);
        }

        vector<int> candidate_sort_idx(candidate_idx.size());
        iota(candidate_sort_idx.begin(), candidate_sort_idx.end(), 0);
        if (k_size < candidate_idx.size()) {
            partial_sort(candidate_sort_idx.begin(), candidate_sort_idx.begin() + k_size, candidate_sort_idx.end(), [&candidate_dists](int i1, int i2){return candidate_dists[i1] < candidate_dists[i2];});
        } else {
            sort(candidate_sort_idx.begin(), candidate_sort_idx.end(), [&candidate_dists](int i1, int i2){return candidate_dists[i1] < candidate_dists[i2];});
        }

        gettimeofday(&end_query, NULL);
        query_time += (1000000 * (end_query.tv_sec - start_query.tv_sec) + end_query.tv_usec - start_query.tv_usec);

        for (int j = 0; j < k_size && j < candidate_idx.size(); j++) {
            queryknn_results[i][j] = candidate_idx[candidate_sort_idx[j]];
        }

        // cout << "Finish the " << i + 1 << "-th query." << endl;
        ++pd_query;
    }
}



void dynamic_activate(vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> &indexes, vector<pair<int, int>> &retrieved_cell, vector<float> &first_half_dists, vector<int> &first_half_idx, vector<float> &second_half_dists, vector<int> &second_half_idx, int collision_num, int kmeans_num_centroid, int subspace_idx) {
    vector<pair<float, int>> activated_cell;

    int retrieved_num = 0;
    activated_cell.push_back(pair<float, int>(first_half_dists[first_half_idx[0]] + second_half_dists[second_half_idx[0]], 0));
    while (true) {
        int cell_position = min_element(activated_cell.begin(), activated_cell.end()) - activated_cell.begin();
        auto iterator = indexes[subspace_idx].find(pair<int, int>(first_half_idx[cell_position], second_half_idx[activated_cell[cell_position].second]));
        if (iterator != indexes[subspace_idx].end() && activated_cell[cell_position].first < FLT_MAX) {
            retrieved_cell.push_back(pair<int, int>(first_half_idx[cell_position], second_half_idx[activated_cell[cell_position].second]));

            retrieved_num += iterator->second.size();

            if (retrieved_num >= collision_num) {
                break;
            }
        }

        if (activated_cell[cell_position].second == 0 && cell_position < kmeans_num_centroid - 1) {
            activated_cell.push_back(pair<float, int>(first_half_dists[first_half_idx[cell_position + 1]] + second_half_dists[second_half_idx[0]], 0));
        }

        if (activated_cell[cell_position].second < kmeans_num_centroid - 1) {
            activated_cell[cell_position].second++;
            activated_cell[cell_position].first = first_half_dists[first_half_idx[cell_position]] + second_half_dists[second_half_idx[activated_cell[cell_position].second]];
        } else {
            activated_cell[cell_position].first = FLT_MAX;
        }
    }
}
