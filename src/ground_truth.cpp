#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <getopt.h>
#include "preprocess.h"
#include "dist_calculation.h"

using namespace std;

void save_groundtruth(const char* filename, long int** data, int num_vectors, int k) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        cout << "Cannot open file " << filename << " for writing" << endl;
        return;
    }
    for (int i = 0; i < num_vectors; ++i) {
        fwrite(data[i], sizeof(long int), k, file);
    }
    fclose(file);
}

int main(int argc, char* argv[]) {
    const char* dataset_path = nullptr;
    const char* query_path = nullptr;
    const char* ground_truth_path = nullptr;
    long int dataset_size = 0;
    int query_size = 0;
    int dimensionality = 0;
    int k = 0;

    int opt;
    while ((opt = getopt(argc, argv, "d:q:g:n:m:D:k:")) != -1) {
        switch (opt) {
            case 'd':
                dataset_path = optarg;
                break;
            case 'q':
                query_path = optarg;
                break;
            case 'g':
                ground_truth_path = optarg;
                break;
            case 'n':
                dataset_size = atol(optarg);
                break;
            case 'm':
                query_size = atoi(optarg);
                break;
            case 'D':
                dimensionality = atoi(optarg);
                break;
            case 'k':
                k = atoi(optarg);
                break;
            default:
                cerr << "Usage: " << argv[0] << " -d <dataset_path> -q <query_path> -g <ground_truth_path> -n <dataset_size> -m <query_size> -D <dimensionality> -k <k>" << endl;
                return 1;
        }
    }

    if (!dataset_path || !query_path || !ground_truth_path || dataset_size == 0 || query_size == 0 || dimensionality == 0 || k == 0) {
        cerr << "All arguments are required." << endl;
        cerr << "Usage: " << argv[0] << " -d <dataset_path> -q <query_path> -g <ground_truth_path> -n <dataset_size> -m <query_size> -D <dimensionality> -k <k>" << endl;
        return 1;
    }

    float** dataset = nullptr;
    float** query_points = nullptr;

    load_data(dataset, const_cast<char*>(dataset_path), dataset_size, dimensionality);
    load_query(query_points, const_cast<char*>(query_path), query_size, dimensionality);

    long int** ground_truth = new long int*[query_size];
    for (int i = 0; i < query_size; ++i) {
        ground_truth[i] = new long int[k];
    }

    for (int i = 0; i < query_size; ++i) {
        priority_queue<pair<float, long int>> top_k;
        for (long int j = 0; j < dataset_size; ++j) {
            float dist = euclidean_distance(query_points[i], dataset[j], dimensionality);
            if (top_k.size() < k) {
                top_k.push({dist, j});
            } else if (dist < top_k.top().first) {
                top_k.pop();
                top_k.push({dist, j});
            }
        }

        vector<long int> neighbors;
        while (!top_k.empty()) {
            neighbors.push_back(top_k.top().second);
            top_k.pop();
        }
        reverse(neighbors.begin(), neighbors.end());
        for (int j = 0; j < k; ++j) {
            ground_truth[i][j] = neighbors[j];
        }
        if ((i + 1) % 100 == 0) {
            cout << "Processed " << (i + 1) << "/" << query_size << " queries." << endl;
        }
    }

    save_groundtruth(ground_truth_path, ground_truth, query_size, k);

    cout << "Ground truth generated successfully." << endl;

    for (long int i = 0; i < dataset_size; ++i) {
        delete[] dataset[i];
    }
    delete[] dataset;

    for (int i = 0; i < query_size; ++i) {
        delete[] query_points[i];
    }
    delete[] query_points;

    for (int i = 0; i < query_size; ++i) {
        delete[] ground_truth[i];
    }
    delete[] ground_truth;

    return 0;
}
