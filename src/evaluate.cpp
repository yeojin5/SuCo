#include "evaluate.h"

void recall_and_ratio(float ** &dataset, float ** &querypoints, int data_dimensionality, int ** &queryknn_results, int ** &gt, int query_size) {
    int ks[6] = {1, 10, 20, 30, 40, 50};
    
    for (int k_index = 0; k_index < sizeof(ks) / sizeof(ks[0]); k_index++) {
        int retrieved_data_num = 0;

        for (int i = 0; i < query_size; i++)
        {
            for (int j = 0; j < ks[k_index]; j++)
            {
                for (int z = 0; z < ks[k_index]; z++) {
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
