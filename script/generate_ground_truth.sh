#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Build the ground_truth executable
echo "Building ground_truth executable..."
make ground_truth
echo "Build complete."

# --- Configuration ---
K=50 # Number of nearest neighbors to find

# Dataset configurations
# Format: "dataset_name dataset_size query_size dimensions"
DATASETS=(
    "deep1m 1000000 1000 96"
    "gist1m 1000000 1000 960"
    "openai1m 1000000 1000 1536"
    "sift1m 1000000 1000 128"
    "sift10m 10000000 1000 128"
)

# --- Ground Truth Generation ---
for config in "${DATASETS[@]}"; do
    # Split the config string into variables
    read -r name size query_size dim <<< "$config"

    echo "--------------------------------------------------"
    echo "Generating ground truth for $name..."
    echo "--------------------------------------------------"

    DATASET_PATH="dataset/${name}/${name}_base.fbin"
    QUERY_PATH="dataset/${name}/${name}_query.fbin"
    GT_PATH="dataset/${name}/${name}_gt_K${K}.fbin"

    # Check if required files exist
    if [ ! -f "$DATASET_PATH" ]; then
        echo "ERROR: Dataset file not found at $DATASET_PATH"
        continue
    fi
    if [ ! -f "$QUERY_PATH" ]; then
        echo "ERROR: Query file not found at $QUERY_PATH"
        continue
    fi

    # Run the ground truth generation command
    ./ground_truth \
        -d "$DATASET_PATH" \
        -q "$QUERY_PATH" \
        -g "$GT_PATH" \
        -n "$size" \
        -m "$query_size" \
        -D "$dim" \
        -k "$K"

    echo "Successfully generated ground truth for $name at $GT_PATH"
done

echo "--------------------------------------------------"
echo "All ground truth generation tasks are complete."
echo "--------------------------------------------------"
