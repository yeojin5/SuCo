#!/bin/bash

# This script runs all test scripts in the script directory.

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <K_SIZE> <alpha> <beta>"
    exit 1
fi

K_SIZE=$1
ALPHA=$2
BETA=$3

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the test scripts with arguments
bash "$SCRIPT_DIR/deep1m_test.sh" $K_SIZE $ALPHA $BETA
bash "$SCRIPT_DIR/gist1m_test.sh" $K_SIZE $ALPHA $BETA
bash "$SCRIPT_DIR/openai1m_test.sh" $K_SIZE $ALPHA $BETA
bash "$SCRIPT_DIR/sift10m_test.sh" $K_SIZE $ALPHA $BETA
