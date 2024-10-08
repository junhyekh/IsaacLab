#!/bin/bash

# Total number of samples
total_samples=26737

# Number of parallel processes (4 in this case)
n=16

# Function to run the Python script for a single index
process_index() {
    local index=$1
    python convert_usd.py $index --headless --device=cpu
}

# Loop through the indices in chunks of 4, assigning one index per process
for (( i=0; i<total_samples; i+=n ))
do
    # For each chunk of n indices, start n processes simultaneously
    for (( j=0; j<n; j++ ))
    do
        current_index=$((i + j))

        # Check if the current index exceeds the total number of samples
        if [[ $current_index -ge $total_samples ]]; then
            break
        fi

        # Run the process for this index in the background
        process_index $current_index &
    done

    # Wait for all n processes to finish before starting the next batch
    wait
done

echo "All processes complete!"