#!/bin/bash

# Set up virtual environment
echo "Setting up virtual environment..."
python -m venv debate_env
source debate_env/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create results directory
RESULTS_DIR="debate_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Run experiments with different repeat values
for repeat in {1..10}; do
    echo "Running experiment with $repeat repeats..."
    
    # Run the evaluation script
    python eval/eval-gsm8k-debate-rep.py \
        --use_ollama \
        --num_samples 100 \
        --repeats $repeat \
        2>&1 | tee "$RESULTS_DIR/debate_log_repeat_${repeat}.txt"
    
    # Move the results file to the results directory
    mv debate_results_fs.json "$RESULTS_DIR/debate_results_repeat_${repeat}.json"
    
    # Sleep for a minute to prevent rate limiting and cool down
    # echo "Sleeping for 60 seconds before next run..."
    # sleep 60
done

# Deactivate virtual environment
deactivate

echo "All experiments completed. Results are in $RESULTS_DIR"