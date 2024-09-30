#!/bin/bash

# Set Python path
export PYTHONPATH=$(pwd)

# Load environment variables
source venv/bin/activate

# Run the Streamlit app
streamlit run sl/multiagent_debate_app.py
