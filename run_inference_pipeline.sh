#!/bin/bash

num_units_range=(220 230 240 250 260 270)

models="Both"
mouse_name="Angie"
num_sims=45000  # 30k sims
use_prev_params=True

python_script="python/src/inference_pipeline.py"

for num_units in "${num_units_range[@]}"; do
    echo "Running inference pipeline with N=$num_units"
    python $python_script \
        --models "$models" \
        --mouse_name "$mouse_name" \
        --num_units $num_units \
        --num_sims $num_sims \
        --use_prev_params $use_prev_params
    echo "Completed run with N=$num_units"
    echo "-----------------------------"
done
