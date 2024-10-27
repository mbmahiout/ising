#!/bin/bash

num_units_range=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)

is_testing=False
mouse_name="Angie"
num_sims=30000  # 30k sims

python_script="python/src/inference_pipeline.py"

for num_units in "${num_units_range[@]}"; do
    echo "Running inference pipeline with N=$num_units"
    python $python_script \
        --is_testing $is_testing \
        --mouse_name "$mouse_name" \
        --num_units $num_units \
        --num_sims $num_sims
    echo "Completed run with N=$num_units"
    echo "-----------------------------"
done
