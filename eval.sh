#!/bin/bash

source /Users/aghamatlabakbarzade/Documents/Courses/ECE4903/venv/bin/activate
python3 /Users/aghamatlabakbarzade/Documents/Courses/ECE4903/rl-starter-files/plot_evaluation.py --path "${1}"
python3 /Users/aghamatlabakbarzade/Documents/Courses/ECE4903/rl-starter-files/plot_training.py --path "${1}"
