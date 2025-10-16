#!/bin/bash

# Script to get the average reward for a given model directory

if [ -z "$1" ]; then
  echo "Usage: $0 <directory_name>"
  exit 1
fi

DIR_NAME=$1
LOG_FILE="storage/$DIR_NAME/log.csv"

if [ ! -f "$LOG_FILE" ]; then
  echo "Error: log.csv not found in $DIR_NAME"
  exit 1
fi

# Get the header to find the column number for return_mean
HEADER=$(head -n 1 "$LOG_FILE")
# Find the column number of return_mean
COLUMN_NUM=$(echo "$HEADER" | tr ',' '\n' | nl | grep -w "return_mean" | awk '{print $1}')

if [ -z "$COLUMN_NUM" ]; then
    echo "Error: return_mean column not found in log.csv"
    exit 1
fi

# Calculate the average of the return_mean column
# Exclude the header row with tail -n +2
AVG_REWARD=$(tail -n +2 "$LOG_FILE" | cut -d, -f$COLUMN_NUM | awk '{sum+=$1} END {print sum/NR}')

echo "Average reward for $DIR_NAME: $AVG_REWARD"

