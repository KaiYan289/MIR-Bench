#!/bin/bash

# Specify the directory containing the Python files
directory="."

# Set the timeout duration in seconds (60 seconds = 1 minute)
timeout_duration=45

# Iterate over each .py file in the directory and execute it
for file in "$directory"/*.py
do
  # Check if the file exists to handle empty directories
  if [[ -f $file ]]; then
    echo "Executing $file..."
    # Run the Python file with a timeout
    timeout $timeout_duration python3 "$file"
    # Check the exit status of the timeout command
    if [[ $? -eq 124 ]]; then
      echo "Execution of $file timed out after $timeout_duration seconds."
    else
      echo "Execution of $file completed."
    fi
  else
    echo "No Python files found in the directory."
  fi
done
