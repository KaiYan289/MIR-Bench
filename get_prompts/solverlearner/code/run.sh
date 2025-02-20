#!/bin/bash

# Specify the directory containing the Python files
directory="."

# Set the timeout duration in seconds (60 seconds = 1 minute)
timeout_duration=1

# Reset the SECONDS variable to track script running time
SECONDS=0

# Find all .py files in the directory and its subdirectories
find "$directory" -name "*.py" | while read -r file
do
  # Check if the file exists to handle empty directories
  if [[ -f $file ]]; then
    # Generate the corresponding .txt file path by replacing .py with .txt
    txt_file="${file%.py}.txt"

    # Get the directory of the txt_file and ensure it exists
    txt_dir=$(dirname "$txt_file")
    mkdir -p "$txt_dir"

    echo "Executing $file, outputting to $txt_file..."

    # Run the Python file with a timeout and redirect output to the .txt file
    timeout $timeout_duration python3 "$file" > "$txt_file" 2>&1

    # Check the exit status of the timeout command
    if [[ $? -eq 124 ]]; then
      echo "Execution of $file timed out after $timeout_duration seconds."
      echo "Execution of $file timed out after $timeout_duration seconds." >> "$txt_file"
    else
      echo "Execution of $file completed."
      echo "Execution of $file completed." >> "$txt_file"
    fi

    # Output the current running time of the bash script
    current_time=$SECONDS
    echo "Current script running time: $current_time seconds"
  else
    echo "No Python files found in the directory."
  fi
done
