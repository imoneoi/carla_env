#!/bin/bash

# Set the initial step number
# Check if the step file exists
if [ -f "$step_file" ]; then
    # Read the step number from the file
    step=$(cat "$step_file")
else
    # Set step to 0 if the file does not exist
    step=0
fi

# current_datetime=$(date +'%Y-%m-%d %H:%M:%S')
total_n=30
step_file="step.txt"

while true; do
    # Save the current step number
    prev_step=$step
    remain_n=$((total_n - prev_step))

    # Run your Python script here
    python record_ai_assisted_dataset.py --n $remain_n --n_jobs 1 --eps 0.0
    
    # Check the exit status of the Python script
    if [ $? -ne 0 ]; then
        # Handle the case when the script fails
        echo "Python script exited unexpectedly!"
        echo "Last recorded step: $prev_step"
        break
    fi

    # Check if the step file exists
    if [ -f "$step_file" ]; then
        # Read the step number from the file
        step=$(cat "$step_file")
    fi

    # Check if the collection step exceeds the total limit
    if [ $step -ge $total_n ]; then
        echo "Collection step reached the total limit!"
        break
    fi

    # Check if CARLA GUI is running
    if pgrep -x "CarlaUE4" >/dev/null; then
        # CARLA GUI is running, so kill the process
        pkill -f CarlaUE4
    fi

    # Add a delay between iterations if needed
    sleep 0.5
done

# # Set the path to the directory containing the JPG images
# image_dir="../dataset/bev_23070623/0"

# # Get the current datetime
# datetime=$(date +"%Y%m%d%H%M%S")

# # Set the output file name using the datetime
# output_file="${datetime}.mp4"

# # Run FFmpeg command to create the MP4 video
# ffmpeg -framerate 30 -start_number 1 -i "${image_dir}/bev_rgb_%d.jpg" -c:v libx264 -r 30 -pix_fmt yuv420p "${output_file}_0"
