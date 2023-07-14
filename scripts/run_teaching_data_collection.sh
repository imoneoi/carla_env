#!/bin/bash

# continue to record data from prev step
continue_record=$1
# total recording iterations
total_n=$2
# data recording path
data_path=$3

step_file="step.txt"
# Set the initial step number
# Check if the step file exists
if [ -f "$step_file" ] && [ "$continue_record" = true ]; then
    # Read the step number from the file
    step=$(cat "$step_file")
else
    # Set step to 0 if the file does not exist or record data from the very beginning
    step=0
    echo "0" > "$step_file"  # Write "0" to the step file
fi

while true; do
    # Save the current step number
    prev_step=$step
    remain_n=$((total_n - prev_step))

    # Run your Python script here
    python record_ai_assisted_dataset.py --save_path $data_path --n $remain_n --n_jobs 1 --timeout 500 --eps 0.0
    
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

    # step=$(expr "$step" + 0)
    # total_n=$(expr "$total_n" + 0)
    # Check if the collection step exceeds the total limit
    if [ "$step" -ge "$total_n" ]; then
        echo "Collection step reached the total limit!"
        break
    fi

    # # Check if CARLA GUI is running
    # if pgrep -x "CarlaUE4" >/dev/null; then
    #     # CARLA GUI is running, so kill the process
    #     pkill -f CarlaUE4
    #     echo "Kill the remaining GUI!"
    # fi

    ## Get the list of CARLA GUI processes
    processes=$(pgrep -x "CarlaUE4")

    # Check the number of CARLA GUI processes
    process_count=$(echo "$processes" | wc -w)

    if [[ $process_count -gt 1 ]]; then
        # Get the newest CARLA GUI process PID
        newest_pid=$(echo "$processes" | tail -n 1)

        # Kill all other CARLA GUI processes except the newest one
        pkill -f "CarlaUE4" -P $newest_pid

        echo "Killed all CARLA GUI processes except the newest one (PID: $newest_pid)"
    else
        echo "Only one CARLA GUI process found. No action needed."
    fi

    # Add a delay between iterations if needed
    sleep 0.5
done

# Set the path to the directory containing the JPG images
image_dir="../dataset/$data_path/0"

# Get the current datetime
datetime=$(date +"%Y%m%d%H%M%S")

# Set the output file name using the datetime
output_file="${datetime}.mp4"

# Run FFmpeg command to create the MP4 video
ffmpeg -framerate 30 -start_number 1 -i "${image_dir}/seg_bev_rgb_%d.jpg" -c:v libx264 -r 30 -pix_fmt yuv420p "${output_file}"

python plot_traj.py --data_path $data_path --n_jobs 1 
