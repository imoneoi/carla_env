# #!/bin/bash

# # Set the initial step number
# step=0
# current_datetime=$(date +'%Y-%m-%d %H:%M:%S')
# total_n=2000

# while true; do
#     # Check the condition
#     if [ $? -ne 0 ] || [ $step -ge $total_n ]; then
#         # Exit the loop if the Python script fails (returns non-zero exit status)
#         break
#     fi

#     # Run your Python script here
#     python record_ai_assisted_dataset.py --n $total_n --n_jobs 1 --eps 0.0 
    
#     # Add a delay between iterations if needed
#     sleep 0.5

#     # Increment the step number
#     ((step++))
# done



# #!/bin/bash

# # Set the initial step number
# step=0
# # current_datetime=$(date +'%Y-%m-%d %H:%M:%S')
# total_n=200
# step_file="step.txt"

# while true; do
#     # Save the current step number
#     prev_step = $step
#     remain_n = $total_n - $prev_step

#     # Run your Python script here
#     python record_ai_assisted_dataset.py --n $remain_n --n_jobs 1 --eps 0.0
    
#     # Check the exit status of the Python script
#     if [ $? -ne 0 ]; then
#         # Handle the case when the script fails
#         echo "Python script exited unexpectedly!"
#         echo "Last recorded step: $prev_step"
#         break
#     fi

#     # Check if the step file exists
#     if [ -f "$step_file" ]; then
#         # Read the step number from the file
#         step=$(cat "$step_file")
#     fi

#     # Save the current step number to a file
#     echo $step > $step_file

#     # Check if the collection step exceeds the total limit
#     if [ $step -ge $total_n ]; then
#         echo "Collection step reached the total limit!"
#         break
#     fi

#     # Add a delay between iterations if needed
#     sleep 0.5
# done

# # Clean up the step file if it exists
# rm -f $step_file



#!/bin/bash

# Set the initial step number
step=0
# current_datetime=$(date +'%Y-%m-%d %H:%M:%S')
total_n=210
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

    # Add a delay between iterations if needed
    sleep 0.5
done
