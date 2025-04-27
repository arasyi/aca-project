#!/bin/bash

# --- Configuration ---
# Base directory containing the Python workload scripts
WORKLOAD_DIR="workloads"

# gem5 related paths
GEM5_ROOT="gem5"
GEM5_BUILD_DIR="$GEM5_ROOT/build/VEGA_X86"
GEM5_BIN="$GEM5_BUILD_DIR/gem5.opt"
GEM5_CONFIG="configs/mi200.py"

# gem5 resources paths
RESOURCES_ROOT="gem5-resources"
DISK_IMAGE="$RESOURCES_ROOT/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml"
KERNEL="$RESOURCES_ROOT/src/x86-ubuntu-gpu-ml/vmlinux-gpu-ml"

# Base directory for gem5 outputs
OUTPUT_BASE_DIR="outputs"
# --- End Configuration ---

# --- Helper Functions ---
# Function to print error messages and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# --- Main Script Logic ---

echo "----------------------------------------"
echo "GEM5 HETEROGENEOUS SYSTEM SIMULATIONS"
echo "----------------------------------------"

# 1. Check if necessary directories/files exist
echo "Verifying paths..."
[ ! -d "$WORKLOAD_DIR" ] && error_exit "Workload directory '$WORKLOAD_DIR' not found."
[ ! -f "$GEM5_BIN" ] && error_exit "gem5 binary '$GEM5_BIN' not found."
[ ! -f "$GEM5_CONFIG" ] && error_exit "gem5 config script '$GEM5_CONFIG' not found."
[ ! -f "$DISK_IMAGE" ] && error_exit "Disk image '$DISK_IMAGE' not found."
[ ! -f "$KERNEL" ] && error_exit "Kernel '$KERNEL' not found."
[ ! -d "$OUTPUT_BASE_DIR" ] && echo "Info: Output base directory '$OUTPUT_BASE_DIR' not found. It will be created by gem5."
echo "Paths verified."
echo "----------------------------------------"

# 2. Read Python scripts from the workload directory
echo "Looking for Python scripts in '$WORKLOAD_DIR'..."
shopt -s nullglob # Prevent errors if no files match
scripts=("$WORKLOAD_DIR"/*.py)
shopt -u nullglob # Turn off nullglob

# Check if any scripts were found
if [ ${#scripts[@]} -eq 0 ]; then
    error_exit "No Python scripts (*.py) found in '$WORKLOAD_DIR'."
fi

# 3. Display the list and ask the user to choose
echo "Available Python scripts:"
options=()
for script_path in "${scripts[@]}"; do
    # Extract just the filename for display
    options+=("$(basename "$script_path")")
done
options+=("Quit") # Add a Quit option

PS3="Enter the number of the script to run (or choose Quit): " # Set the prompt for select
select choice in "${options[@]}"; do
    if [[ "$choice" == "Quit" ]]; then
        echo "Exiting."
        exit 0
    elif [[ -n "$choice" ]]; then
        # Find the full path corresponding to the chosen basename
        chosen_script_basename="$choice"
        chosen_script_path=""
        for script_path in "${scripts[@]}"; do
            if [[ "$(basename "$script_path")" == "$chosen_script_basename" ]]; then
                chosen_script_path="$script_path"
                break
            fi
        done

        if [[ -z "$chosen_script_path" ]]; then
             # This should ideally not happen with 'select' but good to check
             error_exit "Internal error: Could not determine path for selection '$choice'."
        fi

        echo "You selected: $chosen_script_basename"
        echo "Full path: $chosen_script_path"
        break # Exit the select loop after a valid choice
    else
        echo "Invalid choice '$REPLY'. Please try again."
    fi
done
echo "----------------------------------------"

# 4. Ask for additional parameters
read -p "Enter any additional parameters for '$chosen_script_basename' (leave blank for none): " additional_params
echo "----------------------------------------"

# 5. Generate the output directory name
# Remove the .py extension for the directory name base
script_name_no_ext="${chosen_script_basename%.py}"
timestamp=$(date +"%Y%m%d%H%M%S")
output_dir="${script_name_no_ext}-${timestamp}"
full_output_path="${OUTPUT_BASE_DIR}/${output_dir}" # Relative to where the script is run

echo "Output directory will be: $full_output_path"
echo "----------------------------------------"

# 6. Construct and display the gem5 command
# Use an array for safer command construction, especially with spaces in paths or params
gem5_cmd=(
    "$GEM5_BIN"
    -d "$full_output_path" # Output directory for gem5 logs/stats
    "$GEM5_CONFIG"
    --kvm-perf
    --disk-image "$DISK_IMAGE"
    --kernel "$KERNEL"
    --app "$chosen_script_path" # The user-selected Python script
)

# Add additional parameters IF they were provided
# These are appended *after* --app, assuming they are arguments for the Python script itself
if [[ -n "$additional_params" ]]; then
    # If parameters contain spaces and need to be passed as a single argument,
    # the user should quote them when entering them (e.g., --message "Hello World")
    #
    # This uses xargs to parse the string and read null-delimited output into the array
    # readarray/mapfile requires Bash 4+
    # xargs -0 passes null-separated arguments to printf
    # printf '%s\0' prints each argument null-separated
    # readarray -t -d '' reads null-separated input into the array
    readarray -t -d '' tmp_args < <(xargs printf '%s\0' <<<"$additional_params")
    gem5_cmd+=("${tmp_args[@]}")
fi

echo "The following command will be executed:"
# Print the command in a readable format, quoting arguments
printf "%q " "${gem5_cmd[@]}"
echo # Add a newline for clarity
echo "----------------------------------------"

# 7. Execute the command
echo "Starting gem5 simulation..."

# Run the command
"${gem5_cmd[@]}"

# Capture exit status
exit_status=$?

echo "----------------------------------------"
if [ $exit_status -eq 0 ]; then
    echo "gem5 simulation finished successfully."
    echo "Output can be found in: $full_output_path"
else
    echo "gem5 simulation failed with exit status $exit_status."
    echo "Check logs in: $full_output_path"
fi
echo "----------------------------------------"

exit $exit_status