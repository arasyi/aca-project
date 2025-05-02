#!/bin/bash


# Benchmark Configurations
REPETITIONS=5 # How many repetition should be performed

# Parameters
NUM_COMPUTE_UNITS=(4 8 16 32)
GPU_CLOCKS=("1GHz" "1.5GHz" "2GHz" "2.5GHz")
DGPU_NUM_DIRS=(1 2 4 8)
TCC_SIZES=("128KiB" "256KiB" "384KiB" "512KiB")
TCP_SIZES=("8KiB" "16KiB" "32KiB" "64KiB")
NUM_TCCS=(1 2 4 8)

declare -A PARAMS
PARAMS["num-compute-units"]="NUM_COMPUTE_UNITS"
PARAMS["gpu-clock"]="GPU_CLOCKS"
PARAMS["dgpu-num-dirs"]="DGPU_NUM_DIRS"
PARAMS["tcc-size"]="TCC_SIZES"
PARAMS["tcp-size"]="TCP_SIZES"
PARAMS["num-tccs"]="NUM_TCCS"

# --- Configuration ---
# Base directory containing the Python workload scripts
WORKLOAD_DIR="workloads"

# gem5 related paths
GEM5_ROOT="gem5"
GEM5_BUILD_DIR="$GEM5_ROOT/build/VEGA_X86"
GEM5_BIN="$GEM5_BUILD_DIR/gem5.opt"
GEM5_CONFIG="configs/mi200.py"

# gem5 resources paths
RESOURCES_ROOT="gpufs-img"
DISK_IMAGE="$RESOURCES_ROOT/x86-ubuntu-gpu-ml-isca"
KERNEL="$RESOURCES_ROOT/vmlinux-gpu-ml-isca"


# Base directory for gem5 outputs
OUTPUT_BASE_DIR="outputs"
# --- End Configuration ---

# --- Helper Functions ---
# Function to print error messages and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Function to run the simulation
run_simulation() {
    local script_name="$1"
    local param_name="$2"
    local param_value="$3"
    local rep_id="$4"

    local script_name_no_ext="${script_name%.py}"
    local output_dir="${rep_id}_${script_name_no_ext}_${param_name}_${param_value}"
    local full_output_path="${OUTPUT_BASE_DIR}/${output_dir}"

    local script_path="${WORKLOAD_DIR}/${script_name}"

    local gem5_cmd=(
        "$GEM5_BIN"
        -d "$full_output_path" # Output directory for gem5 logs/stats
        "$GEM5_CONFIG"
        --kvm-perf
        --disk-image "$DISK_IMAGE"
        --kernel "$KERNEL"
        --app "$script_path" # The user-selected Python script
        "--${param_name}" "${param_value}"
    )


    echo "----------------------------------------"
    echo "Starting gem5 simulation #${rep_id} on ${script_name} with ${param_name}=${param_value}"
    
    echo "The following command will be executed:"
    # Print the command in a readable format, quoting arguments
    printf "%q " "${gem5_cmd[@]}"
    echo # Add a newline for clarity
    echo "----------------------------------------"

    # Run the command
    "${gem5_cmd[@]}"

    # Capture exit status
    local exit_status=$?

    echo "----------------------------------------"
    if [ $exit_status -eq 0 ]; then
        echo "gem5 simulation finished successfully."
        echo "Output can be found in: $full_output_path"
    else
        echo "gem5 simulation failed with exit status $exit_status."
        echo "Check logs in: $full_output_path"
    fi
    echo "----------------------------------------"
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

# 3. Run all the simulations

for ((rep_id=1; rep_id<=$REPETITIONS; rep_id++)); do
    for script_path in "${scripts[@]}"; do
        for param_name in "${!PARAMS[@]}"; do
            params_array_name="${PARAMS[$param_name]}"

            declare -n params="$params_array_name"

            for param_value in "${params[@]}"; do
                #echo "Script: ${script_path}, Param Name: ${param_name}, Param Value: ${param_value}"
                run_simulation "$(basename "$script_path")" "$param_name" "$param_value" "$rep_id"
            done

            unset $params
        done
    done
done
