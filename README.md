# ACA Project: Heterogeneous System Simulation using gem5

This repository contains the project files for an Advanced Computer Architecture course, focusing on the simulation of heterogeneous systems (CPU+GPU) using the gem5 simulator. The project explores running AI/ML workloads on a simulated system configured with a VEGA_X86 build target.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
    - [1. Clone This Repository](#1-clone-this-repository)
    - [2. Install Build Dependencies](#2-install-build-dependencies)
    - [3. Clone and Build gem5](#3-clone-and-build-gem5)
    - [4. Obtain Full System Resources (Kernel & Disk Image)](#4-obtain-full-system-resources-kernel--disk-image)
- [Project Structure](#project-structure)
- [Workloads](#workloads)
- [Configuration](#configuration)
- [Running the Simulation](#running-the-simulation)
- [Outputs](#outputs)
- [When Encountering Problems](#when-encountering-problems)

## Overview

This project utilizes the gem5 architectural simulator to model and simulate a full system environment capable of running complex workloads, specifically targeting AI/ML applications on a heterogeneous architecture involving both CPUs and GPUs (modeled via the VEGA_X86 build).

## Prerequisites

Before setting up the project, ensure you have the necessary build dependencies installed.

**Core Requirements:**

*   **git:** For version control.
*   **gcc (>=10, <=13):** C++ compiler for gem5.
*   **Clang (7-16):** Alternative C++ compiler for gem5.
*   **SCons (>=3.0):** The build system used by gem5.
*   **Python (>=3.6):** Required for gem5 configuration scripts and building. Development headers (`python3-dev`) are necessary.

**Optional Dependencies:**

*   **protobuf (>=2.1):** For trace generation and playback features in gem5.
*   **Boost:** Required if using the SystemC implementation within gem5.

**Example Installation (Ubuntu 24.04):**

```bash
sudo apt update
sudo apt install build-essential scons python3-dev git pre-commit zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
    libboost-all-dev libhdf5-serial-dev python3-pydot python3-venv python3-tk mypy \
    m4 libcapstone-dev libpng-dev libelf-dev pkg-config wget cmake doxygen
```

*Note: Users on other Linux distributions should install the equivalent packages using their respective package managers.*

## Setup Instructions

Follow these steps to set up the simulation environment.

### 1. Clone This Repository

Clone this project repository to your local machine:

```bash
git clone https://github.com/arasyi/aca-project.git
cd aca-project
```

*All subsequent commands assume you are in the `aca-project` directory unless otherwise specified.*

### 2. Install Build Dependencies

Ensure all prerequisites listed above are installed on your system using your package manager (see example for Ubuntu 24.04).

### 3. Clone and Build gem5

We need a specific version of gem5 built with the appropriate target architecture.

```bash
# Clone the specific gem5 version
git clone --branch v24.1.0.3 --single-branch https://github.com/gem5/gem5.git

# Navigate into the gem5 directory
cd gem5

# Build gem5 for VEGA_X86 (includes GPU support)
# This can take a significant amount of time and disk space.
scons build/VEGA_X86/gem5.opt -j `nproc`

# Return to the project root directory
cd ..
```

This builds the optimized (`.opt`) version of gem5 for the `VEGA_X86` ISA target, which includes support for simulating AMD GPUs alongside x86 CPUs.

### 4. Obtain Full System Resources (Kernel & Disk Image)

Full system simulation requires a compatible Linux kernel and a disk image containing an operating system and necessary user-space libraries/applications.

You can either download pre-built resources (check the [gem5 Resources](https://resources.gem5.org/) page) or build them yourself. The following commands build the `x86-ubuntu-gpu-ml` resources:

```bash
# Clone the gem5-resources repository
git clone https://github.com/gem5/gem5-resources.git

# Navigate to the specific resource directory
cd gem5-resources/src/x86-ubuntu-gpu-ml

# Run the build script (this may also take considerable time and requires ~30GB space)
# It might require Docker or other dependencies mentioned in gem5-resources docs.
./build.sh

# Return to the project root directory
cd ../../../
```

After building (or downloading and extracting), you should have the following files available:

*   **Disk Image:** `gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml`
*   **Kernel:** `gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-gpu-ml`

*Note: Ensure the paths to these resources are correctly configured within the simulation scripts (`run-gem5-simulation.sh`) if they differ from the expected locations.*

## Project Structure

```
aca-project/
├── gem5/                       # Cloned gem5 simulator source and build directory
├── gem5-resources/             # Cloned gem5 resources source/build directory
├── configs/                    # gem5 simulation configuration scripts (adapted from MI200 examples)
├── workloads/                  # AI/ML workload scripts (Python)
│   ├── matrixmultiplication.py
│   ├── linearregression.py
│   └── mlpclassification.py
├── outputs/                    # Directory where simulation outputs (stats, etc.) are stored
├── analysis.ipynb              # Sample jupyter notebook to analyze the simulation outputs 
├── benchmark-all-workloads.sh  # Sample script to launch batch simulations
├── run-gem5-simulation.sh      # Main script to launch simulations
└── README.md                   # This file
```

## Workloads

The `workloads/` directory contains sample AI/ML workloads written in Python, designed to be executed within the simulated full system environment. Current samples include:

*   `matrixmultiplication.py`: A basic matrix multiplication task.
*   `linearregression.py`: A simple linear regression model training/inference.
*   `mlpclassification.py`: A Multi-Layer Perceptron for classification.

These workloads utilize libraries expected to be present in the disk image (e.g., Python with `torch` and `tensorflow` already installed).

## Configuration

The simulation configuration files are located in the `configs/` directory. These Python scripts use the gem5 library to define the simulated hardware system (CPU cores, caches, memory, GPU, interconnects, etc.). The configurations here are adapted from the MI200 example configurations provided within the main gem5 repository, tailored for the `VEGA_X86` build and the `x86-ubuntu-gpu-ml` full-system resources.

## Running the Simulation

To start a simulation:

1.  Ensure the setup steps (gem5 build, resources) are complete.
2.  Make sure you are in the root directory (`aca-project`).
3.  Execute the main run script:

    ```bash
    ./run-gem5-simulation.sh
    ```

4.  The script will prompt you to:
    *   Choose one of the available workloads from the `workloads/` directory.
    *   Enter any additional simulation parameters if required by the script.

## Outputs

The simulation results, including statistics files (`stats.txt`), configuration details (`config.ini`, `config.json`), standard output/error logs (`simout`, `simerr`), and potentially other traces or dumps, will be generated in the `outputs/` directory. A subdirectory will be created for each specific run, timestamped or named after the workload.

## When Encountering Problems

Remember to:

*   Make the `run-gem5-simulation.sh` script executable (`chmod +x run-gem5-simulation.sh`).
*   Ensure the paths within `run-gem5-simulation.sh` and potentially the `configs/` files correctly point to the built gem5 executable (`gem5/build/VEGA_X86/gem5.opt`) and the kernel/disk image files. You might need to adjust these based on where you cloned `gem5` and `gem5-resources` relative to your project root.
*   Also, gem5 is still buggy. You can post an issue or create a discussion at the project repository: https://github.com/gem5/gem5