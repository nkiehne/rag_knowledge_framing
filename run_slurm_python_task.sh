#!/bin/bash
module load miniforge3
module load apptainer

source activate vldb26_env

cleanup() {
    if [ -f "$1" ]; then
        rm "$1"
    fi
}
trap 'cleanup "$1"; exit' INT TERM HUP EXIT

python "$1"