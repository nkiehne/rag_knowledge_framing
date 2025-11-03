module load miniforge3
module load apptainer
#module load gcc/13.2.0
#module load cuda/12.6.2

export TORCH_CUDA_ARCH_LIST="9.0"

source activate dasfaa26_env

python -m jupyter lab --ContentsManager.allow_hidden=True --YDocExtension.disable_rtc=True --port 6868 --notebook-dir ../
