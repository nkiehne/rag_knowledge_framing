import os

IS_DEPLOYED_TASK_FLAG = "IS_DEPLOYED_TASK"

def get_available_gpus(max_utilization=0):
    # find out whether we are running one of our tasks:
    is_ours = os.environ.get(IS_DEPLOYED_TASK_FLAG, False)

    # first try: check env vars for cuda:
    gpu_str = os.environ.get("CUDA_VISIBLE_DEVICES","")
    gpus = [int(y) for y in gpu_str.split(",") if len(y) > 0]

    # if its not set and we didnt spawn the process, give it all gpus
    if not is_ours:
        if len(gpus) == 0:   
            # return a list of all gpus accessible by nvidia-smi
            try:
                import subprocess
                result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
                gpus = [int(l[0]) for i in result.stdout.strip().split("\n") if int((l := i.split(","))[1].strip()) <= max_utilization]
            except:
                print("Failed to run nvidia-smi, returning no gpus")
            return gpus
    # else, return what was set in CUDA_VISIBLE_DEVICES, possibly nothing
    return gpus
