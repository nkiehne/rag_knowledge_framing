# General Information
***
This repository can be used in two ways:
* At the (compute) node level: For this, simply execute the respective notebooks and adapt the parameters to your liking.
    * E.g. `run_requests.ipynb` can be used to run single LLMs on a batch of requests via VLLM serving (see below).
* At the cluster level: Dispatch multiple runs on a slurm-managed cluster or standard multiprocessing for single nodes. This is implemented via the `papermill` library which allows to parameterize an existing notebook.
  
Below we briefly introduce general concepts in this repository.

## Datasets as Request files
We compile all used datasets into JSONL files in the `data/` folder. 
* For reproducibility reasons, these are the _exact_ requests that are send to the LLMs (except for the model identifier, which is varied).
* E.g., `data/math7.jsonl` contains all queries over all prompt types.

<details>
<summary>Example request (1 line)</summary>

```json
{
    "custom_id": "9288106_P_X",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "X",
        "messages": [{
            "role": "user",
            "content": "Annotate the given abstract..."
        }]
    }
}
```

</details>

## LLM inference setup
We use vllm, apptainer and slurm. To adapt our setup to your own infrastructure, make sure to look into:
* `run_requests.ipynb`: For vLLM model serving parameters and apptainer image setting.
* `run_slurm_python_task.sh`: For job-level setup, e.g. module loading (e.g. cuda, miniforge, etc.) and environment setup.
* `deploy_jobs.ipynb`: For dispatching your jobs.
* `deploy.py` lines 528-538: For sbatch configuration.

# Prepare
***
1. Clone the repository

1. Download NCBI Corpus and extract into folder `ncbi`, with this structure:
    * `ncbi/NCBI_corpus_training.txt`
    * `ncbi/NCBI_corpus_development.txt` 
    * `ncbi/NCBI_corpus_testing.txt`

2) Create a new environment in the root level of this repo.
    * `run_slurm_python_task.sh` expects an env named `vldb26`. Adapt as needed if you want to use slurm.
    * Packages are listed in `requirements.txt`.

3) If you plan on running LLMs:
    * Either download the model files in advance, e.g. with `huggingface_hub.download_snapshot`, or remove the env flag `HF_HUB_OFFLINE=1` in `run_requests.ipynb` if your compute node has internet access.
        * If you start many jobs at the same time: We frequently found the HuggingFace Hub to block http requests. Our code thus operates offline per default and expects the models to be already present in the `HF_CACHE` dir.
    * Prepare the apptainer vllm image:
        1. Pull and convert the vLLM docker image to a .sif image via `apptainer pull docker://vllm/vllm-openai`
        2. Change param "vllm_image" in `run_requests.ipynb` to point to the location of your .sif image.
    
# Datasets
***
Our finalized datasets are already included in the `data/` folder. Specifically:
* NCBI Annotation task: `data/ncbi.jsonl`
* Multiplication task: `data/math7.jsonl`
* Drug-Drug Interaction task: Split across the drugs separately:
    * `data/simvastatin.jsonl`
    * `data/upadacitinib.jsonl`
    * `data/digitoxin.jsonl`

## Dataset construction
The steps to construct these files are detailed in:
* NCBI Disease Annotation: `ncbi_prepare_annotation_task.ipynb`
* Drug-Drug Interaction: TODO
* Multiplication: TODO

## DDI Setup ...?

# Reproducing experiments and Evaluation
* `deploy_jobs.ipynb` shows how to start the jobs to reproduce the results from our paper.
    * Make sure to adapt tensor, data and expert-parallel settings for each model, such that your nodes are well utilized.
* All code to produce tables, significance tests and figures is contained in `eval.ipynb`.