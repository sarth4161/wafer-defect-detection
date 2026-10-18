#!/bin/bash
# Evaluate the agent (tool-selection accuracy + answer correctness).   sbatch slurm_eval.sh
# More questions:   sbatch --export=ALL,N=20 slurm_eval.sh      (default N=10 -> 50 questions)
#SBATCH --job-name=wafer-eval
#SBATCH --partition=v100gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --mem=24gb
#SBATCH --time=02:00:00
#SBATCH --output=slurm_eval_%j.log

MODEL=qwen2.5:7b
N=${N:-10}

cd "$SLURM_SUBMIT_DIR"
date; hostname
# module load python/3.11             # <-- same line as in slurm_setup.sh, if you used a module there
source ~/venv/bin/activate
nvidia-smi | head -15

export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=$HOME/ollama_models
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 10000))
export OLLAMA_CONTEXT_LENGTH=8192
ollama serve > ollama_$SLURM_JOB_ID.log 2>&1 &
for i in $(seq 1 30); do curl -s http://$OLLAMA_HOST > /dev/null && break; sleep 2; done

export LLM_BASE_URL=http://$OLLAMA_HOST/v1
export LLM_MODEL=$MODEL
export LLM_API_KEY=ollama

python eval_agent.py --n $N
echo "DONE. Results: outputs/agent_eval.json"
