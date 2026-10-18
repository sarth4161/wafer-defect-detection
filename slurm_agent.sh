#!/bin/bash
# Start the LLM (Ollama) + the agent web app on a GPU node.   sbatch slurm_agent.sh
# Then read the log: it prints the exact ssh command to run on YOUR laptop.
#SBATCH --job-name=wafer-agent
#SBATCH --partition=v100gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --mem=24gb
#SBATCH --time=04:00:00               # the job stops after this time (max 7 days on v100gpu)
#SBATCH --output=slurm_agent_%j.log

MODEL=qwen2.5:7b                      # must be the model you downloaded in slurm_setup.sh

cd "$SLURM_SUBMIT_DIR"
date; hostname
# module load python/3.11             # <-- same line as in slurm_setup.sh, if you used a module there
source ~/venv/bin/activate
nvidia-smi | head -15

# --- 1. LLM server (Ollama), on its own port
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=$HOME/ollama_models
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 10000))
export OLLAMA_CONTEXT_LENGTH=8192     # longer memory for tool results (newer Ollama versions)
ollama serve > ollama_$SLURM_JOB_ID.log 2>&1 &
for i in $(seq 1 30); do curl -s http://$OLLAMA_HOST > /dev/null && break; sleep 2; done
ollama list

# --- 2. tell our code where the LLM is
export LLM_BASE_URL=http://$OLLAMA_HOST/v1
export LLM_MODEL=$MODEL
export LLM_API_KEY=ollama

# --- 3. web app on its own port
PORT=$((30000 + SLURM_JOB_ID % 10000))
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=$PORT
echo "=================================================================="
echo "App is running on node: $(hostname)   port $PORT"
echo "On YOUR LAPTOP (on campus network or VPN) run:"
echo "   ssh -L $PORT:$(hostname):$PORT $USER@10.0.138.33"
echo "Then open:   http://localhost:$PORT"
echo "=================================================================="
python agent_app.py
