#!/bin/bash
# ONE-TIME setup, run as a batch job (IITGN HPC does not allow work on the login node).
#   sbatch slurm_setup.sh
# It makes the python virtualenv, installs the libraries, installs Ollama and downloads the LLM.
# It needs INTERNET on the compute node. If the log says "No internet", see README_HPC.md (Fallback).
#SBATCH --job-name=wafer-setup
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=03:00:00
#SBATCH --output=slurm_setup_%j.log

PYTHON=python3            # <-- CHANGE if needed: python3.11 ... (need Python 3.10 or newer, Gradio needs it)
MODEL=qwen2.5:7b          # smaller option: qwen2.5:3b

cd "$SLURM_SUBMIT_DIR"
date; hostname
# module load python/3.11 # <-- uncomment + edit if your Python comes from a module (`module avail python`)

echo "== python =="
$PYTHON --version
$PYTHON -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || {
    echo "Python is too old (need 3.10+). Run 'module avail python' or ask support.hpc@iitgn.ac.in"; exit 1; }

echo "== internet check =="
curl -sf --max-time 20 -o /dev/null https://pypi.org || {
    echo "No internet on this compute node. See README_HPC.md (Fallback)."; exit 1; }

echo "== virtualenv + libraries =="
$PYTHON -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__)"

echo "== Ollama =="
mkdir -p ~/.local && cd ~/.local
if curl -fL https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz; then
    tar -xzf ollama.tgz
elif curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst; then
    zstd -d -c ollama.tar.zst | tar -x -C ~/.local            # needs the zstd program
else
    echo "Could not download Ollama. See README_HPC.md (Fallback)."; exit 1
fi
rm -f ollama.tgz ollama.tar.zst
cd "$SLURM_SUBMIT_DIR"

echo "== LLM model: $MODEL =="
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/ollama:$LD_LIBRARY_PATH
export OLLAMA_MODELS=$HOME/ollama_models
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 10000))   # own port, so we never clash with other users
ollama serve > ollama_setup_$SLURM_JOB_ID.log 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 30); do curl -s http://$OLLAMA_HOST > /dev/null && break; sleep 2; done
ollama pull $MODEL
ollama list
kill $OLLAMA_PID
echo "SETUP DONE"
