#!/bin/bash
# Preprocess + train the CNN + test the agent tools (no LLM needed).   sbatch slurm_train.sh
#SBATCH --job-name=wafer-train
#SBATCH --partition=v100gpu           # 32 GB GPU, up to 7 days. (p100gpu = 16 GB GPU, up to 5 days)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --mem=32gb                    # preprocess.py reads a 2 GB pickle
#SBATCH --time=02:00:00
#SBATCH --output=slurm_train_%j.log

cd "$SLURM_SUBMIT_DIR"
date; hostname
# module load python/3.11             # <-- same line as in slurm_setup.sh, if you used a module there
source ~/venv/bin/activate
nvidia-smi | head -15

python preprocess.py                  # makes data/wafer_64.npz (also saves lot names)
python baseline.py
python train.py --epochs 20
python evaluate.py
python gradcam.py
python wafer_tools.py                 # tests the agent tools (CNN side), no LLM
echo "DONE. Model: outputs/best_model.pt"
