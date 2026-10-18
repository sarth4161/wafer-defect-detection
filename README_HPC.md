# Running the wafer agent on the IITGN HPC (batch jobs only)

Based on the IITGN HPC user guide. Rules that matter for us:

* **Nothing heavy on the login node. Every run goes through `sbatch`** (guide, FAQ 11).
  Copying files, `unzip`, editing files and reading logs are normal login-node work (the guide only forbids running jobs there).
* Login: `ssh USER@10.0.138.33` (also `hpc.iitgn.ac.in`), with your SSO / network ID.
  **Off campus you need the VPN.**
* GPU partitions: `v100gpu` (1 x V100 32 GB, max 7 days) and `p100gpu` (1 x P100 16 GB, max 5 days).
  We use `v100gpu`. CPU partitions: `small`, `medium`, `large`.
* Always write the time limit (`--time`). When the time is over, the job is killed.
* Home quota is 200 GB. Our files need roughly 15-20 GB (libraries 5 GB, LLM 5 GB, data 3 GB).
* Questions: support.hpc@iitgn.ac.in (send working path, error log and your script).

## What every script does

| Script | Partition | What it does | Run time |
|---|---|---|---|
| `slurm_setup.sh` | small (CPU) | once: Python virtualenv + libraries, Ollama, downloads the LLM | 20-60 min |
| `slurm_train.sh` | v100gpu | preprocess, baseline, train, evaluate, Grad-CAM, test the agent tools | 15-60 min |
| `slurm_agent.sh` | v100gpu | starts the LLM and the web app, prints the ssh command for your laptop | until its time limit |
| `slurm_eval.sh` | v100gpu | 50 test questions -> `outputs/agent_eval.json` | 10-60 min |

## Step by step

**1. Connect and upload (laptop)**
```bash
scp wafer-defect-detection-agent.zip USER@10.0.138.33:~/
scp LSWMD.pkl USER@10.0.138.33:~/              # 2 GB, takes a while
```
Windows without `scp`: use WinSCP or MobaXterm (SFTP, host `10.0.138.33`, port 22).

**2. Unzip (login node)**
```bash
ssh USER@10.0.138.33
unzip wafer-defect-detection-agent.zip          # or: python3 -m zipfile -e wafer-defect-detection-agent.zip .
mv LSWMD.pkl wafer-defect-detection/data/
cd wafer-defect-detection
```
Always run the next commands from this folder.

**3. Find a Python 3.10 or newer (login node, just looking)**
The system Python on Rocky 8 can be old. Gradio needs 3.10+.
```bash
ls /usr/bin/python3*
module avail python
```
Open `slurm_setup.sh` (`nano slurm_setup.sh`). Set `PYTHON=python3.11` (or whatever exists), or uncomment
the `module load python/...` line and give the real module name. Do the same `module load` line in
`slurm_train.sh`, `slurm_agent.sh` and `slurm_eval.sh` if you use a module.

**4. One-time setup (job)**
```bash
sbatch slurm_setup.sh
squeue -u USER
tail -f slurm_setup_*.log                       # Ctrl+C stops watching, not the job
```
Success = the log ends with `SETUP DONE` and shows the model in `ollama list`.
If it says **No internet on this compute node**, go to *Fallback* below.

**5. Train (job)**
```bash
sbatch slurm_train.sh
tail -f slurm_train_*.log
ls data outputs
```
Success = `DONE. Model: outputs/best_model.pt` and you see `data/wafer_64.npz`.
The log also shows `nvidia-smi` (is the GPU visible?) and the result of the tool self-test.
(Already trained? Skip this step but run `python preprocess.py` once via a small job, because the new
`wafer_64.npz` needs lot names. Easiest: just run `slurm_train.sh` again.)

**6. Start the agent app (job)**
```bash
sbatch slurm_agent.sh
tail -f slurm_agent_*.log
```
Wait for the box that says `App is running on node: gpuNN port NNNNN`. Copy the `ssh -L ...` line it prints.
On your **laptop** (campus network or VPN), run that line and leave the window open.
Open the `http://localhost:NNNNN` address from the same box.

Stop it any time: `scancel JOBID` (get the id from `squeue -u USER`).
If the model looks very slow, check that Ollama found the GPU:
`grep -i -E "gpu|cuda" ollama_*.log`

**7. Evaluate (job)**
```bash
sbatch slurm_eval.sh                             # 50 questions
sbatch --export=ALL,N=20 slurm_eval.sh           # 100 questions
tail -f slurm_eval_*.log
scp USER@10.0.138.33:~/wafer-defect-detection/outputs/agent_eval.json .      # run on your laptop
```

## Cheat sheet (from the HPC guide)

| Want | Command |
|---|---|
| submit a job | `sbatch script.sh` |
| my jobs | `squeue -u USER` |
| details of a job | `scontrol show jobid JOBID` |
| finished job info | `sacct -j JOBID` |
| cancel | `scancel JOBID` |
| partitions and free nodes | `sinfo` |
| available software | `module avail` |
| follow a log | `tail -f slurm_agent_JOBID.log` |

## Fallback: compute nodes have no internet

Downloads need internet. If `slurm_setup.sh` stops with "No internet":

1. **Python libraries.** Ask support whether installing with `pip` on the login node is allowed (the guide
   does not mention it; on most clusters it is, because it is only a download). If yes:
   `python3.11 -m venv ~/venv && source ~/venv/bin/activate && pip install -r requirements.txt`
2. **Ollama + the model, from your laptop:**
   * Install Ollama on your laptop and run `ollama pull qwen2.5:7b`.
   * Copy the model folder to the HPC: from `~/.ollama/models` (Windows: `C:\Users\YOU\.ollama\models`) to
     `~/ollama_models` on the HPC, e.g. `scp -r ~/.ollama/models USER@10.0.138.33:~/ollama_models`
   * Download the Linux Ollama file on your laptop (https://github.com/ollama/ollama/blob/main/docs/linux.md
     has the current file name, `ollama-linux-amd64.tgz` or `.tar.zst`), `scp` it to the HPC, then
     `mkdir -p ~/.local && tar -xzf ollama-linux-amd64.tgz -C ~/.local`
     (for `.tar.zst`: `zstd -d -c FILE | tar -x -C ~/.local`).
3. **Or skip Ollama:** run the agent on your laptop with the free Groq API (see `README_AGENT.md`) and copy
   `data/wafer_64.npz` and `outputs/best_model.pt` to the laptop. Training still happens on the HPC.

## Things I could not check (ask support if something fails)

* Do compute nodes have internet?
* Can you open an ssh tunnel from your laptop to a compute-node port (`ssh -L ...`)?
* Is the GPU driver new enough for Ollama's CUDA? If the log shows no GPU, the LLM runs on CPU and is slow;
  try `p100gpu` or ask support for the driver version.
* Which Python modules exist.
* Whether `pip install` on the login node is allowed (only needed for the fallback).

## Common problems

| Problem | Fix |
|---|---|
| Job stays `PD` (pending) | GPU nodes are busy. Wait, or check `squeue -u USER` reason column. There are only 4 V100 and 2 P100 nodes. |
| `Python is too old` in the setup log | set `PYTHON=` or load a module (step 3) |
| `module: command not found` inside a job | the job needs the module init line first (often `source /etc/profile.d/modules.sh`); ask support which one |
| `Connection error` in the agent log | the Ollama server did not start; read `ollama_JOBID.log` |
| Job killed after a while | the `--time` limit ended; raise it (max 7 days on v100gpu) |
| Tunnel connects but the page is empty | wait 1-2 minutes after `App is running` (the model loads), reload |
