#!/bin/bash
##
## gpuexample.sbatch submit a job using a GPU
##
## Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
## All other lines are read by the shell.
##
#SBATCH --account=priority-reu       # priority account to use
#SBATCH --job-name=cse493-hw           # job name
#SBATCH --partition=gpupriority         # queue partition to run the job in
#SBATCH --nodes=1                       # number of nodes to allocate
#SBATCH --ntasks-per-node=1             # number of descrete tasks - keep at one except for MPI
#SBATCH --cpus-per-task=8               # number of cores to allocate - do not allocate more than 16 cores per GPU
#SBATCH --gpus-per-task=1               # number of GPUs to allocate - all GPUs are currently A40 model
#SBATCH --mem=60000                     # 2000 MB of Memory allocated - do not allocate more than 128000 MB mem per GPU
#SBATCH --time=0-16:00:00               # Maximum job run time
#SBATCH --output=cse493-hw-%j.out      # standard output file (%j = jobid)
#SBATCH --error=cse493-hw-%j.err       # standard error file
#SBATCH --mail-user=nashrickert@gmail.com     # email address to recieve job updates
#SBATCH --mail-type=ALL                 # conditions to recieve emial notifications for job
## Run 'man sbatch' for more information on the options above.
module load Anaconda3/2020.07
## module unload GCCcore
## module load CUDA/12.3.0
module load CUDA/11.1.1-GCC-10.2.0
source venv/bin/activate
python3 train.py
source deactivate
