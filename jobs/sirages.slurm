#!/bin/bash -l

#SBATCH --job-name="SIR-Ages"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=10GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output="jobs_oe/xxx-%j.o"
#SBATCH --error="jobs_oe/xxx-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python ModelSIRAges.py --log_path logs/log_SIR-Ages.txt
