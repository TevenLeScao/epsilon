#!/bin/bash
#SBATCH --job-name=make_partial_bigscience_dataset # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=2:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --account=ajs@cpu
#SBATCH --partition=cpu_p1

cd /gpfswork/rech/six/commun/bigscience-training || exit
find . -name "*.jsonl" -exec bash -c '$WORK/code/epsilon/scripts/_stochastic_fraction_of_every_dataset.sh $0' {} \;
