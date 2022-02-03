#!/bin/bash
#SBATCH --partition=gpu_p2
#SBATCH --job-name=test          # nom du job
#SBATCH --ntasks=1                   # nombre total de taches (= nombre de GPU ici)
#SBATCH --nodes=1 # reserving 1 node
#SBATCH --cpus-per-task=9           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=00:15:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=output.txt      # nom du fichier de sortie
#SBATCH -A ovy@gpu

module load pytorch-gpu/py3/1.7.1

python -m src.zoo.populations.train --population_json $1 \
				    --agents_json $2 \
				    --game_json $3 \
				    --training_json $4 \
				    --log_dir $5 \
				    --metrics_save_dir $6 \
				    --print_info_population 1 \
