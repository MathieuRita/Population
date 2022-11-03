#!/bin/bash
#SBATCH --partition=gpu_p13
#SBATCH --job-name=expe          # nom du job
#SBATCH --ntasks=1                   # nombre total de taches (= nombre de GPU ici)
#SBATCH --nodes=1 # reserving 1 node
#SBATCH --cpus-per-task=6           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=output_slurm/test.txt      # nom du fichier de sortie
#SBATCH -A ovy@v100

#module load pytorch-gpu/py3/1.7.1
module load pytorch-gpu/py3/1.11.0

python -m src.zoo.populations.train --population_json $1 \
				    --agents_json $2 \
				    --game_json $3 \
				    --training_json $4 \
				    --log_dir $5 \
				    --metrics_save_dir $6 \
				    --model_save_dir $7 \
				    --print_info_population 1 \
