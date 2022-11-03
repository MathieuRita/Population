date="03_11_2022_n_speakers"

for dir in /gpfswork/rech/nlt/uqm82td/journal_project/Population/experiments/$date/*   
do
    expe_name=${dir##*/}
    agents_json="$dir/json/agents.json"
    population_json="$dir/json/population.json"
    game_json="$dir/json/game.json"
    training_json="$dir/json/training.json"
    log_dir="/gpfswork/rech/nlt/uqm82td/journal_project/Population/logs/$date/$expe_name"
    metrics_sav_dir="/gpfswork/rech/nlt/uqm82td/journal_project/Population/experiments/$date/$expe_name/metrics"
    model_save_dir="/gpfswork/rech/nlt/uqm82td/journal_project/Population/experiments/$date/$expe_name/models"
    dataset_save_dir="/gpfswork/rech/nlt/uqm82td/journal_project/Population/experiments/$date/$expe_name/dataset"
    
    echo "$expe_name"
    
    sbatch run_experiment.sh $population_json $agents_json $game_json $training_json $log_dir $metrics_sav_dir $model_save_dir $dataset_save_dir

done
