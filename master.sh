for dir in /gpfswork/rech/nlt/uqm82td/IT_project/Population_2/experiments/6-6-mi/*   
do
    expe_name=${dir##*/}
    agents_json="$dir/json/agents.json"
    population_json="$dir/json/population.json"
    game_json="$dir/json/game.json"
    training_json="$dir/json/training.json"
    log_dir="/gpfswork/rech/nlt/uqm82td/IT_project/Population_2/logs/6-6-mi/$expe_name/new_4"
    metrics_sav_dir="/gpfswork/rech/nlt/uqm82td/IT_project/Population_2/experiments/6-6-mi/$expe_name/metrics"
    
    echo "$expe_name"
    
    sbatch run_experiment.sh $population_json $agents_json $game_json $training_json $log_dir $metrics_sav_dir

done
