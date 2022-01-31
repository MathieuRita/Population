# Population

PyTorch implementation to run **Emergent Communication in populations of agents**.

## 💻 Run the code 

Generate json files saving dirs (following directory structure) for custom experiments:

```
python -m src.zoo.populations.custom_experiments --default_population_json PATH_DEFAULT_POP_JSON \
                                                --default_agents_json PATH_DEFAULT_AGENTS_JSON \
                                                --default_game_json PATH_DEFAULT_GAME_JSON \
                                                --default_training_json PATH_DEFAULT_TRAINING_JSON \
                                                --experiments_dir PATH_EXPERIMENT_DIR \
                                                --logs_dir PATH_LOG_DIR \
                                                --base_experiment_name BASE_EXPERIMENT_NAME
```

Train a population:

```
python -m src.zoo.populations.train --population_json PATH_POP_JSON \
                                    --agents_json PATH_AGENTS_JSON \
                                    --game_json PATH_GAME_JSON \
                                    --training_json PATH_TRAINING_JSON \
                                    --log_dir PATH_LOG_DIR \
                                    --model_save_dir PATH_SAVE_DIR \
                                    --metrics_save_dir PATH_SAVE_DIR \
```

## 🗂️ Directory structure

```
| experiments

-- | expe_name
--–- | json
------ | population.json
------ | agents.json
------ | game.json
------ | training.json
–--- | metrics
–--- | models

| logs
-- | expe_name
```

## 📆 TO DO

- [ ] Change `dump_batch` en `DataLoader` pour `evaluators`
- [ ] Complete generate_json
- [ ] Change th.tile by .repeat


## ✍️ Associated research papers 

- Rita M., Strub F., Grill J-B., Pietquin O., Dupoux E. (2022). On the role of population heterogeneity in emergent communication. *In Proceedings of International Conference on Learning Representations (ICLR)*.


## 👉 References

The code took inspiration from [EGG toolkit](https://github.com/facebookresearch/EGG)
- EGG: a toolkit for research on Emergence of lanGuage in Games, Eugene Kharitonov, Rahma Chaabouni, Diane Bouchacourt, Marco Baroni. EMNLP 2019.