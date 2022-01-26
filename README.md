# Population

PyTorch implementation to run **Emergent Communication in populations of agents**.

## 💻 Run the code 

```
python -m src.zoo.populations.train --population_json PATH_POP_JSON \
                                    --agents_json PATH_AGENTS_JSON \
                                    --game_json PATH_GAME_JSON \
                                    --training_json PATH_TRAINING_JSON \
                                    --log_dir PATH_LOG_DIR \
                                    --save_dir PATH_SAVE_DIR
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


## ✍️ Associated research papers 

- Coming soon


## 👉 References

The code took inspiration from [EGG toolkit](https://github.com/facebookresearch/EGG)
- EGG: a toolkit for research on Emergence of lanGuage in Games, Eugene Kharitonov, Rahma Chaabouni, Diane Bouchacourt, Marco Baroni. EMNLP 2019.