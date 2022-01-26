import json
import os
import torch as th


# json parsers

def parse_json(json_file):
    with open(json_file) as f:
        data = json.load(f)

    return data


# json generator

def generate_json(experiments_dir: str,
                  logs_dir : str,
                  experiment_name: str = "",
                  default_population_json: str = "",
                  default_agents_json: str = "",
                  default_game_json: str = "",
                  default_training_json: str = "",
                  new_agents : list = None,
                  new_population : dict = None,
                  game_channel : dict = None,
                  game_objects : dict = None,
                  game_dataset : dict = None,
                  game_name : str = None,
                  game_type : str = None,
                  training_n_epochs : int = None,
                  training_split_train_val : float = None,
                  training_batch_size : int = None,
                  training_batches_per_epoch : int = None,
                  training_seed : int = None,
                  ) -> None:
    """
    Generate json files to further run experiments

    new_agents: {"new_name":,"default_name":,"param_1_to_be_changed":,...,"param_k_to_be_changed":}

    :return:
    """

    assert os.path.exists(experiments_dir), "experiments_dir does not exists"
    assert len(experiment_name), "Set an experiment name !"

    if not os.path.exists(f"{experiments_dir}/{experiment_name}"):
        os.mkdir(f"{experiments_dir}/{experiment_name}")
        os.mkdir(f"{experiments_dir}/{experiment_name}/metrics")
        os.mkdir(f"{experiments_dir}/{experiment_name}/json")
        os.mkdir(f"{experiments_dir}/{experiment_name}/models")
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    os.mkdir(f"{logs_dir}/{experiment_name}")

    # Agents
    if len(default_agents_json):
        agents_json = parse_json(default_agents_json)
    else:
        agents_json = dict()

    for i in range(len(new_agents)):
        new_name = new_agents[i]["new_name"]
        default_name = new_agents[i]["default_name"]
        if "param_changes" in new_agents[i]:
            param_changes = new_agents[i]["param_changes"]
        else:
            param_changes = {}

        new_agent = agents_json[default_name]
        for param_to_be_changed, new_param in param_changes.items():
            new_agent[param_to_be_changed] = new_param

        agents_json[new_name] = new_agent

    # Population
    if len(default_population_json):
        population_json = parse_json(default_population_json)
    else:
        population_json = dict()

    if new_population is not None:
        for param_to_be_changed, new_param in new_population["param_changes"].items():
            population_json[param_to_be_changed] = new_param

    # Game
    if len(default_game_json):
        game_json = parse_json(default_game_json)
    else:
        game_json = dict()

    if game_channel is not None:
        if "max_len" in game_channel:
            game_json["channel"]["max_len"] = game_channel["max_len"]
        if "voc_size" in game_channel:
            game_json["channel"]["voc_size"] = game_channel["voc_size"]

    if game_objects is not None:

        if "object_type" in game_objects:
            game_json["objects"]["object_type"] = game_objects["object_type"]
        if "object_type" in game_objects:
            game_json["objects"]["n_attributes"] = game_objects["n_attributes"]
        if "object_type" in game_objects:
            game_json["objects"]["n_values"] = game_objects["n_values"]

    if game_dataset is not None:

        if "object_type" in game_dataset:
            game_json["dataset"]["n_elements"] = game_dataset["n_elements"]
        if "object_type" in game_dataset:
            game_json["dataset"]["split_proportion"] = game_dataset["split_proportion"]

    if game_name is not None:
        game_json["game_name"] = game_name

    if game_type is not None:
        game_json["game_type"] = game_type

    # Training
    if len(default_training_json):
        training_json = parse_json(default_training_json)
    else:
        training_json = dict()

    if training_n_epochs is not None:
        training_json["n_epochs"] = training_n_epochs
    if training_split_train_val is not None:
        training_json["split_train_val"] = training_split_train_val
    if training_batch_size is not None:
        training_json["batch_size"] = training_batch_size
    if training_batches_per_epoch is not None:
        training_json["batches_per_epoch"] = training_batches_per_epoch
    if training_seed is not None:
        training_json["seed"] = training_seed

    # Save json files
    with open(f"{experiments_dir}/{experiment_name}/json/population.json", 'w') as outfile:
        json.dump(population_json, outfile)

    with open(f"{experiments_dir}/{experiment_name}/json/agents.json", 'w') as outfile:
        json.dump(agents_json, outfile)

    with open(f"{experiments_dir}/{experiment_name}/json/game.json", 'w') as outfile:
        json.dump(game_json, outfile)

    with open(f"{experiments_dir}/{experiment_name}/json/training.json", 'w') as outfile:
        json.dump(training_json, outfile)




# For experiments

def find_lengths(messages: th.Tensor) -> th.Tensor:
    """
    :param messages: A tensor of term ids, encoded as Long values, of size (batch size, max sequence length).
    :returns A tensor with lengths of the sequences, including the end-of-sequence symbol <eos> (in EGG, it is 0).
    If no <eos> is found, the full length is returned (i.e. messages.size(1)).
    >>> messages = th.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
    >>> lengths = find_lengths(messages)
    >>> lengths
    tensor([3, 6])
    """
    max_k = messages.size(1)
    zero_mask = messages == 0
    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths
