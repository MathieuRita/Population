import json
import os
import torch as th
from collections import defaultdict


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

def fill_missing_training_params(training_params):

    """
    Assert that the minimum training parameters are given in the json with training params.
    Set all the non-unsed params to values such that they have no effect on the training
    """

    # Minimal training parameters
    assert "n_epochs" in training_params, \
        "The number of epochs n_epochs in not indicated in the training params"
    assert "device" in training_params, \
        "The device in not indicated in the training params"
    #assert "split_train_val" in training_params, \
    #    "The train/val ratio split_train_val in not indicated in the training params"
    assert "batch_size" in training_params, \
        "The batch_size batch_size in not indicated in the training params"
    assert "train_batches_per_epoch" in training_params, \
        "The number of batches per epoch train_batches_per_epoch in not indicated in the training params"
    assert "n_epochs" in training_params, \
        "The number of epochs n_epochs in not indicated in the training params"

    # Main training params
    if "seed" not in training_params:
        training_params["seed"] = 1

    if "broadcasting" not in training_params:
        training_params["broadcasting"] = 0

    if "MI_batch_size" not in training_params:
        training_params["MI_batch_size"] = training_params["batch_size"]
    if "val_batches_per_epoch" not in training_params:
        training_params["val_batches_per_epoch"] = 1
    if "test_batches_per_epoch" not in training_params:
        training_params["test_batches_per_epoch"] = 1
    if "MI_batches_per_epoch" not in training_params:
        training_params["MI_batches_per_epoch"] = 1

    # Freq of training types
    if "train_communication_freq" not in training_params:
        if training_params["broadcasting"]==0:
            training_params["train_communication_freq"] = 1
        else:
            training_params["train_communication_freq"] = training_params["n_epochs"] + 1
    if "validation_freq" not in training_params: # Perf on the validation set
        training_params["validation_freq"] = 1
    if "evaluator_freq" not in training_params:
        training_params["evaluator_freq"] = training_params["n_epochs"] + 1
    if "reset_agents_freq" not in training_params:
        training_params["reset_agents_freq"] = training_params["n_epochs"] + 1
    if "train_broadcasting_freq" not in training_params:
        training_params["train_broadcasting_freq"] = training_params["n_epochs"] + 1
    if "train_imitation_freq" not in training_params:
        training_params["train_imitation_freq"] = training_params["n_epochs"] + 1
    if "train_mi_freq" not in training_params:
        training_params["train_mi_freq"] = training_params["n_epochs"] + 1
    if "train_custom_freq" not in training_params:
        training_params["train_custom_freq"] = training_params["n_epochs"] + 1
    if "train_communication_and_mi_freq" not in training_params:
        training_params["train_communication_and_mi_freq"] = training_params["n_epochs"] + 1
    if "train_kl_freq" not in training_params:
        training_params["train_kl_freq"] = training_params["n_epochs"] + 1
    if "train_mi_with_lm_freq" not in training_params:
        training_params["train_mi_with_lm_freq"] = training_params["n_epochs"] + 1
    if "train_comm_and_check_gradient" not in training_params:
        training_params["train_comm_and_check_gradient"] = training_params["n_epochs"] + 1

    # Save models frequency
    if "save_models_freq" not in training_params:
        training_params["save_models_freq"] = training_params["n_epochs"] + 1

    # Evaluation metrics (for evaluator)
    if "metrics_to_measure" not in training_params:
        training_params["metrics_to_measure"] = defaultdict(int)

    if "writing" not in training_params["metrics_to_measure"]: # frequency at which we write a metric
        training_params["metrics_to_measure"]["writing"] = training_params["n_epochs"] + 1

    if "topographic_similarity" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["topographic_similarity"] = training_params["n_epochs"] + 1
    if "language_similarity" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["language_similarity"] = training_params["n_epochs"] + 1
    if "reward_decomposition" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["reward_decomposition"] = training_params["n_epochs"] + 1
    if "external_receiver_evaluation" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["external_receiver_evaluation"] = training_params["n_epochs"] + 1
    if "similarity_to_init_languages" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["similarity_to_init_languages"] = training_params["n_epochs"] + 1
    if "reward_decomposition" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["reward_decomposition"] = training_params["n_epochs"] + 1
    if "similarity_to_init_languages" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["similarity_to_init_languages"] = training_params["n_epochs"] + 1
    if "MI" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["MI"] = training_params["n_epochs"] + 1
    if "overfitting" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["overfitting"] = training_params["n_epochs"] + 1
    if "similarity_to_init_languages" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["similarity_to_init_languages"] = training_params["n_epochs"] + 1
    if "divergence_to_untrained_speakers" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["divergence_to_untrained_speakers"] = training_params["n_epochs"] + 1
    if "accuracy_with_untrained_speakers" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["accuracy_with_untrained_speakers"] = training_params["n_epochs"] + 1
    if "accuracy_with_untrained_listeners" not in training_params["metrics_to_measure"]:
        training_params["metrics_to_measure"]["accuracy_with_untrained_listeners"] = training_params["n_epochs"] + 1

    # Eval receiver id for MI measures
    if "eval_receiver_id" not in training_params:
        training_params["eval_receiver_id"] = ""

    # Custom training
    if "custom_steps" not in training_params:
        training_params["custom_steps"] = 0
    if "custom_early_stopping" not in training_params:
        training_params["custom_early_stopping"] = 0
    if "max_steps" not in training_params:
        training_params["max_steps"] = 0

    return training_params

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
