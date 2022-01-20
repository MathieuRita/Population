# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from src.core.games import build_game
from src.core.trainers import build_trainer
from src.core.population import build_population
from .datasets_onehot import build_one_hot_dataset, split_data_into_population, build_one_hot_dataloader
from .utils import parse_json

def get_params(params):

    parser = argparse.ArgumentParser()

    # Population parameters
    parser.add_argument('--population_json', type=str, help="Path with list of involved agents")
    parser.add_argument('--print_info_population', type=int, default=0, help="Print info about pop at initialization")

    # Agents parameters
    parser.add_argument('--agents_json', type=str, help="Path to agents list")

    # Game parameters
    parser.add_argument('--game_json', type=str, help="Path to agents list")

    # Training parameters
    parser.add_argument('--training_json', type=str, help="Path to training parameters")

    # p_speaker / p_listener
    parser.add_argument('--p_speaker', type=str, help="Path to training parameters")
    parser.add_argument('--p_listener', type=str, help="Path to training parameters")

    # Directory for tensorboard logs
    parser.add_argument('--log_dir', type=str, default = "", help="Directory to save tensorboard vals")

    # Save Directory
    parser.add_argument('--model_save_dir', type=str, default="", help="Directory to save models")
    parser.add_argument('--metrics_save_dir', type=str, default="", help="Directory to save metrics")

    args = parser.parse_args(params)

    return args


def main(params):

    # Params
    opts = get_params(params)
    game_params = parse_json(opts.game_json)
    population_params = parse_json(opts.population_json)
    agent_repertory = parse_json(opts.agents_json)
    training_params = parse_json(opts.training_json)

    # Custom changes
    agent_repertory["sender_default_reco"]["sender_optim_params"] = opts.p_speaker
    agent_repertory["receiver_default_reco"]["receiver_optim_params"] = opts.p_listener

    # Create directories
    if opts.log_dir and not os.path.exists(opts.log_dir):
        os.mkdir(opts.log_dir)
    if opts.model_save_dir and not os.path.exists(opts.model_save_dir):
        os.mkdir(opts.model_save_dir)
    if opts.metrics_save_dir and not os.path.exists(opts.metrics_save_dir):
        os.mkdir(opts.metrics_save_dir)

    # Build population

    population = build_population(population_params = population_params,
                                  agent_repertory= agent_repertory,
                                  game_params = game_params,
                                  device = training_params["device"])

    if opts.print_info_population:
        print(f"✅ Successfully built {population_params['population_type']} {population_params['communication_graph']}"+
              f" population with {population_params['n_agents']} agents")
        print(f"Interaction probs are : ", population.pairs_prob)

    # Build datasets and dataloaders
    full_dataset = build_one_hot_dataset(object_params = game_params["objects"])

    population_split = split_data_into_population(dataset_size=full_dataset.size(0),
                                                  n_elements = full_dataset.size(0),
                                                  split_proportion=training_params["split_train_val"],
                                                  agent_names = population.agent_names,
                                                  population_dataset_type=population_params['dataset_type'],
                                                  seed = training_params["seed"])

    train_loader = build_one_hot_dataloader(game_type = game_params["game_type"],
                                            dataset = full_dataset,
                                            agent_names = population.agent_names,
                                            population_split = population_split,
                                            population_probs = population.pairs_prob,
                                            training_params = training_params,
                                            mode="train",)

    # Faire ici chaque paire / chaque data
    val_loader = build_one_hot_dataloader(  game_type=game_params["game_type"],
                                            dataset=full_dataset,
                                            agent_names=population.agent_names,
                                            population_split=population_split,
                                            population_probs=population.pairs_prob,
                                            training_params=training_params,
                                            mode="val")

    # Build Game
    game = build_game(game_params = game_params,
                      population = population)

    # Build Trainer
    trainer = build_trainer(game = game,
                            train_loader = train_loader,
                            val_loader = val_loader,
                            device = training_params["device"],
                            compute_metrics = True,
                            log_dir = opts.log_dir)


    # Train
    trainer.init_metrics(dump_batch = (full_dataset,
                                        population.agent_names[0],
                                        population.agent_names[1]))

    trainer.train(n_epochs=training_params["n_epochs"],
                  validation_freq=training_params["validation_freq"],
                  MI_freq =1,
                  dump_freq = 1000,
                  l_analysis_freq=1000,
                  div_analysis_freq=1,
                  dump_batch = (full_dataset,
                                population.agent_names[0],
                                population.agent_names[1]))

    if opts.model_save_dir:
        population.save_models(save_dir=opts.model_save_dir)

    if opts.metrics_save_dir:
        trainer.save_metrics(opts.metrics_save_dir)



if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
