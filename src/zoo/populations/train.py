# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from src.core.games import build_game
from src.core.trainers import build_trainer
from src.core.population import build_population
from src.core.evaluators import build_evaluator
from src.core.datasets_onehot import build_one_hot_dataset, split_data_into_population, build_one_hot_dataloader
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

    # Directory for tensorboard logs
    parser.add_argument('--log_dir', type=str, default="", help="Directory to save tensorboard vals")

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

    # Create directories
    if opts.log_dir and not os.path.exists(opts.log_dir):
        os.mkdir(opts.log_dir)
    if opts.model_save_dir and not os.path.exists(opts.model_save_dir):
        os.mkdir(opts.model_save_dir)
    if opts.metrics_save_dir and not os.path.exists(opts.metrics_save_dir):
        os.mkdir(opts.metrics_save_dir)

    # Build population

    population = build_population(population_params=population_params,
                                  agent_repertory=agent_repertory,
                                  game_params=game_params,
                                  device=training_params["device"])

    if opts.print_info_population:
        print(
            f"✅ Successfully built {population_params['population_type']} {population_params['communication_graph']}" +
            f" population with {population_params['n_agents']} agents")
        print(f"Interaction probs are : ", population.pairs_prob)
        print(f"Imitation probs are : ", population.imitation_probs)

    # Build datasets and dataloaders
    full_dataset = build_one_hot_dataset(object_params=game_params["objects"],
                                         n_elements=game_params["dataset"]["n_elements"])

    population_split = split_data_into_population(dataset_size=full_dataset.size(0),
                                                  n_elements=game_params["dataset"]["n_elements"],
                                                  split_proportion=training_params["split_train_val"],
                                                  agent_names=population.agent_names,
                                                  population_dataset_type=population_params['dataset_type'],
                                                  seed=training_params["seed"])

    # Communication task

    train_loader = build_one_hot_dataloader(game_type=game_params["game_type"],
                                            dataset=full_dataset,
                                            agent_names=population.agent_names,
                                            population_split=population_split,
                                            population_probs=population.pairs_prob,
                                            imitation_probs=population.imitation_probs,
                                            training_params=training_params,
                                            task="communication",
                                            mode="train", )

    val_loader = build_one_hot_dataloader(game_type=game_params["game_type"],
                                          dataset=full_dataset,
                                          agent_names=population.agent_names,
                                          population_split=population_split,
                                          population_probs=population.pairs_prob,
                                          imitation_probs=population.imitation_probs,
                                          training_params=training_params,
                                          mode="val")

    # Mutual information task
    mi_loader = build_one_hot_dataloader(game_type=game_params["game_type"],
                                         dataset=full_dataset,
                                         agent_names=population.agent_names,
                                         population_split=population_split,
                                         population_probs=population.pairs_prob,
                                         training_params=training_params,
                                         task="MI",
                                         mode="train")

    # Imitation loaders
    if sum(population_params['is_imitator'])>0.:
        imitation_loader = build_one_hot_dataloader(game_type=game_params["game_type"],
                                                    dataset=full_dataset,
                                                    agent_names=population.agent_names,
                                                    population_split=population_split,
                                                    population_probs=population.pairs_prob,
                                                    imitation_probs=population.imitation_probs,
                                                    training_params=training_params,
                                                    task="imitation",
                                                    mode="train")
    else:
        imitation_loader = None

    if opts.print_info_population: print(f"✅ Successfully built loaders")

    # Build Game
    game = build_game(game_params=game_params,
                      population=population)

    # Build logger
    logger = SummaryWriter(opts.log_dir) if opts.log_dir else None

    # Build Trainer & evaluator

    evaluator = build_evaluator(metrics_to_measure=training_params["metrics_to_measure"],
                                game=game,
                                device=training_params["device"],
                                logger=logger,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                agent_repertory=agent_repertory,
                                game_params=game_params,
                                eval_receiver_id=training_params["eval_receiver_id"],
                                dump_batch=(full_dataset,
                                            population.agent_names[0],
                                            population.agent_names[1]))

    trainer = build_trainer(game=game,
                            evaluator=evaluator,
                            train_loader=train_loader,
                            mi_loader=mi_loader,
                            val_loader=val_loader,
                            imitation_loader=imitation_loader,
                            agent_repertory=agent_repertory,
                            game_params=game_params,
                            device=training_params["device"],
                            compute_metrics=True,
                            metrics_save_dir=opts.metrics_save_dir,
                            logger=logger)

    # Train
    evaluator.step(0)

    trainer.train(n_epochs=training_params["n_epochs"],
                  train_communication_freq=training_params["train_communication_freq"],
                  train_broadcasting_freq=training_params["train_broadcasting_freq"],
                  train_imitation_freq=training_params["train_imitation_freq"],
                  train_mi_freq=training_params["train_mi_freq"],
                  train_kl_freq=training_params["train_kl_freq"],
                  validation_freq=training_params["validation_freq"],
                  evaluator_freq=training_params["evaluator_freq"],
                  custom_steps=training_params["custom_steps"],
                  custom_early_stopping=training_params["custom_early_stopping"])

    if opts.model_save_dir:
        population.save_models(save_dir=opts.model_save_dir)

    if opts.metrics_save_dir:
        evaluator.save_metrics(opts.metrics_save_dir)
        evaluator.save_messages(opts.metrics_save_dir)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
