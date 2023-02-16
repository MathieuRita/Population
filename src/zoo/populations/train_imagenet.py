# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from src.core.games import build_game
from src.core.trainers import build_trainer
from src.core.population import build_population
from src.core.evaluators import build_evaluator
from src.core.datasets_image import build_image_dataloader
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

    # Communication task
    if "n_training_files" in game_params["dataset"]:
        n_files = game_params["dataset"]["n_training_files"]
    else:
        n_files = None

    train_loader = build_image_dataloader(game_type=game_params["game_type"],
                                          dataset_dir=game_params["dataset"]["path"],
                                          n_files = n_files,
                                          agent_names=population.agent_names,
                                          population_probs=population.pairs_prob,
                                          training_params=training_params,
                                          mode="train")

    val_loader = build_image_dataloader(game_type=game_params["game_type"],
                                        dataset_dir=game_params["dataset"]["path"],
                                        agent_names=population.agent_names,
                                        population_probs=population.pairs_prob,
                                        training_params=training_params,
                                        mode="valid")

    test_loader = build_image_dataloader(game_type=game_params["game_type"],
                                         dataset_dir=game_params["dataset"]["path"],
                                         agent_names=population.agent_names,
                                         population_probs=population.pairs_prob,
                                         training_params=training_params,
                                         mode="test")

    # Mutual information task
    mi_loader = train_loader

    # Imitation loaders
    if sum(population_params['is_imitator']) > 0.:
        imitation_loader = build_image_dataloader(game_type=game_params["game_type"],
                                                  dataset_dir=game_params["dataset"]["path"],
                                                  agent_names=population.agent_names,
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
                                test_loader=test_loader,
                                mi_loader=mi_loader,
                                agent_repertory=agent_repertory,
                                game_params=game_params,
                                eval_receiver_id=training_params["eval_receiver_id"],
                                n_epochs=training_params["n_epochs"])

    trainer = build_trainer(game=game,
                            trainer_type = training_params["trainer_type"],
                            evaluator=evaluator,
                            train_loader=train_loader,
                            mi_loader=mi_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            imitation_loader=imitation_loader,
                            agent_repertory=agent_repertory,
                            game_params=game_params,
                            device=training_params["device"],
                            compute_metrics=True,
                            metrics_save_dir=opts.metrics_save_dir,
                            models_save_dir=opts.model_save_dir,
                            logger=logger)

    # Train
    # evaluator.step(0)

    trainer.train(n_epochs=training_params["n_epochs"],
                  train_communication_freq=training_params["train_communication_freq"],
                  train_broadcasting_freq=training_params["train_broadcasting_freq"],
                  train_imitation_freq=training_params["train_imitation_freq"],
                  train_custom_freq=training_params["train_custom_freq"],
                  train_kl_freq=training_params["train_kl_freq"],
                  validation_freq=training_params["validation_freq"],
                  evaluator_freq=training_params["evaluator_freq"],
                  save_models_freq=training_params["save_models_freq"],
                  custom_steps=training_params["custom_steps"],
                  max_steps=training_params["max_steps"],
                  custom_early_stopping=training_params["custom_early_stopping"])

    if opts.model_save_dir:
        population.save_models(save_dir=opts.model_save_dir)

    if opts.metrics_save_dir:
        evaluator.save_metrics(opts.metrics_save_dir)
        evaluator.save_messages(opts.metrics_save_dir)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
