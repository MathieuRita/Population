# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch as th
from src.core.agents import get_agent
from src.core.games import build_game
from src.core.trainers import build_trainer
from src.core.datasets_onehot import build_one_hot_dataset, build_target_messages, build_one_hot_dataloader
from .utils import parse_json

def get_params(params):

    parser = argparse.ArgumentParser()

    # Agents parameters
    parser.add_argument('--agents_json', type=str, help="Path to agents list")

    # Agent name
    parser.add_argument('--agent_name', type=str, help="Agent to pretrain")

    # Game parameters
    parser.add_argument('--game_json', type=str, help="Path to agents list")

    # Pretrained language
    parser.add_argument('--pretrained_language', type=str, default=None, help="Path to training parameters")

    # Training parameters
    parser.add_argument('--training_json', type=str, help="Path to training parameters")

    # Directory for saving models
    parser.add_argument('--save_dir', type=str, default="", help="Directory to save pretrained models")

    # Directory for tensorboard logs
    parser.add_argument('--log_dir', type=str, default="", help="Directory to save tensorboard vals")

    args = parser.parse_args(params)

    return args

def main(params):

    # Params
    opts = get_params(params)
    game_params = parse_json(opts.game_json)
    agent_repertory = parse_json(opts.agents_json)
    training_params = parse_json(opts.training_json)

    # Build agent
    agent= get_agent(agent_name = opts.agent_name,
                       agent_repertory = agent_repertory,
                       game_params = game_params,
                       device = training_params["device"])


    # Build datasets and dataloaders
    full_dataset = build_one_hot_dataset(object_params=game_params["objects"],n_elements=game_params["dataset"]["n_elements"])
    full_target_messages = build_target_messages(n_elements=game_params["dataset"]["n_elements"],
                                                 pretrained_language=opts.pretrained_language,
                                                 channel_params=game_params["channel"])

    train_loader = build_one_hot_dataloader(game_type=game_params["game_type"],
                                            dataset=full_dataset,
                                            target_messages = full_target_messages,
                                            training_params=training_params,
                                            mode="train")

    # Build Game
    game = build_game(game_params=game_params,
                      agent=agent)

    # Build Trainer
    trainer = build_trainer(game=game,
                            trainer_type=training_params["trainer_type"],
                            train_loader=train_loader,
                            val_loader = None,
                            device=training_params["device"],
                            compute_metrics=True,
                            game_params=None,
                            agent_repertory=None,
                            evaluator=None)

    # Train
    trainer.train(n_epochs=training_params["n_epochs"])

    # Save pretrained models
    if opts.save_dir: th.save(agent.sender.state_dict(), f"{opts.save_dir}/sender.pt")
    if opts.save_dir: th.save(agent.object_encoder.state_dict(), f"{opts.save_dir}/object_encoder.pt")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])