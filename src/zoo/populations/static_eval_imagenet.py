# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from src.core.population import build_population
from src.core.games import build_game
from src.core.static_evaluator import get_static_evaluator
from .utils import parse_json


def get_params(params):
    parser = argparse.ArgumentParser()

    # Population parameters
    parser.add_argument('--population_json', type=str, help="Path with list of involved agents")

    # Agents parameters
    parser.add_argument('--agents_json', type=str, help="Path to agents list")

    # Game parameters
    parser.add_argument('--game_json', type=str, help="Path to agents list")

    # Eval parameters
    parser.add_argument('--eval_json', type=str, help="Path to eval info")

    # Dataset dir
    parser.add_argument('--save_dir', type=str, help="Path to dataset dir")

    args = parser.parse_args(params)

    return args


def main(params):
    # Params
    opts = get_params(params)
    population_params = parse_json(opts.population_json)
    game_params = parse_json(opts.game_json)
    eval_params = parse_json(opts.eval_json)
    agent_repertory = parse_json(opts.agents_json)
    if "distances" not in eval_params: eval_params["distances"] = {"input":"cosine_similarity",
                                                                 "message":"edit_distance",
                                                                 "projection":"cosine_similarity"}

    # Build population
    population = build_population(population_params=population_params,
                                  agent_repertory=agent_repertory,
                                  game_params=game_params,
                                  device=eval_params["device"])

    # Build Game
    game = build_game(game_params=game_params,
                      population=population)

    # Build Evaluator

    static_evaluator = get_static_evaluator(game=game,
                                            population=population,
                                            agents_to_evaluate = eval_params["agents_to_evaluate"],
                                            metrics_to_measure = eval_params["metrics_to_evaluate"],
                                            eval_receiver_id = eval_params["eval_receiver_id"],
                                            agent_repertory=agent_repertory,
                                            game_params=game_params,
                                            dataset_dir=game_params["dataset"]["path"],
                                            image_dataset = eval_params["image_dataset"],
                                            couple_to_evaluate=eval_params["couple_to_evaluate"],
                                            save_dir = opts.save_dir,
                                            distance_input=eval_params["distances"]["input"],
                                            distance_message=eval_params["distances"]["message"],
                                            distance_projection=eval_params["distances"]["projection"],
                                            device = eval_params["device"])

    static_evaluator.step(print_results = True, save_results = True)




if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
