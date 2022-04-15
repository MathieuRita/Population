import torch as th
import numpy as np
from scipy.stats import spearmanr
from .utils import move_to, find_lengths
from .language_metrics import compute_language_similarity
from collections import defaultdict


class StaticEvaluator:

    def __init__(self,
                 game,
                 population,
                 metrics_to_measure,
                 agents_to_evaluate,
                 dataset_dir,
                 save_dir,
                 device: str = "cpu") -> None:

        self.game = game
        self.population = population
        self.device = th.device(device)
        self.agents_to_evaluate = agents_to_evaluate
        self.metrics_to_measure = metrics_to_measure
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir

    def step(self,
             print_results: bool = False,
             save_results: bool = False) -> None:

        if "topographic_similarity" in self.metrics_to_measure:
            topographic_similarity = self.estimate_topographic_similarity()
        else:
            topographic_similarity = None

        if "MI" in self.metrics_to_measure:
            raise NotImplementedError
        else:
            mutual_information = None

        if "success" in self.metrics_to_measure:
            raise NotImplementedError
        else:
            success = None

        if "max_generalization" in self.metrics_to_measure:
            raise NotImplementedError
        else:
            max_generalization = None

        if save_results:
            self.save_results(save_dir=self.save_dir,
                              topographic_similarity=topographic_similarity,
                              mutual_information=mutual_information,
                              success=success,
                              max_generalization=max_generalization)

        if print_results:
            self.print_results(topographic_similarity=topographic_similarity,
                               mutual_information=mutual_information,
                               success=success,
                               max_generalization=max_generalization)

    def estimate_topographic_similarity(self,
                                        distance_input: str = "common_attributes",
                                        distance_message: str = "edit_distance",
                                        N_sampling: int = 100,
                                        batch_size: int = 1000) -> dict:

        topographic_similarity_results = dict()
        full_dataset = th.load(f"{self.dataset_dir}/full_dataset.pt")

        self.game.train()

        with th.no_grad():

            for agent_name in self.agents_to_evaluate:
                agent = self.population.agents[agent_name]
                if agent.sender is not None:
                    topographic_similarity_results[agent_name] = defaultdict(list)
                    train_split = th.load(f"{self.dataset_dir}/{agent_name}_train_split.pt")
                    val_split = th.load(f"{self.dataset_dir}/{agent_name}_val_split.pt")
                    test_split = th.load(f"{self.dataset_dir}/{agent_name}_test_split.pt")

                    splits = {"train": train_split,
                              "val": val_split,
                              "test": test_split}

                    for split_type in splits:

                        dataset = full_dataset[splits[split_type]]

                        # Train
                        for _ in range(N_sampling):
                            num_inputs = min(batch_size, dataset.size(0))
                            inputs_1 = dataset[th.multinomial(th.ones(dataset.size(0)),
                                                                 num_inputs,
                                                                 replacement=False)].to(self.device)
                            inputs_2 = dataset[th.multinomial(th.ones(dataset.size(0)),
                                                                 num_inputs,
                                                                 replacement=False)].to(self.device)

                            inputs_embedding_1 = agent.encode_object(inputs_1)
                            messages_1, _, _ = agent.send(inputs_embedding_1)
                            messages_len_1 = find_lengths(messages_1)
                            messages_1, messages_len_1 = messages_1.cpu().numpy(), messages_len_1.cpu().numpy()

                            inputs_embedding_2 = agent.encode_object(inputs_2)
                            messages_2, _, _ = agent.send(inputs_embedding_2)
                            messages_len_2 = find_lengths(messages_2)
                            messages_2, messages_len_2 = messages_2.cpu().numpy(), messages_len_2.cpu().numpy()

                            if distance_input == "common_attributes":
                                distances_inputs = np.mean(1 - 1 * ((inputs_1 - inputs_2[:, 1, :]) == 0), axis=1)
                            else:
                                raise NotImplementedError

                            if distance_message == "edit_distance":
                                distances_messages = 1 - compute_language_similarity(messages_1=messages_1,
                                                                                     messages_2=messages_2,
                                                                                     len_messages_1=messages_len_1,
                                                                                     len_messages_2=messages_len_2)
                            else:
                                raise NotImplementedError

                            top_sim = spearmanr(distances_inputs, distances_messages).correlation

                            topographic_similarity_results[agent_name][split_type].append(top_sim)

        return topographic_similarity_results

    def estimate_MI(self):

        raise NotImplementedError

    def compute_success(self):

        raise NotImplementedError

    def save_results(self,
                     save_dir: str,
                     topographic_similarity: dict = None,
                     mutual_information: dict = None,
                     success: dict = None,
                     max_generalization: dict = None) -> None:

        # Topographic similarity
        if topographic_similarity is not None:
            for agent_name in topographic_similarity:
                for dataset_type in topographic_similarity[agent_name]:
                    th.save(f"{save_dir}/topsim_{agent_name}_{dataset_type}.pt")

        if mutual_information is not None:
            raise NotImplementedError

        if success is not None:
            raise NotImplementedError

        if max_generalization is not None:
            raise NotImplementedError

    def print_results(self,
                      topographic_similarity: dict = None,
                      mutual_information: dict = None,
                      success: dict = None,
                      max_generalization: dict = None):

        # Topographic similarity
        if topographic_similarity is not None:
            print("\n### TOPOGRAPHIC SIMILARITY### \n")
            for agent_name in topographic_similarity:
                print(f"Sender : {agent_name}")
                for dataset_type in topographic_similarity[agent_name]:
                    ts_values = topographic_similarity[agent_name][dataset_type]
                    print(f"{dataset_type} : mean={np.mean(ts_values)}, std = {np.std(ts_values)}")

        if mutual_information is not None:
            raise NotImplementedError

        if success is not None:
            raise NotImplementedError

        if max_generalization is not None:
            raise NotImplementedError


def get_static_evaluator(game,
                         population,
                         metrics_to_measure,
                         agents_to_evaluate,
                         dataset_dir,
                         save_dir,
                         device: str = "cpu"):
    evaluator = StaticEvaluator(game=game,
                                population=population,
                                metrics_to_measure=metrics_to_measure,
                                agents_to_evaluate=agents_to_evaluate,
                                dataset_dir=dataset_dir,
                                save_dir=save_dir,
                                device=device)

    return evaluator
