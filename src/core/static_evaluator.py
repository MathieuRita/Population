import torch as th
import numpy as np
import os
from scipy.stats import spearmanr
from .utils import move_to, find_lengths, from_att_to_one_hot_celeba
from .language_metrics import compute_language_similarity
from .agents import get_agent
from .datasets_onehot import get_all_one_hot_elements
from collections import defaultdict
from torch.nn import CosineSimilarity


class StaticEvaluator:

    def __init__(self,
                 game,
                 population,
                 metrics_to_measure,
                 agents_to_evaluate,
                 eval_receiver_id,
                 agent_repertory,
                 game_params,
                 dataset_dir,
                 save_dir,
                 uniform_sampling : bool = True,
                 device: str = "cpu") -> None:

        self.game = game
        self.population = population
        self.device = th.device(device)
        self.agents_to_evaluate = agents_to_evaluate
        self.metrics_to_measure = metrics_to_measure
        self.eval_receiver_id = eval_receiver_id
        self.agent_repertory = agent_repertory
        self.game_params = game_params
        self.uniform_sampling = uniform_sampling
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir

    def step(self,
             print_results: bool = False,
             save_results: bool = False) -> None:

        if "topographic_similarity" in self.metrics_to_measure:
            topographic_similarity = self.estimate_topographic_similarity()
        else:
            topographic_similarity = None

        if "h_x_m" in self.metrics_to_measure:
            h_x_m, accuracy_h_x_m, val_losses_results, val_accuracies_results = self.estimate_h_x_m()
        else:
            h_x_m, accuracy_h_x_m, val_losses_results, val_accuracies_results = None, None, None, None

        if "h_x_m_tot" in self.metrics_to_measure:
            h_x_m_tot, accuracy_h_x_m_tot,test_losses_results, test_accuracies_results = self.estimate_h_x_m_tot()
        else:
            h_x_m_tot, accuracy_h_x_m_tot,test_losses_results, test_accuracies_results = None, None, None, None

        if "success" in self.metrics_to_measure:
            success = compute_success()
        else:
            success = None

        if "speed_of_learning_listener" in self.metrics_to_measure:
            speed_of_learning_listener = self.estimate_speed_of_learning_listener()
        else:
            speed_of_learning_listener = None

        if "speed_of_learning_speaker" in self.metrics_to_measure:
            speed_of_learning_speaker = self.estimate_speed_of_learning_speaker()
        else:
            speed_of_learning_speaker = None

        if save_results:
            self.save_results(save_dir=self.save_dir,
                              topographic_similarity=topographic_similarity,
                              h_x_m=h_x_m,
                              accuracy_h_x_m = accuracy_h_x_m,
                              h_x_m_tot=h_x_m_tot,
                              accuracy_h_x_m_tot=accuracy_h_x_m_tot,
                              test_losses_results=test_losses_results,
                              test_accuracies_results=test_accuracies_results,
                              success=success,
                              val_losses_results=val_losses_results,
                              val_accuracies_results=val_accuracies_results,
                              speed_of_learning_listener=speed_of_learning_listener,
                              speed_of_learning_speaker=speed_of_learning_speaker)

        if print_results:
            self.print_results(topographic_similarity=topographic_similarity,
                               h_x_m=h_x_m,
                               accuracy_h_x_m=accuracy_h_x_m,
                               h_x_m_tot=h_x_m_tot,
                               accuracy_h_x_m_tot=accuracy_h_x_m_tot,
                               success=success,
                               val_losses_results=val_losses_results,
                               val_accuracies_results=val_accuracies_results,
                               speed_of_learning_listener=speed_of_learning_listener,
                               speed_of_learning_speaker=speed_of_learning_speaker)

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
                                equal_att = 1 - 1 * ((inputs_1.argmax(2) - inputs_2.argmax(2)) == 0).cpu().numpy()
                                distances_inputs = np.mean(equal_att,
                                                           axis=1)
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

    def estimate_h_x_m(self,
                       batch_size: int = 1024,
                       n_batch_val : int = 5):

        h_x_m_results = defaultdict()
        accuracy_results = defaultdict()
        val_losses_results = defaultdict()
        val_accuracies_results = defaultdict()
        full_dataset = th.load(f"{self.dataset_dir}/full_dataset.pt")

        for agent_name in self.agents_to_evaluate:

            agent = self.population.agents[agent_name]
            if agent.sender is not None:
                h_x_m_results[agent_name] = defaultdict(list)
                accuracy_results[agent_name] = defaultdict(list)
                train_split = th.load(f"{self.dataset_dir}/{agent_name}_train_split.pt")
                val_split = th.load(f"{self.dataset_dir}/{agent_name}_val_split.pt")
                test_split = th.load(f"{self.dataset_dir}/{agent_name}_test_split.pt")

                splits = {"train": train_split,
                          "val": val_split,
                          "test": test_split}

                for split_type in splits:

                    self.population.agents[self.eval_receiver_id] = get_agent(agent_name=self.eval_receiver_id,
                                                                              agent_repertory=self.agent_repertory,
                                                                              game_params=self.game_params,
                                                                              device=self.device)

                    self.game.train()
                    dataset = full_dataset[splits[split_type]]
                    task = "communication"

                    losses = []
                    accuracies = []

                    if split_type=="train": val_losses,val_accuracies = [], []

                    step = 0
                    continue_training = True

                    while continue_training:

                        # Prepare dataset
                        n_batch = max(round(len(splits[split_type]) / batch_size),1)
                        permutation = th.multinomial(th.ones(len(splits[split_type])),
                                                     len(splits[split_type]),
                                                     replacement=False)

                        if n_batch * batch_size - len(splits[split_type])<len(splits[split_type]):
                            replacement=False
                        else:
                            replacement = True

                        batch_fill = th.multinomial(th.ones(len(splits[split_type])),
                                                    n_batch * batch_size - len(splits[split_type]),
                                                    replacement=replacement)

                        permutation = th.cat((permutation,batch_fill),dim=0)

                        mean_loss = 0.
                        mean_accuracy = 0.

                        for i in range(n_batch):
                            batch_data = dataset[permutation[i * batch_size:(i + 1) * batch_size]]

                            eval_receiver = self.population.agents[self.eval_receiver_id]
                            batch = move_to((batch_data, agent_name, self.eval_receiver_id), self.device)
                            metrics = self.game(batch, compute_metrics=True)

                            eval_receiver.tasks[task]["optimizer"].zero_grad()
                            eval_receiver.tasks[task]["loss_value"].backward()
                            eval_receiver.tasks[task]["optimizer"].step()

                            mean_loss += eval_receiver.tasks[task]["loss_value"].detach().item()
                            mean_accuracy += metrics["accuracy"].detach().item()

                        losses.append(mean_loss / n_batch)
                        accuracies.append(mean_accuracy / n_batch)

                        if split_type=="train":
                            with th.no_grad():

                                mean_val_loss=0.
                                mean_val_acc=0.

                                for _ in range(n_batch_val):

                                    batch_data = full_dataset[splits["val"]]

                                    eval_receiver = self.population.agents[self.eval_receiver_id]
                                    batch = move_to((batch_data, agent_name, self.eval_receiver_id), self.device)
                                    metrics = self.game(batch, compute_metrics=True)

                                    mean_val_loss+=eval_receiver.tasks[task]["loss_value"].detach().item()
                                    mean_val_acc+=metrics["accuracy"].detach().item()

                                val_losses.append(mean_val_loss/n_batch_val)
                                val_accuracies.append(mean_val_acc/n_batch_val)

                        step += 1

                        if step == 2500 : continue_training = False

                    h_x_m_results[agent_name][split_type].append(np.mean(losses[-5:]))
                    accuracy_results[agent_name][split_type].append(np.mean(accuracies[-5:]))
                    if split_type=="train":
                        val_losses_results[agent_name] = val_losses
                        val_accuracies_results[agent_name] = val_accuracies
                    print(f"Done split {split_type} : {np.mean(losses[-5:])} / {np.mean(accuracies[-5:])}")

        return h_x_m_results, accuracy_results, val_losses_results, val_accuracies_results

    def estimate_h_x_m_tot(self,
                           batch_size: int = 1024,
                           n_batch_val : int = 10):

        h_x_m_results = defaultdict()
        test_losses_results = defaultdict()
        test_accuracies_results = defaultdict(list)
        accuracy_results = defaultdict(list)
        full_dataset = th.load(f"{self.dataset_dir}/full_dataset.pt")

        for agent_name in self.agents_to_evaluate:

            agent = self.population.agents[agent_name]

            train_split = th.load(f"{self.dataset_dir}/{agent_name}_train_split.pt")
            val_split = th.load(f"{self.dataset_dir}/{agent_name}_val_split.pt")
            test_split = th.load(f"{self.dataset_dir}/{agent_name}_test_split.pt")

            splits = {"train": train_split,
                      "val": val_split,
                      "test": test_split}

            if agent.sender is not None:
                h_x_m_results[agent_name] = list()
                accuracy_results[agent_name] = list()

                self.population.agents[self.eval_receiver_id] = get_agent(agent_name=self.eval_receiver_id,
                                                                          agent_repertory=self.agent_repertory,
                                                                          game_params=self.game_params,
                                                                          device=self.device)

                self.game.train()
                dataset = full_dataset
                task = "communication"

                losses = []
                accuracies = []


                step = 0
                continue_training = True

                while continue_training:

                    # Prepare dataset
                    n_batch = max(round(len(dataset) / batch_size),1)
                    permutation = th.multinomial(th.ones(len(dataset)),
                                                 len(dataset),
                                                 replacement=False)

                    if n_batch * batch_size - len(dataset)<len(dataset):
                        replacement=False
                    else:
                        replacement = True

                    batch_fill = th.multinomial(th.ones(len(dataset)),
                                                n_batch * batch_size - len(dataset),
                                                replacement=replacement)

                    permutation = th.cat((permutation,batch_fill),dim=0)

                    mean_loss = 0.
                    mean_accuracy = 0.

                    for i in range(n_batch):
                        batch_data = dataset[permutation[i * batch_size:(i + 1) * batch_size]]

                        eval_receiver = self.population.agents[self.eval_receiver_id]
                        batch = move_to((batch_data, agent_name, self.eval_receiver_id), self.device)
                        metrics = self.game(batch, compute_metrics=True)

                        eval_receiver.tasks[task]["optimizer"].zero_grad()
                        eval_receiver.tasks[task]["loss_value"].backward()
                        eval_receiver.tasks[task]["optimizer"].step()

                        mean_loss += eval_receiver.tasks[task]["loss_value"].detach().item()
                        mean_accuracy += metrics["accuracy"].detach().item()

                    if step%1000==0:
                        print(mean_loss/n_batch)
                    losses.append(mean_loss / n_batch)
                    accuracies.append(mean_accuracy / n_batch)

                    step += 1

                    if step == 15000 : continue_training = False

                test_losses=[]
                test_accuracies=[]

                with th.no_grad():

                    mean_test_loss = 0.
                    mean_test_acc = 0.

                    for _ in range(n_batch_val):
                        batch_data = full_dataset[splits["test"]]

                        eval_receiver = self.population.agents[self.eval_receiver_id]
                        batch = move_to((batch_data, agent_name, self.eval_receiver_id), self.device)
                        metrics = self.game(batch, compute_metrics=True)

                        mean_test_loss += eval_receiver.tasks[task]["loss_value"].detach().item()
                        mean_test_acc += metrics["accuracy"].detach().item()

                    test_losses.append(mean_test_loss / n_batch_val)
                    test_accuracies.append(mean_test_acc / n_batch_val)

                test_losses_results[agent_name] = test_losses
                test_accuracies_results[agent_name] = test_accuracies

                h_x_m_results[agent_name].append(np.mean(losses[-10:]))
                accuracy_results[agent_name].append(np.mean(accuracies[-10:]))
                print(f"Done measure : {np.mean(losses[-5:])} / {np.mean(accuracies[-5:])}")
                print(f"Done measure test : {np.mean(test_losses[-5:])} / {np.mean(test_accuracies[-5:])}")

        return h_x_m_results, accuracy_results, test_losses_results, test_accuracies_results

    def estimate_speed_of_learning_listener(self,
                                            batch_size: int = 1024,
                                            acc_threshold=0.99
                                            ):


        speed_of_learning = defaultdict()
        full_dataset = th.load(f"{self.dataset_dir}/full_dataset.pt")

        for agent_name in self.agents_to_evaluate:
            agent = self.population.agents[agent_name]

            train_split = th.load(f"{self.dataset_dir}/{agent_name}_train_split.pt")
            val_split = th.load(f"{self.dataset_dir}/{agent_name}_val_split.pt")
            test_split = th.load(f"{self.dataset_dir}/{agent_name}_test_split.pt")

            splits = {"train": train_split,
                      "val": val_split,
                      "test": test_split}

            if agent.sender is not None:

                self.population.agents[self.eval_receiver_id] = get_agent(agent_name=self.eval_receiver_id,
                                                                          agent_repertory=self.agent_repertory,
                                                                          game_params=self.game_params,
                                                                          device=self.device)

                self.game.train()
                dataset = full_dataset[splits['train']]
                task = "communication"

                losses = []
                accuracies = []

                step = 0
                continue_training = True

                while continue_training:

                    # Prepare dataset
                    n_batch = max(round(len(dataset) / batch_size),1)
                    permutation = th.multinomial(th.ones(len(dataset)),
                                                 len(dataset),
                                                 replacement=False)

                    if n_batch * batch_size - len(dataset)<len(dataset):
                        replacement=False
                    else:
                        replacement = True

                    batch_fill = th.multinomial(th.ones(len(dataset)),
                                                n_batch * batch_size - len(dataset),
                                                replacement=replacement)

                    permutation = th.cat((permutation,batch_fill),dim=0)

                    mean_loss = 0.
                    mean_accuracy = 0.

                    for i in range(n_batch):
                        batch_data = dataset[permutation[i * batch_size:(i + 1) * batch_size]]

                        eval_receiver = self.population.agents[self.eval_receiver_id]
                        batch = move_to((batch_data, agent_name, self.eval_receiver_id), self.device)
                        metrics = self.game(batch, compute_metrics=True)

                        eval_receiver.tasks[task]["optimizer"].zero_grad()
                        eval_receiver.tasks[task]["loss_value"].backward()
                        eval_receiver.tasks[task]["optimizer"].step()

                        mean_loss += eval_receiver.tasks[task]["loss_value"].detach().item()
                        mean_accuracy += metrics["accuracy"].detach().item()

                    losses.append(mean_loss / n_batch)
                    accuracies.append(mean_accuracy / n_batch)

                    step += 1

                    if step == 10000 : continue_training = False

                if len(np.where(np.array(accuracies) >= acc_threshold)[0])>0:
                    speed_of_learning[agent_name] = np.min(np.where(np.array(accuracies) >= acc_threshold)[0])
                else:
                    speed_of_learning[agent_name] = -1

        return speed_of_learning

    def estimate_speed_of_learning_speaker(self,
                                           batch_size: int = 1024,
                                           training_procedure="supervision"):

        if training_procedure=="supervision":

            speed_of_learning = defaultdict()
            full_dataset = th.load(f"{self.dataset_dir}/full_dataset.pt")

            for agent_name in self.agents_to_evaluate:
                agent = self.population.agents[agent_name]

                train_split = th.load(f"{self.dataset_dir}/{agent_name}_train_split.pt")
                val_split = th.load(f"{self.dataset_dir}/{agent_name}_val_split.pt")
                test_split = th.load(f"{self.dataset_dir}/{agent_name}_test_split.pt")

                splits = {"train": train_split,
                          "val": val_split,
                          "test": test_split}

                if agent.sender is not None:
                    imitator = self.population.agents["imitator"]

                    self.game.train()
                    dataset = full_dataset[splits['train']]
                    task = "imitation"

                    losses = []
                    accuracies = []

                    step = 0
                    continue_training = True

                    while continue_training:

                        # Prepare dataset
                        n_batch = max(round(len(dataset) / batch_size), 1)
                        permutation = th.multinomial(th.ones(len(dataset)),
                                                     len(dataset),
                                                     replacement=False)

                        if n_batch * batch_size - len(dataset) < len(dataset):
                            replacement = False
                        else:
                            replacement = True

                        batch_fill = th.multinomial(th.ones(len(dataset)),
                                                    n_batch * batch_size - len(dataset),
                                                    replacement=replacement)

                        permutation = th.cat((permutation, batch_fill), dim=0)

                        mean_loss = 0.

                        for i in range(n_batch):
                            batch_data = move_to(dataset[permutation[i * batch_size:(i + 1) * batch_size]],
                                                 self.device)

                            self.game.imitation_instance(batch_data,agent_name,"imitator")

                            imitator.tasks[task]["optimizer"].zero_grad()
                            imitator.tasks[task]["loss_value"].backward()
                            imitator.tasks[task]["optimizer"].step()

                            mean_loss += imitator.tasks[task]["loss_value"].detach().item()

                        losses.append(mean_loss / n_batch)

                        step += 1

                        if step == 100: continue_training = False

                    #if len(np.where(np.array(accuracies) >= loss_threshold)[0]) > 0:
                    #    speed_of_learning[agent_name] = np.min(np.where(np.array(losses) >= loss_threshold)[0])
                    #else:
                    #    speed_of_learning[agent_name] = -1
                    speed_of_learning[agent_name] = losses[-1]

        else:
            raise AssertionError

        return speed_of_learning

    def compute_success(self,
                        batch_size=1024,
                        n_steps=10):

        accuracy_results = defaultdict()
        full_dataset = th.load(f"{self.dataset_dir}/full_dataset.pt")

        for sender_name in self.agents_to_evaluate:

            agent_sender = self.population.agents[sender_name]

            if agent_sender.sender is not None:

                accuracy_results[sender_name] = defaultdict()

                for receiver_name in self.agents_to_evaluate:

                    agent_receiver = self.population.agents[receiver_name]
                    accuracy_results[sender_name][receiver_name] = defaultdict()

                    if agent_receiver.receiver is not None:

                        train_split = th.load(f"{self.dataset_dir}/{sender_name}_train_split.pt")
                        val_split = th.load(f"{self.dataset_dir}/{sender_name}_val_split.pt")
                        test_split = th.load(f"{self.dataset_dir}/{sender_name}_test_split.pt")

                        splits = {"train": train_split,
                                  "val": val_split,
                                  "test": test_split}

                        for split_type in splits:

                            accuracy_results[sender_name][receiver_name][split_type] = list()

                            self.game.train()
                            if not self.uniform_sampling and split_type!="train":
                                dataset = get_all_one_hot_elements(self.game_params["objects"])
                            else:
                                dataset = full_dataset[splits[split_type]]

                            for _ in range(n_steps):

                                # Prepare dataset
                                n_batch = max(round(len(dataset) / batch_size), 1)
                                permutation = th.multinomial(th.ones(len(dataset)),
                                                             len(dataset),
                                                             replacement=False)

                                if n_batch * batch_size - len(dataset) < len(dataset):
                                    replacement = False
                                else:
                                    replacement = True

                                batch_fill = th.multinomial(th.ones(len(dataset)),
                                                            n_batch * batch_size - len(dataset),
                                                            replacement=replacement)

                                permutation = th.cat((permutation, batch_fill), dim=0)

                                mean_accuracy = 0.

                                for i in range(n_batch):
                                    batch_data = dataset[permutation[i * batch_size:(i + 1) * batch_size]]

                                    batch = move_to((batch_data, sender_name, receiver_name), self.device)
                                    metrics = self.game(batch, compute_metrics=True, reduce = False)
                                    mean_accuracy += metrics["accuracy"].detach().cpu().numpy()

                                accuracy_results[sender_name][receiver_name][split_type].append(mean_accuracy / n_batch)

                            accuracy_results[sender_name][receiver_name][split_type] = \
                                th.stack(accuracy_results[sender_name][receiver_name][split_type]).mean(0)
        return accuracy_results

    def get_messages(self):

        raise NotImplementedError

    def save_results(self,
                     save_dir: str,
                     topographic_similarity: dict = None,
                     h_x_m: dict = None,
                     accuracy_h_x_m: dict = None,
                     h_x_m_tot : dict = None,
                     accuracy_h_x_m_tot : dict = None,
                     test_losses_results : dict = None,
                     test_accuracies_results : dict = None,
                     success: dict = None,
                     val_losses_results: dict = None,
                     val_accuracies_results: dict = None,
                     speed_of_learning_listener : dict = None,
                     speed_of_learning_speaker : dict = None) -> None:

        # Topographic similarity
        if topographic_similarity is not None:
            for agent_name in topographic_similarity:
                for dataset_type in topographic_similarity[agent_name]:
                    np.save(f"{save_dir}/topsim_{agent_name}_{dataset_type}.npy",
                            topographic_similarity[agent_name][dataset_type])

        if h_x_m is not None:
            for agent_name in h_x_m:
                for dataset_type in h_x_m[agent_name]:
                    np.save(f"{save_dir}/h_x_m_{agent_name}_{dataset_type}.npy",
                            h_x_m[agent_name][dataset_type])

        if h_x_m_tot is not None:
            for agent_name in h_x_m_tot:
                    np.save(f"{save_dir}/h_x_m_tot_{agent_name}.npy",
                            h_x_m_tot[agent_name])

                    np.save(f"{save_dir}/h_x_m_tot_test_{agent_name}.npy",
                            test_losses_results[agent_name])


        if accuracy_h_x_m_tot is not None:
            for agent_name in accuracy_h_x_m_tot:
                    np.save(f"{save_dir}/accuracy_h_x_m_tot_{agent_name}.npy",
                            accuracy_h_x_m_tot[agent_name])
                    np.save(f"{save_dir}/accuracy_h_x_m_tot_test_{agent_name}.npy",
                            test_accuracies_results[agent_name])

        if accuracy_h_x_m is not None:
            for agent_name in accuracy_h_x_m:
                for dataset_type in accuracy_h_x_m[agent_name]:
                    np.save(f"{save_dir}/accuracy_h_x_m_{agent_name}_{dataset_type}.npy",
                            accuracy_h_x_m[agent_name][dataset_type])
                np.save(f"{save_dir}/val_loss_{agent_name}.npy",
                            val_losses_results[agent_name])
                np.save(f"{save_dir}/val_accuracy_{agent_name}.npy",
                        val_accuracies_results[agent_name])

        if speed_of_learning_listener is not None:
            for agent_name in speed_of_learning_listener:
                np.save(f"{save_dir}/speed_of_learning_listener_{agent_name}.npy",
                        speed_of_learning_listener[agent_name])
                np.save(f"{save_dir}/speed_of_learning_listener_{agent_name}.npy",
                        speed_of_learning_listener[agent_name])

        if speed_of_learning_speaker is not None:
            for agent_name in speed_of_learning_speaker:
                np.save(f"{save_dir}/speed_of_learning_speaker_{agent_name}.npy",
                        speed_of_learning_speaker[agent_name])
                np.save(f"{save_dir}/speed_of_learning_speaker_{agent_name}.npy",
                        speed_of_learning_speaker[agent_name])

        if success is not None:
            for sender_name in success:
                for receiver_name in success[sender_name]:
                    for split_type in success[sender_name][receiver_name]:
                        mean_accuracy = success[sender_name][receiver_name][split_type]
                        np.save(f"{save_dir}/accuracy_{sender_name}_{receiver_name}_{split_type}.npy",
                                mean_accuracy)

    def print_results(self,
                      topographic_similarity: dict = None,
                      h_x_m: dict = None,
                      accuracy_h_x_m: dict = None,
                      h_x_m_tot: dict = None,
                      accuracy_h_x_m_tot: dict = None,
                      success: dict = None,
                      val_losses_results: dict = None,
                      val_accuracies_results: dict = None,
                      speed_of_learning_listener : dict = None,
                      speed_of_learning_speaker : dict = None) -> None:

        # Topographic similarity
        if topographic_similarity is not None:
            print("\n### TOPOGRAPHIC SIMILARITY### \n")
            for agent_name in topographic_similarity:
                print(f"Sender : {agent_name}")
                for dataset_type in topographic_similarity[agent_name]:
                    ts_values = topographic_similarity[agent_name][dataset_type]
                    print(f"{dataset_type} : mean={np.mean(ts_values)}, std = {np.std(ts_values)}")

        if h_x_m is not None:
            print("\n### CONDITIONAL ENTROPY### \n")
            for agent_name in h_x_m:
                print(f"Sender : {agent_name}")
                for dataset_type in h_x_m[agent_name]:
                    h_value = h_x_m[agent_name][dataset_type]
                    accuracy_h = accuracy_h_x_m[agent_name][dataset_type]
                    print(f"{dataset_type} : {h_value} (accuracy = {accuracy_h})")
                print(f"Minimal validation loss = {np.min(val_losses_results[agent_name])}")
                print(f"Maximal validation accuracy = {np.max(val_accuracies_results[agent_name])}")

        if h_x_m_tot is not None:
            print("\n### CONDITIONAL ENTROPY TOT### \n")
            for agent_name in h_x_m_tot:
                print(f"Sender : {agent_name}")
                h_value = h_x_m_tot[agent_name]
                accuracy_h = accuracy_h_x_m_tot[agent_name]
                print(f"Measure : {h_value} (accuracy = {accuracy_h})")

        if speed_of_learning_listener is not None:
            for agent_name in speed_of_learning_listener:
                print(f"Sender : {agent_name}")
                speed_of_learning_value = speed_of_learning_listener[agent_name]
                print(f'Speed of learning = {speed_of_learning_value}')

        if speed_of_learning_speaker is not None:
            for agent_name in speed_of_learning_speaker:
                print(f"Sender : {agent_name}")
                speed_of_learning_value = speed_of_learning_speaker[agent_name]
                print(f'Speed of imitating = {speed_of_learning_value}')

        if success is not None:
            for sender_name in success:
                for receiver_name in success[sender_name]:
                    for split_type in success[sender_name][receiver_name]:
                        mean_accuracy = success[sender_name][receiver_name][split_type]
                        print(f"Sender : {sender_name} ; Receiver : {receiver_name}")
                        print(f"accuracy_{split_type} = {mean_accuracy}")


class StaticEvaluatorImage:

    def __init__(self,
                 game,
                 population,
                 metrics_to_measure,
                 agents_to_evaluate,
                 eval_receiver_id,
                 agent_repertory,
                 game_params,
                 dataset_dir,
                 image_dataset,
                 save_dir,
                 device: str = "cpu") -> None:

        self.game = game
        self.population = population
        self.device = th.device(device)
        self.agents_to_evaluate = agents_to_evaluate
        self.metrics_to_measure = metrics_to_measure
        self.eval_receiver_id = eval_receiver_id
        self.agent_repertory = agent_repertory
        self.game_params = game_params
        self.dataset_dir = dataset_dir
        self.image_dataset = image_dataset
        self.save_dir = save_dir

    def step(self,
             print_results: bool = False,
             save_results: bool = False) -> None:

        if "topographic_similarity" in self.metrics_to_measure:
            if self.image_dataset=="imagenet":
                topographic_similarity_cosine = self.estimate_topographic_similarity(distance_input="cosine_similarity")
                topographic_similarity_attributes=None
            elif self.image_dataset=="celeba":
                topographic_similarity_cosine = self.estimate_topographic_similarity(distance_input="cosine_similarity")
                topographic_similarity_attributes = self.estimate_topographic_similarity(distance_input="common_attributes")
            else:
                raise("Specify a known type of image dataset")
        else:
            topographic_similarity = None

        if save_results:
            self.save_results(save_dir=self.save_dir,
                              topographic_similarity_cosine=topographic_similarity_cosine,
                              topographic_similarity_attributes=topographic_similarity_attributes)

        if print_results:
            self.print_results(topographic_similarity_cosine=topographic_similarity_cosine,
                              topographic_similarity_attributes=topographic_similarity_attributes)

    def estimate_topographic_similarity(self,
                                        distance_input: str = "cosine_similarity",
                                        distance_message: str = "edit_distance",
                                        N_sampling: int = 100,
                                        batch_size: int = 1000) -> dict:

        topographic_similarity_results = dict()

        self.game.train()

        with th.no_grad():

            for agent_name in self.agents_to_evaluate:
                agent = self.population.agents[agent_name]
                if agent.sender is not None:
                    topographic_similarity_results[agent_name] = list()

                    test_set = [th.load(f"{self.dataset_dir}/{f}") for f in os.listdir(self.dataset_dir) if "test" in f]

                    dataset = test_set

                    # Train
                    for _ in range(N_sampling):

                        # Sample random file
                        random_file_id = np.random.choice(len(dataset))
                        random_file = dataset[random_file_id]

                        # Select random split inside the file
                        random_samples_ids_1 = np.random.choice(len(random_file), batch_size, replace=False)
                        random_samples_ids_2 = np.random.choice(len(random_file), batch_size, replace=False)
                        if distance_input == "cosine_similarity":
                            inputs_1 = th.Tensor(
                                [sample["logit"] for sample in np.array(random_file)[random_samples_ids_1]]).to(self.device)
                            inputs_2 = th.Tensor(
                                [sample["logit"] for sample in np.array(random_file)[random_samples_ids_2]]).to(self.device)
                        elif distance_input == "common_attributes":
                            inputs_1 = th.Tensor(
                                [sample["logit"] for sample in np.array(random_file)[random_samples_ids_1]]).to(
                                self.device)
                            inputs_2 = th.Tensor(
                                [sample["logit"] for sample in np.array(random_file)[random_samples_ids_2]]).to(
                                self.device)
                            att_1 = th.Tensor(
                                [from_att_to_one_hot_celeba(sample["attributes"]) for sample in np.array(random_file)[random_samples_ids_1]]).to(
                                self.device)
                            att_2 = th.Tensor(
                                [from_att_to_one_hot_celeba(sample["attributes"]) for sample in np.array(random_file)[random_samples_ids_2]]).to(
                                self.device)
                        else:
                            raise("Specify a known distance")

                        inputs_embedding_1 = agent.encode_object(inputs_1)
                        messages_1, _, _ = agent.send(inputs_embedding_1)
                        messages_len_1 = find_lengths(messages_1)
                        messages_1, messages_len_1 = messages_1.cpu().numpy(), messages_len_1.cpu().numpy()

                        inputs_embedding_2 = agent.encode_object(inputs_2)
                        messages_2, _, _ = agent.send(inputs_embedding_2)
                        messages_len_2 = find_lengths(messages_2)
                        messages_2, messages_len_2 = messages_2.cpu().numpy(), messages_len_2.cpu().numpy()

                        if distance_input == "cosine_similarity":
                            cos = CosineSimilarity(dim=1)
                            distances_inputs = 1-cos(inputs_1,inputs_2).cpu().numpy()
                        elif distance_input == "common_attributes":
                            equal_att = 1 - 1 * ((att_1 - att_2) == 0).cpu().numpy()
                            distances_inputs = np.mean(equal_att,
                                                       axis=1)

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

                        topographic_similarity_results[agent_name].append(top_sim)

        return topographic_similarity_results

    def save_results(self,
                     save_dir: str,
                     topographic_similarity_cosine: dict = None,
                     topographic_similarity_attributes: dict = None) -> None:

        # Topographic similarity
        if topographic_similarity_cosine is not None:
            for agent_name in topographic_similarity_cosine:
                    np.save(f"{save_dir}/topsim_cosine_{agent_name}.npy",
                            topographic_similarity_cosine[agent_name])

        if topographic_similarity_attributes is not None:
            for agent_name in topographic_similarity_attributes:
                    np.save(f"{save_dir}/topsim_attributes_{agent_name}.npy",
                            topographic_similarity_attributes[agent_name])

    def print_results(self,
                      topographic_similarity_cosine: dict = None,
                      topographic_similarity_attributes: dict = None):

        # Topographic similarity
        if topographic_similarity_cosine is not None:
            print("\n### TOPOGRAPHIC SIMILARITY### \n")
            for agent_name in topographic_similarity_cosine:
                print(f"Sender : {agent_name}")
                ts_values = topographic_similarity_cosine[agent_name]
                print(f"Cosine  : mean={np.mean(ts_values)}, std = {np.std(ts_values)}")


        if topographic_similarity_attributes is not None:
            print("\n### TOPOGRAPHIC SIMILARITY### \n")
            for agent_name in topographic_similarity_attributes:
                print(f"Sender : {agent_name}")
                ts_values = topographic_similarity_attributes[agent_name]
                print(f"Attributes  : mean={np.mean(ts_values)}, std = {np.std(ts_values)}")


def get_static_evaluator(game,
                         population,
                         metrics_to_measure,
                         agents_to_evaluate,
                         eval_receiver_id,
                         agent_repertory,
                         game_params,
                         dataset_dir,
                         save_dir,
                         image_dataset : str =None,
                         device: str = "cpu"):

    if image_dataset is None:
        evaluator = StaticEvaluator(game=game,
                                    population=population,
                                    metrics_to_measure=metrics_to_measure,
                                    agents_to_evaluate=agents_to_evaluate,
                                    eval_receiver_id=eval_receiver_id,
                                    agent_repertory=agent_repertory,
                                    game_params=game_params,
                                    dataset_dir=dataset_dir,
                                    save_dir=save_dir,
                                    device=device)

    else:
        evaluator = StaticEvaluatorImage(game=game,
                                        population=population,
                                        metrics_to_measure=metrics_to_measure,
                                        agents_to_evaluate=agents_to_evaluate,
                                        eval_receiver_id=eval_receiver_id,
                                        agent_repertory=agent_repertory,
                                        game_params=game_params,
                                        dataset_dir=dataset_dir,
                                        save_dir=save_dir,
                                        image_dataset = image_dataset,
                                        device=device)

    return evaluator
