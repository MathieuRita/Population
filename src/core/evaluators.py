import torch as th
import numpy as np
from scipy.stats import spearmanr
from .utils import move_to, find_lengths
import torch.nn.functional as F
from .language_metrics import compute_language_similarity
from .agents import get_agent, copy_agent
from collections import defaultdict


class Evaluator:

    def __init__(self,
                 metrics_to_measure,
                 game,
                 dump_batch,
                 train_loader,
                 val_loader,
                 test_loader,
                 mi_loader,
                 agent_repertory,
                 game_params,
                 eval_receiver_id,
                 n_epochs,
                 logger: th.utils.tensorboard.SummaryWriter = None,
                 device: str = "cpu") -> None:

        self.game = game
        self.population = game.population
        self.n_epochs = n_epochs
        self.dump_batch = dump_batch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mi_loader = mi_loader
        self.agent_repertory = agent_repertory
        self.game_params = game_params
        self.eval_receiver_id = eval_receiver_id
        self.device = th.device(device)
        self.writer = logger
        self.init_languages = None

        # Metrics init
        self.metrics_to_measure = metrics_to_measure
        self.stored_metrics = {}

        if self.metrics_to_measure["reward_decomposition"]<self.n_epochs:
            self.stored_metrics["reward"] = list()
            self.stored_metrics["reward_coordination"] = list()
            self.stored_metrics["reward_information"] = list()
        if self.metrics_to_measure["language_similarity"]<self.n_epochs:
            self.stored_metrics["language_similarity"] = list()
        if self.metrics_to_measure["similarity_to_init_languages"]<self.n_epochs:
            self.stored_metrics["similarity_to_init_languages"] = list()
        if self.metrics_to_measure["divergence_to_untrained_speakers"]<self.n_epochs:
            self.stored_metrics["divergence_to_untrained_speakers"] = list()
        if self.metrics_to_measure["accuracy_with_untrained_speakers"]<self.n_epochs:
            self.stored_metrics["accuracy_with_untrained_speakers"] = list()
        if self.metrics_to_measure["accuracy_with_untrained_listeners"]<self.n_epochs:
            self.stored_metrics["accuracy_with_untrained_listeners"] = list()
        if self.metrics_to_measure["topographic_similarity"]<self.n_epochs:
            self.stored_metrics["topographic_similarity"] = list()
        if self.metrics_to_measure["external_receiver_evaluation"]<self.n_epochs:
            self.stored_metrics["external_receiver_train_acc"] = list()
            self.stored_metrics["external_receiver_val_acc"] = list()
            self.stored_metrics["external_receiver_train_loss"] = list()
            self.stored_metrics["external_receiver_val_loss"] = list()
            self.stored_metrics["etl"] = list()
        if self.metrics_to_measure["MI"]<self.n_epochs:
            self.stored_metrics["MI"] = defaultdict(list)

    def step(self,
             epoch: int) -> None:

        if epoch % self.metrics_to_measure["reward_decomposition"] == 0:
            reward_total, reward_information, reward_coordination = self.reward_decomposition()
            self.stored_metrics["reward"].append(reward_total)
            self.stored_metrics["reward_coordination"].append(reward_coordination)
            self.stored_metrics["reward_information"].append(reward_information)

        if epoch % self.metrics_to_measure["language_similarity"] == 0:
            similarity_matrix = self.evaluate_language_similarity(n_samples=10)
            self.stored_metrics["language_similarity"].append(similarity_matrix)

        if epoch % self.metrics_to_measure["similarity_to_init_languages"] == 0:
            similarity_to_init_languages = self.evaluate_distance_from_init_language(n_samples=10)
            self.stored_metrics["similarity_to_init_languages"].append(similarity_to_init_languages)

        if epoch % self.metrics_to_measure["divergence_to_untrained_speakers"] == 0:
            divergence_matrix = self.divergence_to_untrained_speakers(n_samples=50)
            self.stored_metrics["divergence_to_untrained_speakers"].append(divergence_matrix)

        if epoch % self.metrics_to_measure["accuracy_with_untrained_speakers"] == 0:
            accuracy_with_untrained_speakers = self.accuracy_with_untrained_speakers()
            self.stored_metrics["accuracy_with_untrained_speakers"].append(accuracy_with_untrained_speakers)

        if epoch % self.metrics_to_measure["accuracy_with_untrained_listeners"] == 0:
            accuracy_with_untrained_listeners = self.accuracy_with_untrained_listeners()
            self.stored_metrics["accuracy_with_untrained_listeners"].append(accuracy_with_untrained_listeners)

        if epoch % self.metrics_to_measure["topographic_similarity"] == 0:
            top_sim = self.evaluate_topographic_similarity()
            self.stored_metrics["topographic_similarity"].append(top_sim)

        if epoch % self.metrics_to_measure["external_receiver_evaluation"] == 0:
            train_acc, top_val_acc, etl, train_loss, val_loss = self.evaluate_external_receiver()
            self.stored_metrics["external_receiver_train_acc"].append(train_acc)
            self.stored_metrics["external_receiver_val_acc"].append(top_val_acc)
            self.stored_metrics["external_receiver_train_loss"].append(train_loss)
            self.stored_metrics["external_receiver_val_loss"].append(val_loss)
            self.stored_metrics["etl"].append(etl)

        if epoch % self.metrics_to_measure["MI"] == 0:
            mi_values = self.evaluate_mi_with_optimal_listener()
            for sender in mi_values:
                self.stored_metrics["MI"][sender].append(mi_values[sender])

        if epoch % self.metrics_to_measure["overfitting"] == 0:
            self.evaluate_overfitting(iter=epoch,eval_sender=True,eval_receiver=True)


        if epoch % self.metrics_to_measure["writing"] == 0 and self.writer is not None:
            self.log_metrics(iter=epoch)

    def evaluate_mi_with_optimal_listener(self) -> defaultdict():

        mi_values = defaultdict()

        for sender_id in self.population.sender_names:

            # MI train
            if self.population.agents[sender_id].optimal_listener is not None:
                # TRAIN TIL CONVERGENCE

                agent_sender = self.population.agents[sender_id]

                mi_values[sender_id] = defaultdict()

                for dataset_type in agent_sender.optimal_listener:

                    if dataset_type=="train":
                        loader = self.train_loader
                    elif dataset_type=="val":
                        loader = self.val_loader
                    elif dataset_type=="test":
                        loader = self.test_loader
                    else:
                        raise Exception("Specify a known dataset type for MI evaluation")

                    optimal_listener_id = agent_sender.optimal_listener[dataset_type]
                    optimal_listener = self.population.agents[optimal_listener_id]

                    model_parameters = list(optimal_listener.receiver.parameters()) + \
                                       list(optimal_listener.object_decoder.parameters())
                    optimal_listener.tasks["communication"]["optimizer"] = th.optim.Adam(model_parameters,
                                                                                         lr=0.0005)

                    prev_loss_value = [0.]
                    step = 0
                    task = "communication"

                    continue_optimal_listener_training = True

                    while continue_optimal_listener_training:

                        self.game.train()

                        mean_loss_value = 0.
                        n_batch = 0

                        for batch in loader:
                            inputs, sender_id = batch.data, batch.sender_id
                            inputs = inputs[th.randperm(inputs.size()[0])]
                            batch = move_to((inputs, sender_id, optimal_listener_id), self.device)

                            _ = self.game(batch)

                            optimal_listener.tasks[task]["optimizer"].zero_grad()
                            optimal_listener.tasks[task]["loss_value"].backward()
                            optimal_listener.tasks[task]["optimizer"].step()

                            mean_loss_value += optimal_listener.tasks[task]["loss_value"].item()
                            n_batch += 1

                        mean_loss_value /= n_batch
                        step += 1

                        if (len(prev_loss_value) > 9 and abs(mean_loss_value - np.mean(prev_loss_value)) < 10e-4) or \
                                step > 200:
                        #if step == 2500:
                            continue_optimal_listener_training = False
                            prev_loss_value.append(mean_loss_value)
                            prev_loss_value.pop(0)
                        else:
                            prev_loss_value.append(mean_loss_value)
                            if len(prev_loss_value) > 10:
                                prev_loss_value.pop(0)

                    mi_values[sender_id][dataset_type] = np.mean(prev_loss_value[-5:])

        return mi_values

    def evaluate_overfitting(self,
                             n_step_train : int = 400,
                             iter: int = 200,
                             eval_sender : bool = False,
                             eval_receiver : bool = False):

        if eval_receiver:
            for i,sender_id in enumerate(self.population.sender_names):
                for j,receiver_id in enumerate(self.population.receiver_names):

                    # Copy agents
                    self.population.agents[sender_id+"_copy"] = copy_agent(self.population.agents[sender_id],
                                                                           agent_repertory=self.agent_repertory,
                                                                           game_params=self.game_params,
                                                                           device=self.device)
                    self.population.agents[receiver_id + "_copy"] = copy_agent(self.population.agents[receiver_id],
                                                                               agent_repertory=self.agent_repertory,
                                                                               game_params=self.game_params,
                                                                               device=self.device)

                    for i in range(n_step_train):

                        self.game.train()

                        mean_train_acc = 0.
                        mean_train_loss = 0.
                        n_batch = 0

                        # for batch in self.train_loader:
                        for batch in self.train_loader:

                            task = "communication"

                            receiver_copy = self.population.agents[receiver_id + "_copy"]

                            batch = move_to((batch.data, sender_id+"_copy", receiver_id + "_copy"), self.device)
                            metrics = self.game(batch, compute_metrics=True)

                            receiver_copy.tasks[task]["optimizer"].zero_grad()
                            receiver_copy.tasks[task]["loss_value"].backward()
                            receiver_copy.tasks[task]["optimizer"].step()

                            mean_train_acc += metrics["accuracy"].detach().item()
                            mean_train_loss += receiver_copy.tasks[task]["loss_value"].detach().item()
                            n_batch += 1


                        self.writer.add_scalar(f'{receiver_id}_copy_{iter}/train accuracy',
                                               mean_train_acc / n_batch,
                                               i)
                        self.writer.add_scalar(f'{receiver_id}_copy_{iter}/train loss',
                                               mean_train_loss / n_batch,
                                               i)

                        self.game.eval()

                        mean_val_acc = 0.
                        mean_val_loss = 0.
                        n_batch = 0

                        # Eval
                        with th.no_grad():
                            for batch in self.val_loader:
                                batch = move_to((batch.data, sender_id+"_copy", receiver_id + "_copy"), self.device)
                                receiver_copy = self.population.agents[receiver_id + "_copy"]
                                metrics = self.game(batch, compute_metrics=True)

                                mean_val_acc += metrics["accuracy"].detach().item()
                                mean_val_loss += receiver_copy.tasks[task]["loss_value"].item()
                                n_batch += 1

                        self.writer.add_scalar(f'{receiver_id}_copy_{iter}/Val accuracy',
                                               mean_val_acc / n_batch,
                                               i)
                        self.writer.add_scalar(f'{receiver_id}_copy_{iter}/Val loss',
                                               mean_val_loss / n_batch,
                                               i)

        if eval_sender:

            for i, sender_id in enumerate(self.population.sender_names):
                for j, receiver_id in enumerate(self.population.receiver_names):

                    # Copy agents
                    self.population.agents[sender_id + "_copy"] = copy_agent(self.population.agents[sender_id],
                                                                             agent_repertory=self.agent_repertory,
                                                                             game_params=self.game_params,
                                                                             device=self.device)
                    self.population.agents[receiver_id + "_copy"] = copy_agent(self.population.agents[receiver_id],
                                                                               agent_repertory=self.agent_repertory,
                                                                               game_params=self.game_params,
                                                                               device=self.device)

                    for i in range(n_step_train):

                        self.game.train()

                        mean_train_acc = 0.
                        mean_train_loss = 0.
                        n_batch = 0

                        # for batch in self.train_loader:
                        for batch in self.train_loader:
                            task = "communication"

                            receiver_copy = self.population.agents[receiver_id + "_copy"]
                            sender_copy = self.population.agents[sender_id + "_copy"]

                            batch = move_to((batch.data, sender_id + "_copy", receiver_id + "_copy"), self.device)
                            metrics = self.game(batch, compute_metrics=True)

                            sender_copy.tasks[task]["optimizer"].zero_grad()
                            sender_copy.tasks[task]["loss_value"].backward()
                            sender_copy.tasks[task]["optimizer"].step()

                            mean_train_acc += metrics["accuracy"].detach().item()
                            mean_train_loss += receiver_copy.tasks[task]["loss_value"].detach().item()
                            n_batch += 1

                        self.writer.add_scalar(f'{sender_id}_copy_{iter}/train accuracy',
                                               mean_train_acc / n_batch,
                                               i)
                        self.writer.add_scalar(f'{sender_id}_copy_{iter}/train loss',
                                               mean_train_loss / n_batch,
                                               i)

                        self.game.eval()

                        mean_val_acc = 0.
                        mean_val_loss = 0.
                        n_batch = 0

                        # Eval
                        with th.no_grad():
                            for batch in self.val_loader:
                                batch = move_to((batch.data, sender_id + "_copy", receiver_id + "_copy"), self.device)
                                receiver_copy = self.population.agents[receiver_id + "_copy"]
                                sender_copy = self.population.agents[sender_id + "_copy"]
                                metrics = self.game(batch, compute_metrics=True)

                                mean_val_acc += metrics["accuracy"].detach().item()
                                mean_val_loss += receiver_copy.tasks[task]["loss_value"].item()
                                n_batch += 1

                        self.writer.add_scalar(f'{sender_id}_copy_{iter}/Val accuracy',
                                               mean_val_acc / n_batch,
                                               i)
                        self.writer.add_scalar(f'{sender_id}_copy_{iter}/Val loss',
                                               mean_val_loss / n_batch,
                                               i)

    def evaluate_external_receiver(self, n_step_train: int = 200, early_stopping: bool = True):

        train_accs = np.zeros(len(self.population.sender_names))
        top_val_accs = np.zeros(len(self.population.sender_names))
        train_loss_all_senders = np.zeros(len(self.population.sender_names))
        val_loss_all_senders = np.zeros(len(self.population.sender_names))
        etls = np.zeros(len(self.population.sender_names))

        for i, sender_id in enumerate(self.population.sender_names):

            # TO MODIFY AGENT_PARAMS
            self.population.agents[self.eval_receiver_id] = get_agent(agent_name=self.eval_receiver_id,
                                                                      agent_repertory=self.agent_repertory,
                                                                      game_params=self.game_params,
                                                                      device=self.device)

            train_accuracies = []
            train_losses = []
            val_accuracies = []
            val_losses = []
            step = 0
            continue_training = True

            while continue_training:

                # Train

                self.game.train()

                mean_train_acc = 0.
                mean_train_loss = 0.
                n_batch = 0

                # for batch in self.train_loader:
                for batch in self.train_loader:

                    eval_receiver = self.population.agents[self.eval_receiver_id]

                    task = "communication"

                    batch = move_to((batch.data, sender_id, self.eval_receiver_id), self.device)
                    metrics = self.game(batch, compute_metrics=True)

                    eval_receiver.tasks[task]["optimizer"].zero_grad()
                    eval_receiver.tasks[task]["loss_value"].backward()
                    eval_receiver.tasks[task]["optimizer"].step()

                    mean_train_acc += metrics["accuracy"].detach().item()
                    mean_train_loss += eval_receiver.tasks[task]["loss_value"].detach().item()
                    n_batch += 1

                train_accuracies.append(mean_train_acc / n_batch)
                train_losses.append(mean_train_loss / n_batch)

                self.game.eval()

                mean_val_acc = 0.
                mean_val_loss = 0.
                n_batch = 0

                # Eval
                with th.no_grad():
                    for batch in self.test_loader:
                        batch = move_to((batch.data, sender_id, self.eval_receiver_id), self.device)
                        agent_receiver = self.population.agents[self.eval_receiver_id]
                        metrics = self.game(batch, compute_metrics=True)

                        mean_val_acc += metrics["accuracy"].detach().item()
                        mean_val_loss += agent_receiver.tasks[task]["loss_value"].item()
                        n_batch += 1

                val_accuracies.append(mean_val_acc / n_batch)
                val_losses.append(mean_val_loss / n_batch)

                step += 1

                if early_stopping:
                    continue_training = not (
                            len(train_losses) > 200 and (train_losses[-1] > np.mean(train_losses[-20:]) - 0.0001) \
                            or step == 1000)
                else:
                    continue_training = (step > n_step_train)

            train_acc = np.max(train_accuracies[-5:])
            top_val_acc = np.max(val_accuracies)
            train_loss = train_losses[-1]
            top_val_loss = np.min(val_losses)
            if len(np.where(np.array(train_accuracies) > 0.98)[0]):
                etl = np.min(np.where(np.array(train_accuracies) > 0.98)[0])
            else:
                etl = -1

            train_accs[i] = train_acc
            top_val_accs[i] = top_val_acc
            train_loss_all_senders[i] = train_loss
            val_loss_all_senders[i] = top_val_loss
            etls[i] = etl

        return train_accs, top_val_accs, etls, train_loss_all_senders, val_loss_all_senders

    def save_messages(self, save_dir):

        self.game.eval()

        with th.no_grad():

            messages_per_agents = []
            entropy_per_agents = []

            inputs = self.dump_batch[0]
            inputs = move_to(inputs, device=self.device)
            for agent_id in self.population.sender_names:
                inputs_embedding = self.population.agents[agent_id].encode_object(inputs)
                messages, _, entropy_sender = self.population.agents[agent_id].send(inputs_embedding)
                messages_per_agents.append(messages.cpu().numpy())
                entropy_per_agents.append(entropy_sender.cpu().numpy())

            for agent_id in self.population.untrained_sender_names:
                inputs_embedding = self.population.agents[agent_id].encode_object(inputs)
                messages, _, entropy_sender = self.population.agents[agent_id].send(inputs_embedding)
                messages_per_agents.append(messages.cpu().numpy())
                entropy_per_agents.append(entropy_sender.cpu().numpy())

            np.save(f"{save_dir}/messages.npy", np.stack(messages_per_agents))
            np.save(f"{save_dir}/entropy_messages.npy", np.stack(entropy_per_agents))

    def reward_decomposition(self):

        self.game.train()

        with th.no_grad():
            n_x, n_att, n_val = self.dump_batch[0].size(0), self.dump_batch[0].size(1), self.dump_batch[0].size(2)
            id_sender, id_receiver = self.dump_batch[1], self.dump_batch[2]

            inputs = move_to(self.dump_batch[0], self.device)
            inputs_embedding = self.game.population.agents[id_sender].encode_object(inputs)
            messages, _, _ = self.game.population.agents[id_sender].send(inputs_embedding)

            agent_receiver = self.game.population.agents[id_receiver]
            message_embedding = agent_receiver.receive(messages)
            output_receiver = agent_receiver.reconstruct_from_message_embedding(message_embedding)
            loss_receiver = agent_receiver.compute_receiver_loss(inputs=inputs,
                                                                 output_receiver=output_receiver)

            id_sampled_messages = np.arange(n_x)
            n_m = n_x
            # id_sampled_messages = np.random.choice(metrics["messages"].size(0), n_m)
            sampled_messages = messages[id_sampled_messages]
            sampled_messages = sampled_messages.unsqueeze(0)
            sampled_messages = sampled_messages.repeat(n_x, 1, 1)
            sampled_messages = sampled_messages.permute(1, 0, 2)
            sampled_messages = sampled_messages.reshape([n_x * n_m, sampled_messages.size(-1)])
            sampled_x = inputs.repeat(n_m, 1, 1, 1)
            sampled_x = sampled_x.reshape([n_m * n_x, n_att, n_val])

            sampled_x = move_to(sampled_x, self.device)
            sampled_messages = move_to(sampled_messages, self.device)
            log_probs = self.game.population.agents[id_sender].get_log_prob_m_given_x(sampled_x,
                                                                                      sampled_messages)

            # log_probs -> pm1_x1,...,pm1_xn ; pm2_x1,...,pm2_xn,.....
            message_lengths = find_lengths(sampled_messages)
            max_len = sampled_messages.size(1)
            mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
                                               num_classes=max_len + 1), dim=1)[:, :-1]
            log_probs = (log_probs * mask_eos).sum(dim=1)

            log_pi_m_x = log_probs.reshape([n_m, n_x])
            pi_m_x = th.exp(log_pi_m_x)
            p_x = th.ones(n_x) / n_x  # Here we set p(x)=1/batch_size
            p_x = p_x.to(pi_m_x.device)  # Fix device issue

            log_pi_m = th.log((pi_m_x * p_x).sum(1))

            f_pi_m = (th.log(pi_m_x.diagonal(0)) + th.log(p_x) - log_pi_m)
            pi_l = th.exp(-1 * loss_receiver)
            f_pi_l = th.log(pi_l)

            reward_total = f_pi_l.mean().item()
            reward_coordination = -(f_pi_m - f_pi_l).mean().item()
            reward_information = f_pi_m.mean().item()

            return reward_total, reward_information, reward_coordination

    def evaluate_topographic_similarity(self,
                                        n_pairs: int = 1000,
                                        dist_input: str = "common_attributes",
                                        dist_message: str = "edit_distance"):

        topographic_similarity = np.zeros(len(self.population.sender_names))

        inputs = move_to(self.dump_batch[0], self.device)
        inputs_np = self.dump_batch[0].numpy()

        self.game.train()

        with th.no_grad():

            for i, sender_id in enumerate(self.population.sender_names):
                inputs_embedding = self.population.agents[sender_id].encode_object(inputs)
                messages, _, _ = self.population.agents[sender_id].send(inputs_embedding)
                messages_len = find_lengths(messages)

                # Sample pairs of messages ; inputs
                idx = np.random.randint(0, inputs_np.shape[0], n_pairs * 2)
                inputs_pairs = inputs_np[idx].reshape((n_pairs, 2, -1))
                messages_pairs = messages.cpu().numpy()[idx].reshape((n_pairs, 2, -1))
                messages_len_pairs = messages_len.cpu().numpy()[idx].reshape((n_pairs, 2))

                if dist_input == "common_attributes":
                    distances_inputs = np.mean(1 - 1 * ((inputs_pairs[:, 0, :] - inputs_pairs[:, 1, :]) == 0), axis=1)
                else:
                    raise NotImplementedError

                if dist_message == "edit_distance":
                    distances_messages = 1 - compute_language_similarity(messages_1=messages_pairs[:, 0, :],
                                                                         messages_2=messages_pairs[:, 1, :],
                                                                         len_messages_1=messages_len_pairs[:, 0],
                                                                         len_messages_2=messages_len_pairs[:, 1])
                #elif dist_message == "common_symbols":
                #    distances_messages = np.mean(1 - 1 * ((inputs_pairs[:, 0, :] - inputs_pairs[:, 1, :]) == 0), axis=1)
                else:
                    raise NotImplementedError

                top_sim = spearmanr(distances_inputs, distances_messages).correlation

                topographic_similarity[i] = top_sim

        return topographic_similarity

    def evaluate_language_similarity(self,
                                     n_samples: int = 50,
                                     method: str = "edit_distance"):

        self.game.train()

        with th.no_grad():

            messages_per_agents = []
            message_lengths_per_agents = []

            inputs = self.dump_batch[0]
            inputs = inputs.repeat([n_samples] + [1] * (len(inputs.size()) - 1))
            inputs = move_to(inputs, device=self.device)
            for agent_id in self.population.sender_names:
                inputs_embedding = self.population.agents[agent_id].encode_object(inputs)
                messages, _, _ = self.population.agents[agent_id].send(inputs_embedding)
                message_lengths = find_lengths(messages)
                messages_per_agents.append(messages)
                message_lengths_per_agents.append(message_lengths)

            similarity_matrix = np.identity(len(self.population.sender_names))

            for i in range(len(messages_per_agents) - 1):
                for j in range(i + 1, len(messages_per_agents)):
                    if method == "edit_distance":
                        similarity_val = compute_language_similarity(messages_1=messages_per_agents[i],
                                                                     messages_2=messages_per_agents[j],
                                                                     len_messages_1=message_lengths_per_agents[i],
                                                                     len_messages_2=message_lengths_per_agents[
                                                                         j]).mean()

                    else:
                        raise Exception("Specify known method")

                    similarity_matrix[i, j] = similarity_val
                    similarity_matrix[j, i] = similarity_val

            return similarity_matrix

    def evaluate_distance_from_init_language(self, n_samples: int = 50):

        self.game.train()

        with th.no_grad():

            inputs = self.dump_batch[0]
            inputs = inputs.repeat([n_samples] + [1] * (len(inputs.size()) - 1))
            inputs = move_to(inputs, device=self.device)

            if self.init_languages is None:

                self.init_languages = []
                self.init_languages_lengths = []
                for agent_id in self.population.sender_names:
                    inputs_embedding = self.population.agents[agent_id].encode_object(inputs)
                    messages, _, _ = self.population.agents[agent_id].send(inputs_embedding)
                    message_lengths = find_lengths(messages)
                    self.init_languages.append(messages)
                    self.init_languages_lengths.append(message_lengths)

            messages_per_agents = []
            message_lengths_per_agents = []

            for agent_id in self.population.sender_names:
                inputs_embedding = self.population.agents[agent_id].encode_object(inputs)
                messages, _, _ = self.population.agents[agent_id].send(inputs_embedding)
                message_lengths = find_lengths(messages)
                messages_per_agents.append(messages)
                message_lengths_per_agents.append(message_lengths)

            similarity_to_init_languages = np.zeros((len(self.population.sender_names),
                                                     len(self.population.sender_names)))

            for i in range(len(messages_per_agents)):
                for j in range(len(self.init_languages)):
                    similarity_val = compute_language_similarity(messages_1=messages_per_agents[i],
                                                                 messages_2=self.init_languages[j],
                                                                 len_messages_1=message_lengths_per_agents[i],
                                                                 len_messages_2=self.init_languages_lengths[j]).mean()
                    similarity_to_init_languages[i, j] = similarity_val

            return similarity_to_init_languages

    def divergence_to_untrained_speakers(self, n_samples: int = 50):

        self.game.train()

        with th.no_grad():
            inputs = self.dump_batch[0]
            inputs = inputs.repeat([n_samples] + [1] * (len(inputs.size()) - 1))
            inputs = move_to(inputs, device=self.device)

            divergence_matrix = np.zeros((len(self.population.sender_names),
                                          len(self.population.untrained_sender_names)))

            for i, agent_id_p in enumerate(self.population.sender_names):
                inputs_embedding = self.population.agents[agent_id_p].encode_object(inputs)
                messages_1, log_prob_1, _ = self.population.agents[agent_id_p].send(inputs_embedding)
                message_lengths_1 = find_lengths(messages_1)
                max_len = messages_1.size(1)
                mask_eos = 1 - th.cumsum(F.one_hot(message_lengths_1.to(th.int64),
                                                   num_classes=max_len + 1), dim=1)[:, :-1]
                log_prob_1 = (log_prob_1 * mask_eos).sum(dim=1)

                for j, agent_id_q in enumerate(self.population.untrained_sender_names):
                    log_prob_2 = self.game.population.agents[agent_id_q].get_log_prob_m_given_x(inputs,
                                                                                                messages_1)

                    log_prob_2 = (log_prob_2 * mask_eos).sum(dim=1)

                    divergence_matrix[i, j] = th.sum(
                        th.exp(log_prob_1) * (log_prob_1 - log_prob_2)).item() / inputs.size(0)

        return divergence_matrix

    def accuracy_with_untrained_speakers(self, n_samples=10):

        self.game.train()

        accuracy_with_untrained_speakers = np.zeros((len(self.population.receiver_names),
                                                     len(self.population.untrained_sender_names)))

        with th.no_grad():
            inputs = self.dump_batch[0]
            inputs = inputs.repeat([n_samples] + [1] * (len(inputs.size()) - 1))
            inputs = move_to(inputs, device=self.device)

            for i, receiver_id in enumerate(self.population.receiver_names):
                for j, untrained_sender_id in enumerate(self.population.untrained_sender_names):
                    _, _, metrics = self.game((inputs, untrained_sender_id, receiver_id), compute_metrics=True)

                    accuracy_with_untrained_speakers[i, j] = metrics["accuracy"]

        return accuracy_with_untrained_speakers

    def accuracy_with_untrained_listeners(self, n_samples=10):

        self.game.train()

        accuracy_with_untrained_listeners = np.zeros((len(self.population.sender_names),
                                                      len(self.population.untrained_receiver_names)))

        with th.no_grad():
            inputs = self.dump_batch[0]
            inputs = inputs.repeat([n_samples] + [1] * (len(inputs.size()) - 1))
            inputs = move_to(inputs, device=self.device)

            for i, sender_id in enumerate(self.population.sender_names):
                for j, untrained_receiver_id in enumerate(self.population.untrained_receiver_names):
                    _, _, metrics = self.game((inputs, sender_id, untrained_receiver_id), compute_metrics=True)

                    accuracy_with_untrained_listeners[i, j] = metrics["accuracy"]

        return accuracy_with_untrained_listeners

    def log_metrics(self,
                    iter: int) -> None:

        # Reward decomposition
        if self.metrics_to_measure["reward_decomposition"]<self.n_epochs:
            id_sender = 0
            self.writer.add_scalar(f'{id_sender}/Reward information',
                                   self.stored_metrics["reward_information"][-1], iter)
            self.writer.add_scalar(f'{id_sender}/Reward coordination',
                                   self.stored_metrics["reward_coordination"][-1], iter)
            self.writer.add_scalar(f'{id_sender}/Reward total',
                                   self.stored_metrics["reward"][-1], iter)

        # Language similarity
        if self.metrics_to_measure["language_similarity"]<self.n_epochs:
            similarity_matrix = self.stored_metrics["language_similarity"][-1]
            print(similarity_matrix)
            unique_sim = [similarity_matrix[i, j] for i in range(len(similarity_matrix) - 1) \
                          for j in range(i + 1, len(similarity_matrix))]
            self.writer.add_scalar(f'average_similarity', np.mean(unique_sim), iter)

        # Similarity to init language
        if self.metrics_to_measure["similarity_to_init_languages"]<self.n_epochs:
            similarity_to_init_languages = self.stored_metrics["similarity_to_init_languages"][-1]
            for i, agent_id in enumerate(self.population.sender_names):
                for j, agent_id_init in enumerate(self.population.sender_names):
                    self.writer.add_scalar(f'{agent_id}/similarity_to_{agent_id_init}',
                                           similarity_to_init_languages[i, j],
                                           iter)

        # Divergence to untrained speakers
        if self.metrics_to_measure["divergence_to_untrained_speakers"]<self.n_epochs:
            divergence_matrix = self.stored_metrics["divergence_to_untrained_speakers"][-1]
            for i, agent_id in enumerate(self.population.sender_names):
                for j, agent_id_untrained in enumerate(self.population.untrained_sender_names):
                    self.writer.add_scalar(f'{agent_id}/div_to_{agent_id_untrained}',
                                           divergence_matrix[i, j],
                                           iter)

                    self.writer.add_scalar(f'div/div_to_{agent_id_untrained}',
                                           divergence_matrix[i, j],
                                           iter)

        if self.metrics_to_measure["accuracy_with_untrained_speakers"]<self.n_epochs:
            accuracy_matrix = self.stored_metrics["accuracy_with_untrained_speakers"][-1]
            for i, receiver_id in enumerate(self.population.receiver_names):
                for j, untrained_sender_id in enumerate(self.population.untrained_sender_names):
                    self.writer.add_scalar(f'{receiver_id}/Accuracy with {untrained_sender_id}',
                                           accuracy_matrix[i, j],
                                           iter)

        if self.metrics_to_measure["accuracy_with_untrained_listeners"]<self.n_epochs:
            accuracy_matrix = self.stored_metrics["accuracy_with_untrained_listeners"][-1]
            for i, sender_id in enumerate(self.population.sender_names):
                for j, untrained_receiver_id in enumerate(self.population.untrained_receiver_names):
                    self.writer.add_scalar(f'{sender_id}/Accuracy with {untrained_receiver_id}',
                                           accuracy_matrix[i, j],
                                           iter)

        if self.metrics_to_measure["topographic_similarity"]<self.n_epochs:
            topographic_similarity = self.stored_metrics["topographic_similarity"][-1]
            for i, sender_id in enumerate(self.population.sender_names):
                self.writer.add_scalar(f'{sender_id}/Topographic similarity',
                                       topographic_similarity[i],
                                       iter)

        if self.metrics_to_measure["external_receiver_evaluation"]<self.n_epochs:
            if len(self.stored_metrics["external_receiver_train_acc"]):
                train_acc = self.stored_metrics["external_receiver_train_acc"][-1]
                top_val_acc = self.stored_metrics["external_receiver_val_acc"][-1]
                train_loss =  self.stored_metrics["external_receiver_train_loss"][-1]
                top_val_loss = self.stored_metrics["external_receiver_val_loss"][-1]
                etl = self.stored_metrics["etl"][-1]
                for i, sender_id in enumerate(self.population.sender_names):
                    self.writer.add_scalar(f'{sender_id}/Train acc external receiver',
                                           train_acc[i],
                                           iter)
                    self.writer.add_scalar(f'{sender_id}/Top val acc external receiver',
                                           top_val_acc[i],
                                           iter)
                    self.writer.add_scalar(f'{sender_id}/Train loss external receiver',
                                           train_loss[i],
                                           iter)
                    self.writer.add_scalar(f'{sender_id}/Top val loss external receiver',
                                           top_val_loss[i],
                                           iter)
                    self.writer.add_scalar(f'{sender_id}/ETL',
                                           etl[i],
                                           iter)

        if self.metrics_to_measure["MI"]<self.n_epochs:
            for sender in self.stored_metrics["MI"]:
                mi_values = self.stored_metrics["MI"][sender][-1]
                for dataset_type in mi_values:
                    mi_value = mi_values[dataset_type]
                    self.writer.add_scalar(f'{sender_id}/MI_{dataset_type}',
                                           mi_value,
                                           iter)

    def save_metrics(self, save_dir):

        if self.metrics_to_measure["reward_decomposition"]<self.n_epochs:
            np.save(f"{save_dir}/reward_total.npy",
                    self.stored_metrics["reward"])
            np.save(f"{save_dir}/reward_information.npy",
                    self.stored_metrics["reward_information"])
            np.save(f"{save_dir}/reward_coordination.npy",
                    self.stored_metrics["reward_coordination"])
        if self.metrics_to_measure["language_similarity"]<self.n_epochs:
            np.save(f"{save_dir}/language_similarity.npy",
                    np.stack(self.stored_metrics["language_similarity"]))
        if self.metrics_to_measure["similarity_to_init_languages"]<self.n_epochs:
            np.save(f"{save_dir}/language_similarity_to_init.npy",
                    np.stack(self.stored_metrics["similarity_to_init_languages"]))
        if self.metrics_to_measure["divergence_to_untrained_speakers"]<self.n_epochs:
            np.save(f"{save_dir}/divergence_to_untrained_speakers.npy",
                    np.stack(self.stored_metrics["divergence_to_untrained_speakers"]))
        if self.metrics_to_measure["accuracy_with_untrained_listeners"]<self.n_epochs:
            np.save(f"{save_dir}/accuracy_with_untrained_listeners.npy",
                    np.stack(self.stored_metrics["accuracy_with_untrained_listeners"]))
        if self.metrics_to_measure["accuracy_with_untrained_speakers"]<self.n_epochs:
            np.save(f"{save_dir}/accuracy_with_untrained_speakers.npy",
                    np.stack(self.stored_metrics["accuracy_with_untrained_speakers"]))
        if self.metrics_to_measure["topographic_similarity"]<self.n_epochs:
            np.save(f"{save_dir}/topographic_similarity.npy",
                    np.stack(self.stored_metrics["topographic_similarity"]))
        if self.metrics_to_measure["external_receiver_evaluation"]<self.n_epochs:
            np.save(f"{save_dir}/external_receiver_train_acc.npy",
                    np.stack(self.stored_metrics["external_receiver_train_acc"]))
            np.save(f"{save_dir}/external_receiver_val_acc.npy",
                    np.stack(self.stored_metrics["external_receiver_val_acc"]))
            np.save(f"{save_dir}/etl.npy",
                    np.stack(self.stored_metrics["etl"]))


def build_evaluator(metrics_to_measure,
                    game,
                    train_loader,
                    val_loader,
                    test_loader,
                    mi_loader,
                    agent_repertory,
                    game_params,
                    eval_receiver_id,
                    n_epochs,
                    dump_batch = None,
                    logger: th.utils.tensorboard.SummaryWriter = None,
                    device: str = "cpu"):
    evaluator = Evaluator(metrics_to_measure=metrics_to_measure,
                          n_epochs=n_epochs,
                          game=game,
                          logger=logger,
                          dump_batch=dump_batch,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          test_loader=test_loader,
                          mi_loader=mi_loader,
                          agent_repertory=agent_repertory,
                          game_params=game_params,
                          eval_receiver_id=eval_receiver_id,
                          device=device)

    return evaluator
