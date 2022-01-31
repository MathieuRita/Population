import torch as th
import numpy as np
from .utils import move_to, find_lengths
import torch.nn.functional as F
from .language_metrics import compute_language_similarity


class Evaluator:

    def __init__(self,
                 metrics_to_measure,
                 game,
                 dump_batch,
                 logger: th.utils.tensorboard.SummaryWriter = None,
                 device: str = "cpu") -> None:

        self.game = game
        self.population = game.population
        self.dump_batch = dump_batch
        self.device = th.device(device)
        self.writer = logger
        self.init_languages = None

        # Metrics init
        self.metrics_to_measure = metrics_to_measure
        self.stored_metrics = {}

        if self.metrics_to_measure["reward_decomposition"]:
            self.stored_metrics["reward"] = list()
            self.stored_metrics["reward_coordination"] = list()
            self.stored_metrics["reward_information"] = list()
        if self.metrics_to_measure["language_similarity"]:
            self.stored_metrics["language_similarity"] = list()
        if self.metrics_to_measure["similarity_to_init_languages"]:
            self.stored_metrics["similarity_to_init_languages"] = list()
        if self.metrics_to_measure["divergence_to_untrained_speakers"]:
            self.stored_metrics["divergence_to_untrained_speakers"] = list()
        if self.metrics_to_measure["accuracy_with_untrained_speakers"]:
            self.stored_metrics["accuracy_with_untrained_speakers"] = list()
        if self.metrics_to_measure["accuracy_with_untrained_listeners"]:
            self.stored_metrics["accuracy_with_untrained_listeners"] = list()

    def step(self,
             epoch: int) -> None:

        if self.metrics_to_measure["reward_decomposition"]:
            reward_total, reward_information, reward_coordination = self.reward_decomposition()
            self.stored_metrics["reward"].append(reward_total)
            self.stored_metrics["reward_coordination"].append(reward_coordination)
            self.stored_metrics["reward_information"].append(reward_information)

        if self.metrics_to_measure["language_similarity"]:
            similarity_matrix = self.evaluate_language_similarity(n_samples=10)
            self.stored_metrics["language_similarity"].append(similarity_matrix)

        if self.metrics_to_measure["similarity_to_init_languages"]:
            similarity_to_init_languages = self.evaluate_distance_from_init_language(n_samples=10)
            self.stored_metrics["similarity_to_init_languages"].append(similarity_to_init_languages)

        if self.metrics_to_measure["divergence_to_untrained_speakers"]:
            divergence_matrix = self.divergence_to_untrained_speakers(n_samples=50)
            self.stored_metrics["divergence_to_untrained_speakers"].append(divergence_matrix)

        if self.metrics_to_measure["accuracy_with_untrained_speakers"]:
            accuracy_with_untrained_speakers = self.accuracy_with_untrained_speakers()
            self.stored_metrics["accuracy_with_untrained_speakers"].append(accuracy_with_untrained_speakers)

        if self.metrics_to_measure["accuracy_with_untrained_listeners"]:
            accuracy_with_untrained_listeners = self.accuracy_with_untrained_listeners()
            self.stored_metrics["accuracy_with_untrained_listeners"].append(accuracy_with_untrained_listeners)

        if self.writer is not None:
            self.log_metrics(iter=epoch)

    def save_messages(self,save_dir):

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
            
            np.save(f"{save_dir}/messages.npy",np.stack(messages_per_agents))
            np.save(f"{save_dir}/entropy_messages.npy",np.stack(entropy_per_agents))

    def reward_decomposition(self):

        self.game.train()

        with th.no_grad():
            n_x, n_att, n_val = self.dump_batch[0].size(0), self.dump_batch[0].size(1), self.dump_batch[0].size(2)
            id_sender = self.dump_batch[1]
            batch = move_to(self.dump_batch, self.device)
            _, loss_receiver, metrics = self.game(batch, compute_metrics=True)

            id_sampled_messages = np.arange(n_x)
            n_m = n_x
            # id_sampled_messages = np.random.choice(metrics["messages"].size(0), n_m)
            sampled_messages = metrics["messages"][id_sampled_messages]
            sampled_messages = sampled_messages.unsqueeze(0)
            sampled_messages = sampled_messages.repeat(n_x, 1, 1)
            sampled_messages = sampled_messages.permute(1, 0, 2)
            sampled_messages = sampled_messages.reshape([n_x * n_m, sampled_messages.size(-1)])
            sampled_x = batch[0].repeat(n_m, 1, 1, 1)
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
            p_x = p_x.to(pi_m_x.device) # Fix device issue

            log_pi_m = th.log((pi_m_x * p_x).sum(1))

            f_pi_m = (th.log(pi_m_x.diagonal(0)) + th.log(p_x) - log_pi_m)
            pi_l = th.exp(-1 * loss_receiver)
            f_pi_l = th.log(pi_l)

            reward_total = f_pi_l.mean().item()
            reward_coordination = -(f_pi_m - f_pi_l).mean().item()
            reward_information = f_pi_m.mean().item()

            return reward_total, reward_information, reward_coordination

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
        if self.metrics_to_measure["reward_decomposition"]:
            id_sender = 0
            self.writer.add_scalar(f'{id_sender}/Reward information',
                                   self.stored_metrics["reward_information"][-1], iter)
            self.writer.add_scalar(f'{id_sender}/Reward coordination',
                                   self.stored_metrics["reward_coordination"][-1], iter)
            self.writer.add_scalar(f'{id_sender}/Reward total',
                                   self.stored_metrics["reward"][-1], iter)

        # Language similarity
        if self.metrics_to_measure["language_similarity"]:
            similarity_matrix = self.stored_metrics["similarity_matrix"][-1]
            unique_sim = [similarity_matrix[i, j] for i in range(len(similarity_matrix) - 1) \
                          for j in range(i + 1, len(similarity_matrix))]
            self.writer.add_scalar(f'average_similarity', np.mean(unique_sim), iter)

        # Similarity to init language
        if self.metrics_to_measure["similarity_to_init_languages"]:
            similarity_to_init_languages = self.stored_metrics["similarity_to_init_languages"][-1]
            for i, agent_id in enumerate(self.population.sender_names):
                for j, agent_id_init in enumerate(self.population.sender_names):
                    self.writer.add_scalar(f'{agent_id}/similarity_to_{agent_id_init}',
                                           similarity_to_init_languages[i, j],
                                           iter)

        # Divergence to untrained speakers
        if self.metrics_to_measure["divergence_to_untrained_speakers"]:
            divergence_matrix = self.stored_metrics["divergence_to_untrained_speakers"][-1]
            for i, agent_id in enumerate(self.population.sender_names):
                for j, agent_id_untrained in enumerate(self.population.untrained_sender_names):
                    self.writer.add_scalar(f'{agent_id}/div_to_{agent_id_untrained}',
                                           divergence_matrix[i, j],
                                           iter)

                    self.writer.add_scalar(f'div/div_to_{agent_id_untrained}',
                                           divergence_matrix[i, j],
                                           iter)

        if self.metrics_to_measure["accuracy_with_untrained_speakers"]:
            accuracy_matrix = self.stored_metrics["accuracy_with_untrained_speakers"][-1]
            for i, receiver_id in enumerate(self.population.receiver_names):
                for j, untrained_sender_id in enumerate(self.population.untrained_sender_names):
                    self.writer.add_scalar(f'{receiver_id}/Accurcy with {untrained_sender_id}',
                                           accuracy_matrix[i, j],
                                           iter)

        if self.metrics_to_measure["accuracy_with_untrained_listeners"]:
            accuracy_matrix = self.stored_metrics["accuracy_with_untrained_listeners"][-1]
            for i, sender_id in enumerate(self.population.sender_names):
                for j, untrained_receiver_id in enumerate(self.population.untrained_receiver_names):
                    self.writer.add_scalar(f'{sender_id}/Accuracy with {untrained_receiver_id}',
                                           accuracy_matrix[i, j],
                                           iter)

    def save_metrics(self, save_dir):

        if self.metrics_to_measure["reward_decomposition"]:
            np.save(f"{save_dir}/reward_total.npy",
                    self.stored_metrics["reward"])
            np.save(f"{save_dir}/reward_information.npy",
                    self.stored_metrics["reward_information"])
            np.save(f"{save_dir}/reward_coordination.npy",
                    self.stored_metrics["reward_coordination"])
        if self.metrics_to_measure["language_similarity"]:
            np.save(f"{save_dir}/language_similarity.npy",
                    np.stack(self.stored_metrics["language_similarity"]))
        if self.metrics_to_measure["similarity_to_init_languages"]:
            np.save(f"{save_dir}/language_similarity_to_init.npy",
                    np.stack(self.stored_metrics["similarity_to_init_languages"]))
        if self.metrics_to_measure["divergence_to_untrained_speakers"]:
            np.save(f"{save_dir}/divergence_to_untrained_speakers.npy",
                    np.stack(self.stored_metrics["divergence_to_untrained_speakers"]))
        if self.metrics_to_measure["accuracy_with_untrained_listeners"]:
            np.save(f"{save_dir}/accuracy_with_untrained_listeners.npy",
                    np.stack(self.stored_metrics["accuracy_with_untrained_listeners"]))
        if self.metrics_to_measure["accuracy_with_untrained_speakers"]:
            np.save(f"{save_dir}/accuracy_with_untrained_speakers.npy",
                    np.stack(self.stored_metrics["accuracy_with_untrained_speakers"]))


def build_evaluator(metrics_to_measure,
                    game,
                    dump_batch,
                    logger: th.utils.tensorboard.SummaryWriter = None,
                    device: str = "cpu"):
    evaluator = Evaluator(metrics_to_measure=metrics_to_measure,
                          game=game,
                          logger=logger,
                          dump_batch=dump_batch,
                          device=device)

    return evaluator