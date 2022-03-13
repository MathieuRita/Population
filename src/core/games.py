import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .losses import accuracy, find_lengths, cross_entropy_imitation


class ReconstructionGame(nn.Module):

    def __init__(self,
                 population: object,
                 voc_size : int,
                 max_len : int,
                 noise_level : float = 0.):

        super(ReconstructionGame, self).__init__()
        self.population = population
        self.voc_size=voc_size
        self.max_len = max_len
        self.noise_level = noise_level

    def communication_instance(self,
                               inputs: th.Tensor,
                               sender_id: th.Tensor,
                               receiver_id: th.Tensor,
                               compute_metrics: bool = False):

        """
        :param noise_threshold: 0 -> no noise
        :param compute_metrics:
        :param receiver_id:
        :param sender_id:
        :param inputs:
        :param x: tuple (sender_id,receiver_id,batch)
        :return: (loss_sender,loss_receiver) [batch_size,batch_size]
        """

        agent_sender = self.population.agents[sender_id]
        agent_receiver = self.population.agents[receiver_id]

        # Agent Sender sends message based on input
        inputs_embedding = agent_sender.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = agent_sender.send(inputs_embedding)

        # Noise in communication channel
        random_messages = th.randint(low=1, high=self.voc_size, size=messages.size(),device=messages.device)
        probs = th.rand(messages.size(), device=messages.device)
        mask = 1 * (probs < self.noise_level)
        messages = (1 - mask) * messages + mask * random_messages

        # Agent receiver encodes message and predict the reconstructed object
        message_embedding = agent_receiver.receive(messages)
        output_receiver = agent_receiver.reconstruct_from_message_embedding(message_embedding)

        task = "communication"

        reward = agent_sender.tasks[task]["loss"].reward_fn(inputs=inputs,
                                                            receiver_output=output_receiver).detach()
        loss = agent_sender.tasks[task]["loss"].compute(reward=reward,
                                                        sender_log_prob=log_prob_sender,
                                                        sender_entropy=entropy_sender,
                                                        message=messages
                                                        )
        agent_sender.tasks[task]["loss_value"] = loss.mean()

        loss = agent_receiver.tasks[task]["loss"].compute(inputs=inputs,
                                                          receiver_output=output_receiver)

        agent_receiver.tasks[task]["loss_value"] = loss.mean()

        # Compute additional metrics
        metrics = {}

        if compute_metrics:
            # accuracy
            metrics["accuracy"] = accuracy(inputs, output_receiver, game_mode="reconstruction").mean()
            # Sender entropy
            metrics["sender_entropy"] = entropy_sender.mean().item()
            # Sender log prob
            metrics["sender_log_prob"] = log_prob_sender.detach()
            # Raw messages
            metrics["messages"] = messages
            # Average message length
            metrics["message_length"] = find_lengths(messages).float().mean().item()
            # Output receiver
            # metrics["receiver_output"] = output_receiver.detach()

        return metrics

    def forward(self, batch, compute_metrics: bool = False, random_messages : bool = False):

        metrics = self.communication_instance(*batch, compute_metrics=compute_metrics, random_messages=random_messages)

        return metrics

    def communication_multi_listener_instance(self,
                                              inputs: th.Tensor,
                                              sender_id: th.Tensor,
                                              receiver_ids: list,
                                              weight_receivers: dict,
                                              reward_noise : bool = False,
                                              compute_metrics: bool = False):

        """
        :param compute_metrics:
        :param receiver_id:
        :param sender_id:
        :param inputs:
        :param x: tuple (sender_id,receiver_id,batch)
        :return: (loss_sender,loss_receiver) [batch_size,batch_size]
        """

        agent_sender = self.population.agents[sender_id]
        accuracies = {}

        # Agent Sender sends message based on input
        inputs_embedding = agent_sender.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = agent_sender.send(inputs_embedding)

        task = "communication"

        average_reward = th.zeros((messages.size(0)), device=messages.device)

        # Agent receiver encodes message and predict the reconstructed object
        for receiver_id in receiver_ids:
            agent_receiver = self.population.agents[receiver_id]
            message_embedding = agent_receiver.receive(messages)
            output_receiver = agent_receiver.reconstruct_from_message_embedding(message_embedding)
            accuracies[receiver_id] = accuracy(inputs, output_receiver, game_mode="reconstruction").mean()

            reward = agent_sender.tasks[task]["loss"].reward_fn(inputs=inputs,
                                                                receiver_output=output_receiver).detach()

            loss_receiver = agent_receiver.tasks[task]["loss"].compute(inputs=inputs,
                                                                       receiver_output=output_receiver)

            agent_receiver.tasks[task]["loss_value"] = loss_receiver.mean()

            average_reward += weight_receivers[receiver_id] * reward

        average_reward /= sum([v for _, v in weight_receivers.items()])

        if reward_noise:
            average_reward-=2+2*th.normal(th.zeros(average_reward.size(0))).to(average_reward.device)
            #average_reward=average_reward - 1.

        loss_sender = agent_sender.tasks[task]["loss"].compute(reward=average_reward,
                                                               sender_log_prob=log_prob_sender,
                                                               sender_entropy=entropy_sender,
                                                               message=messages
                                                               )

        agent_sender.tasks[task]["loss_value"] = loss_sender.mean()

        # Compute additional metrics
        metrics = {}

        if compute_metrics:
            # accuracy
            metrics["accuracy"] = accuracies
            # Sender entropy
            metrics["sender_entropy"] = entropy_sender.mean().item()
            # Sender log prob
            metrics["sender_log_prob"] = log_prob_sender.detach()
            # Raw messages
            metrics["messages"] = messages
            # Average message length
            metrics["message_length"] = find_lengths(messages).float().mean().item()
            # Output receiver
            # metrics["receiver_output"] = output_receiver.detach()

        return metrics

    def imitation_instance(self,
                           inputs: th.Tensor,
                           sender_id: str,
                           imitator_id: str):
        task = "imitation"

        agent_sender = self.population.agents[sender_id]
        agent_imitator = self.population.agents[imitator_id]

        # Agent Sender sends message based on input
        inputs_embedding = agent_sender.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = agent_sender.send(inputs_embedding)

        # Imitator tries to imitate messages
        inputs_imitation = (1 - agent_imitator.tasks[task]["lm_mode"]) * inputs

        _, log_probs_imitation = agent_imitator.get_log_prob_m_given_x(inputs_imitation,
                                                                       messages,
                                                                       return_whole_log_probs=True)
        log_imitation = -1 * cross_entropy_imitation(sender_log_prob=log_probs_imitation,
                                                     target_messages=messages)

        # Imitator

        loss = agent_imitator.tasks[task]["loss"].compute(sender_log_prob=log_probs_imitation,
                                                          target_messages=messages)

        agent_imitator.tasks[task]["loss_value"] = loss.mean()

        # Sender
        message_lengths = find_lengths(messages)
        max_len = messages.size(1)
        mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
                                           num_classes=max_len + 1), dim=1)[:, :-1]
        reward = (log_imitation - (log_prob_sender * mask_eos).sum(dim=1)).detach()

        loss = agent_sender.tasks[task]["loss"].compute(reward=reward,
                                                        sender_log_prob=log_prob_sender,
                                                        sender_entropy=entropy_sender,
                                                        message=messages)

        agent_sender.tasks[task]["loss_value"] = loss.mean()

    def communication_and_kl_instance(self,
                                      inputs: th.Tensor,
                                      sender_id: th.Tensor,
                                      receiver_id: list,
                                      weights: dict,
                                      compute_metrics: bool = False):

        """
        :param compute_metrics:
        :param receiver_id:
        :param sender_id:
        :param inputs:
        :param x: tuple (sender_id,receiver_id,batch)
        :return: (loss_sender,loss_receiver) [batch_size,batch_size]
        """

        agent_sender = self.population.agents[sender_id]
        agent_receiver = self.population.agents[receiver_id]

        # Agent Sender sends message based on input
        inputs_embedding = agent_sender.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = agent_sender.send(inputs_embedding)

        prob_lm = agent_sender.language_model.get_prob_messages(messages)

        message_lengths = find_lengths(messages)
        max_len = messages.size(1)
        mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
                                           num_classes=max_len + 1), dim=1)[:, :-1]

        # Agent receiver encodes message and predict the reconstructed object
        message_embedding = agent_receiver.receive(messages)
        output_receiver = agent_receiver.reconstruct_from_message_embedding(message_embedding)

        task = "communication"

        loss = agent_receiver.tasks[task]["loss"].compute(inputs=inputs,
                                                          receiver_output=output_receiver)

        agent_receiver.tasks[task]["loss_value"] = loss.mean()

        reward_communication = agent_sender.tasks[task]["loss"].reward_fn(inputs=inputs,
                                                                          receiver_output=output_receiver).detach()

        p_x = th.Tensor([1 / (6**6)]).to("cuda")

        reward_kl = th.log(prob_lm).detach() + reward_communication \
                    - th.log(p_x) - (log_prob_sender.detach() * mask_eos.detach()).sum(dim=1)

        reward = weights["communication"]*reward_communication + weights["KL"]*reward_kl
        reward /= sum([v for _, v in weights.items()])

        loss = agent_sender.tasks[task]["loss"].compute(reward=reward,
                                                        sender_log_prob=log_prob_sender,
                                                        sender_entropy=entropy_sender,
                                                        message=messages
                                                        )
        agent_sender.tasks[task]["loss_value"] = loss.mean()

        # Compute additional metrics
        metrics = {}

        if compute_metrics:
            # accuracy
            metrics["accuracy"] = accuracy(inputs, output_receiver, game_mode="reconstruction").mean()
            # Sender entropy
            metrics["sender_entropy"] = entropy_sender.mean().item()
            # Sender log prob
            metrics["sender_log_prob"] = log_prob_sender.detach()
            # Raw messages
            metrics["messages"] = messages
            # Average message length
            metrics["message_length"] = find_lengths(messages).float().mean().item()
            # Output receiver
            # metrics["receiver_output"] = output_receiver.detach()

        return metrics

    def mi_instance(self,
                    inputs: th.Tensor,
                    sender_id: str):

        task = "mutual_information"

        agent_sender = self.population.agents[sender_id]

        # Agent Sender sends message based on input
        inputs_embedding = agent_sender.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = agent_sender.send(inputs_embedding)

        messages_lm = [messages]
        for _ in range(20):
            messages_bis, _, _ = agent_sender.send(inputs_embedding)
            messages_lm.append(messages_bis)
        messages_lm = th.stack(messages_lm).view(-1, messages_lm[0].size(1))
        agent_sender.train_language_model(messages_lm)

        prob_lm = agent_sender.language_model.get_prob_messages(messages)

        # Sender
        p_x = th.Tensor([1 / 256]).to("cuda")
        message_lengths = find_lengths(messages)
        max_len = messages.size(1)
        mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
                                           num_classes=max_len + 1), dim=1)[:, :-1]

        reward = th.log(p_x) + (log_prob_sender.detach() * mask_eos.detach()).sum(dim=1) - th.log(prob_lm).detach()

        loss = agent_sender.tasks[task]["loss"].compute(reward=reward.detach(),
                                                        sender_log_prob=log_prob_sender,
                                                        sender_entropy=entropy_sender,
                                                        message=messages)

        agent_sender.tasks[task]["loss_value"] = loss.mean()

        return reward

    def direct_mi_instance(self,
                           inputs: th.Tensor,
                           sender_id: str,
                           imitator_id: str):

        task = "mutual_information"

        agent_sender = self.population.agents[sender_id]
        agent_imitator = self.population.agents[imitator_id]

        # Agent Sender sends message based on input
        inputs_embedding = agent_sender.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = agent_sender.send(inputs_embedding)

        # Imitator tries to imitate messages
        inputs_imitation = 0. * inputs
        # Concat starting token
        voc_size = 10
        start_token = th.Tensor(messages.size(0) * [voc_size]).to(int).to(messages.device)
        start_token = start_token.unsqueeze(1)
        messages_imit = th.cat((start_token, messages), dim=1)
        log_probs_imitation, _ = agent_imitator.get_log_prob_m_given_x(inputs_imitation,
                                                                       messages_imit,
                                                                       return_whole_log_probs=True)

        log_probs_imitation = log_probs_imitation[:, 1:]
        # log_imitation = -1 * cross_entropy_imitation(sender_log_prob=log_probs_imitation,
        #                                             target_messages=messages)

        # Imitator
        ## Nothing

        # Sender
        p_x = th.Tensor([1 / 256]).to("cuda")
        message_lengths = find_lengths(messages)
        max_len = messages.size(1)
        mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
                                           num_classes=max_len + 1), dim=1)[:, :-1]
        log_prob_sender = log_prob_sender * mask_eos.detach()
        log_probs_imitation = log_probs_imitation * mask_eos.detach()
        reward = th.log(p_x) + (log_prob_sender.sum(dim=1).detach() - log_probs_imitation.sum(dim=1).detach())

        loss = agent_sender.tasks[task]["loss"].compute(reward=reward,
                                                        sender_log_prob=log_prob_sender,
                                                        sender_entropy=entropy_sender,
                                                        message=messages)

        agent_sender.tasks[task]["loss_value"] = loss.mean()

        return reward


class ReferentialGame(nn.Module):

    def __init__(self,
                 population: object):
        super(ReferentialGame, self).__init__()
        self.population = population

    def game_instance(self,
                      inputs: th.Tensor,
                      distractors: th.Tensor,
                      sender_id: th.Tensor,
                      receiver_id: th.Tensor,
                      compute_metrics: bool = False):
        """
        :param x: tuple (sender_id,receiver_id,batch)
        :return: (loss_sender,loss_receiver) [batch_size,batch_size]
        """

        agent_sender = self.population.agents[sender_id]
        agent_receiver = self.population.agents[receiver_id]

        # Dimensions
        batch_size, n_distractors, dim_obj_1, dim_obj_2 = distractors.size(0), distractors.size(1), \
                                                          distractors.size(2), distractors.size(3)

        # Agent sender sends message based on object
        inputs_encoded = agent_sender.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = agent_sender.send(inputs_encoded)

        # Agent receiver encodes messages and distractors and predicts input objects
        message_embedding = agent_receiver.receive(messages)

        distractors = distractors.reshape((batch_size * n_distractors, dim_obj_1, dim_obj_2))
        distractors_embedding = agent_receiver.encode_object(distractors)
        distractors_embedding = distractors_embedding.reshape(
            (batch_size, n_distractors, distractors_embedding.size(-1)))
        inputs_embedding = agent_receiver.encode_object(inputs)
        output_receiver = agent_receiver.predict_referential_candidate(message_embedding=message_embedding,
                                                                       inputs_embedding=inputs_embedding,
                                                                       distractors_embedding=distractors_embedding)
        # [batch_size,1+n_distractors]

        loss_sender = agent_sender.compute_sender_loss(inputs=inputs,
                                                       sender_log_prob=log_prob_sender,
                                                       sender_entropy=entropy_sender,
                                                       messages=messages,
                                                       receiver_output=output_receiver)  # [batch_size]

        loss_receiver = agent_receiver.compute_receiver_loss(inputs=inputs,
                                                             output_receiver=output_receiver)  # [batch_size]

        # Compute additional metrics
        metrics = {}

        if compute_metrics:
            # accuracy
            metrics["accuracy"] = accuracy(inputs, output_receiver, game_mode="referential").mean()
            # Sender entropy
            metrics["sender_entropy"] = entropy_sender.mean().item()
            # Raw messages
            metrics["messages"] = messages
            # Average message length
            metrics["message_length"] = find_lengths(messages).float().mean().item()

        return loss_sender, loss_receiver.mean(), metrics

    def forward(self, batch, compute_metrics: bool = False):
        loss_sender, loss_receiver, metrics = self.game_instance(*batch, compute_metrics=compute_metrics)

        return loss_sender, loss_receiver, metrics


class ReconstructionImitationGame(nn.Module):

    def __init__(self,
                 population: object):
        super(ReconstructionImitationGame, self).__init__()
        self.population = population

    def game_instance(self,
                      inputs: th.Tensor,
                      sender_id: str,
                      receiver_id: str,
                      imitator_id: str,
                      compute_metrics: bool = False):
        """
        :param compute_metrics:
        :param receiver_id:
        :param imitator_id:
        :param sender_id:
        :param inputs:
        :param x: tuple (sender_id,receiver_id,batch)
        :return: (loss_sender,loss_receiver) [batch_size,batch_size]
        """

        agent_sender = self.population.agents[sender_id]
        agent_receiver = self.population.agents[receiver_id]
        agent_imitator = self.population.agents[imitator_id]

        # Agent Sender sends message based on input
        inputs_embedding = agent_sender.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = agent_sender.send(inputs_embedding)

        # Agent receiver encodes message and predict the reconstructed object
        message_embedding = agent_receiver.receive(messages)
        output_receiver = agent_receiver.reconstruct_from_message_embedding(message_embedding)

        # Imitator tries to imitate messages
        _, log_probs_imitation = agent_imitator.get_log_prob_m_given_x(inputs, messages, return_whole_log_probs=True)
        log_imitation = -1 * cross_entropy_imitation(sender_log_prob=log_probs_imitation,
                                                     target_messages=messages)

        for task in agent_imitator.tasks:
            loss = agent_imitator.tasks[task]["loss"].compute(sender_log_prob=log_probs_imitation,
                                                              target_messages=messages)

            agent_imitator.tasks[task]["loss_value"] = loss.mean()

        for task in agent_sender.tasks:
            reward = agent_sender.tasks[task]["loss"].reward_fn(inputs=inputs,
                                                                receiver_output=output_receiver,
                                                                log_imitation=log_imitation).detach()
            loss = agent_sender.tasks[task]["loss"].compute(reward=reward,
                                                            sender_log_prob=log_prob_sender,
                                                            sender_entropy=entropy_sender,
                                                            message=messages)
            agent_sender.tasks[task]["loss_value"] = loss.mean()

        for task in agent_receiver.tasks:
            loss = agent_receiver.tasks[task]["loss"].compute(inputs=inputs,
                                                              receiver_output=output_receiver)

            agent_receiver.tasks[task]["loss_value"] = loss.mean()

        # Compute additional metrics
        metrics = {}

        if compute_metrics:
            # accuracy
            metrics["accuracy"] = accuracy(inputs, output_receiver, game_mode="reconstruction").mean()
            # Sender entropy
            metrics["sender_entropy"] = entropy_sender.mean().item()
            # Sender log prob
            metrics["sender_log_prob"] = log_prob_sender.detach()
            # Raw messages
            metrics["messages"] = messages
            # Average message length
            metrics["message_length"] = find_lengths(messages).float().mean().item()
            # Output receiver
            # metrics["receiver_output"] = output_receiver.detach()

        return metrics

    def forward(self, batch, compute_metrics: bool = False, return_imitation_loss: bool = False):

        return self.game_instance(*batch,
                                  compute_metrics=compute_metrics)


class PretrainingGame(nn.Module):

    def __init__(self,
                 agent: object) -> None:
        super(PretrainingGame, self).__init__()
        self.agent = agent

    def game_instance(self,
                      inputs: th.Tensor,
                      target_messages: th.Tensor,
                      compute_metrics: bool = False):
        """
        :param x: tuple (sender_id,receiver_id,batch)
        :return: (loss_sender,loss_receiver) [batch_size,batch_size]
        """

        # Agent sender sends message based on object
        inputs_encoded = self.agent.encode_object(inputs)
        messages, _, log_prob_sender, entropy_sender = self.agent.send(inputs_encoded, return_whole_log_probs=True)

        loss_sender = cross_entropy_imitation(sender_log_prob=log_prob_sender,target_messages=target_messages)  # [batch_size]

        # Compute additional metrics
        metrics = {}

        if compute_metrics:
            # Raw messages
            metrics["messages"] = messages
            # Sender entropy
            metrics["sender_entropy"] = entropy_sender.mean().item()
            # Average message length
            metrics["message_length"] = find_lengths(messages).float().mean().item()

        return loss_sender.mean(), metrics

    def forward(self, batch, compute_metrics: bool = False):
        loss_sender, metrics = self.game_instance(*batch, compute_metrics=compute_metrics)

        return loss_sender, metrics


def build_game(game_params: dict,
               population: object = None,
               agent: object = None):
    if game_params["game_type"] == "reconstruction":
        assert population is not None, "Specify a population to play the game"
        if "noise_level" in game_params["channel"]:
            noise_level = game_params["channel"]["noise_level"]
        else:
            noise_level=0.
        game = ReconstructionGame(population=population,
                                  voc_size=game_params["channel"]["voc_size"],
                                  max_len=game_params["channel"]["max_len"],
                                  noise_level=noise_level)
    elif game_params["game_type"] == "reconstruction_imitation":
        assert population is not None, "Specify a population to play the game"
        game = ReconstructionImitationGame(population=population)
    elif game_params["game_type"] == "referential":
        assert population is not None, "Specify a population to play the game"
        game = ReferentialGame(population=population)
    elif game_params["game_type"] == "speaker_pretraining":
        assert agent is not None, "Specify a Speaker agent to be pretrained"
        game = PretrainingGame(agent=agent)
    else:
        raise "Specify a known game type"

    return game
