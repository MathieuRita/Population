import torch as th
import torch.nn as nn
from .losses import accuracy, find_lengths


class ReconstructionGame(nn.Module):

    def __init__(self,
                 population: object):

        super(ReconstructionGame, self).__init__()
        self.population = population

    def game_instance(self,
                      inputs: th.Tensor,
                      sender_id: th.Tensor,
                      receiver_id: th.Tensor,
                      compute_metrics: bool = False):

        """
        :param x: tuple (sender_id,receiver_id,batch)
        :return: (loss_sender,loss_receiver) [batch_size,batch_size]
        """

        agent_sender = self.population.agents[sender_id]
        agent_receiver = self.population.agents[receiver_id]

        # Agent Sender sends message based on input
        inputs_embedding = agent_sender.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = agent_sender.send(inputs_embedding)

        # Agent receiver encodes message and predict the reconstructed object
        message_embedding = agent_receiver.receive(messages)
        output_receiver = agent_receiver.reconstruct_from_message_embedding(message_embedding)

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

        return loss_sender.mean(), loss_receiver.mean(), metrics

    def forward(self, batch, compute_metrics: bool = False):

        loss_sender, loss_receiver, metrics = self.game_instance(*batch, compute_metrics=compute_metrics)

        return loss_sender, loss_receiver, metrics

    def imitation_instance(self,
                           inputs: th.Tensor,
                           N_samples: int):

        target_messages = []

        for sender_id in self.population.sender_names:
            agent_sender = self.population.agents[sender_id]
            inputs_embedding = agent_sender.encode_object(inputs)
            for _ in range(N_samples):
                messages, _, _ = agent_sender.send(inputs_embedding)
                target_messages.append(messages)

        target_messages = th.stack(target_messages)
        target_messages = target_messages.reshape(target_messages.size(0) * target_messages.size(1),
                                                  target_messages.size(2))

        inputs = inputs.repeat(len(self.population.sender_names) * N_samples, 1, 1)
        losses = {}

        for sender_id in self.population.sender_names:
            agent_sender = self.population.agents[sender_id]
            inputs_encoded = agent_sender.encode_object(inputs)

            messages, _, log_prob_sender, entropy_sender = agent_sender.send(inputs_encoded,
                                                                             return_whole_log_probs=True)

            loss_sender = agent_sender.compute_sender_imitation_loss(sender_log_prob=log_prob_sender,
                                                                     target_messages=target_messages)  # [batch_size]

            losses[sender_id] = loss_sender.mean()

        return losses


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

        return loss_sender.mean(), loss_receiver.mean(), metrics

    def forward(self, batch, compute_metrics: bool = False):
        loss_sender, loss_receiver, metrics = self.game_instance(*batch, compute_metrics=compute_metrics)

        return loss_sender, loss_receiver, metrics


class ReconstructionImitationGame(nn.Module):

    def __init__(self,
                 population: object):
        super(ReconstructionImitationGame, self).__init__()
        self.population = population

    def game_instance(self,
                      inputs:th.Tensor,
                      sender_id:str,
                      receiver_id:str,
                      imitator_id:str,
                      compute_metrics: bool = False,
                      return_imitation_loss:bool = False):
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

        loss_imitator = agent_imitator.compute_sender_imitation_loss(sender_log_prob=log_probs_imitation,
                                                                    target_messages=messages)  # [batch_size]

        loss_sender = agent_sender.compute_sender_loss(inputs=inputs,
                                                       sender_log_prob=log_prob_sender,
                                                       sender_entropy=entropy_sender,
                                                       messages=messages,
                                                       receiver_output=output_receiver,
                                                       neg_log_imit=loss_imitator)  # [batch_size]

        loss_receiver = agent_receiver.compute_receiver_loss(inputs=inputs,
                                                             output_receiver=output_receiver)  # [batch_size]

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

        if return_imitation_loss:
            return loss_sender.mean(), loss_receiver.mean(), loss_imitator.mean(), metrics
        else:
            return loss_sender.mean(), loss_receiver.mean(), metrics

    def forward(self, batch, compute_metrics: bool = False,return_imitation_loss:bool=False):

        return self.game_instance(*batch,
                                compute_metrics=compute_metrics,
                                return_imitation_loss=return_imitation_loss)


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

        loss_sender = self.agent.compute_sender_imitation_loss(sender_log_prob=log_prob_sender,
                                                               target_messages=target_messages)  # [batch_size]

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
        game = ReconstructionGame(population=population)
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
