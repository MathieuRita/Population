import torch as th
import torch.nn.functional as F
from .utils import find_lengths


def cross_entropy(inputs, receiver_output, multi_attribute: bool = False) -> th.Tensor():
    """

    :param multi_attribute:
    :param inputs: one hots [batch_size,n_attributes,n_values]
    :param receiver_output: log-prob of receiver predictions [batch_size,n_attributes,n_values]
    :return: log_prob of correct reconstruction
    """

    # Dimensions
    batch_size = inputs.size(0)
    n_attributes = inputs.size(1)
    n_classes = inputs.size(2)

    # Reshape tensors
    inputs = inputs.reshape((batch_size * n_attributes, n_classes))  # [batch_size*n_attributes,n_classes]
    inputs = inputs.argmax(dim=1)  # [batch_size*n_attributes]
    receiver_output = receiver_output.reshape(
        (batch_size * n_attributes, n_classes))  # [batch_size*n_attributes,n_classes]

    # Compute loss
    log_pi_receiver = F.cross_entropy(receiver_output, inputs, reduction="none")  # [batch_size*n_attributes]
    if multi_attribute:
        log_pi_receiver = log_pi_receiver.reshape((batch_size, n_attributes))  # [batch_size,n_attributes]
    else:
        log_pi_receiver = log_pi_receiver.reshape((batch_size, n_attributes)).sum(dim=1)  # [batch_size]

    return log_pi_receiver


def cross_entropy_imitation(sender_log_prob: th.Tensor = None, target_messages: th.Tensor = None) -> th.Tensor:
    batch_size, max_len, vocab_size = sender_log_prob.size(0), sender_log_prob.size(1), sender_log_prob.size(2)

    message_lengths = find_lengths(target_messages)

    sender_log_prob = sender_log_prob.reshape((batch_size * max_len, vocab_size))  # [batch_size*max_len,vocab_size]
    target_messages = target_messages.reshape((batch_size * max_len))  # [batch_size*max_len]

    loss = F.cross_entropy(sender_log_prob, target_messages, reduction="none")
    loss = loss.reshape((batch_size, max_len))

    mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
                                       num_classes=max_len + 1), dim=1)[:, :-1]  # eg. [1,1,0] if length=2 and ml=3
    loss = (loss * mask_eos)  # [batch_size,max_len]

    loss = loss.sum(dim=1)

    return loss


def accuracy(inputs,
             receiver_output,
             game_mode: str,
             output_transformation: str = "greedy_reduction",
             all_attributes_equal: bool = False,
             reduce_attributes : bool = True,
             idx_correct_object: int = 0) -> th.Tensor():
    if game_mode == "reconstruction":
        # Flatten inputs over values
        inputs = inputs.argmax(dim=2)  # [batch_size,n_attributes]

        # Receiver candidates
        if output_transformation=="greedy_reduction":
            receiver_output = receiver_output.argmax(dim=2)
        elif output_transformation=="sampling_reduction":
            raise NotImplementedError

        # Accuracy
        acc = (inputs == receiver_output)  # [batch_size,n_attributes]

        if all_attributes_equal:
            acc = (1 * th.all(acc, dim=1)).float()  # [batch_size]
        else:
            acc = (1 * acc).float() # [batch_size,n_attributes]
            if reduce_attributes: acc=acc.mean(dim=1) # [batch_size]

    elif game_mode == "referential":
        acc = 1 * (receiver_output.argmax(dim=1) == idx_correct_object).float()

    return acc


def get_log_prob_given_index(receiver_output, idx: int = 0):
    return receiver_output[:, idx]


def baseline_norm(reward):
    return reward - reward.mean()


# Specific losses
class ReinforceLoss:

    def __init__(self,
                 reward_type: str = "log",
                 baseline_type: str = "normalization_batch",
                 entropy_reg_coef: float = 0.,
                 length_reg_coef: float = 0.):

        self.baseline_type = baseline_type
        self.entropy_reg_coef = entropy_reg_coef
        self.length_reg_coef = length_reg_coef
        self.reward_type = reward_type

        if reward_type == "log":
            self.reward_fn = lambda inputs, receiver_output, log_imitation=None: \
                -1 * cross_entropy(inputs, receiver_output)
        elif reward_type == "accuracy_reconstruction":
            self.reward_fn = lambda inputs, receiver_output, output_transformation="greedy_reduction", log_imitation=None: accuracy(inputs,
                                                                 receiver_output,
                                                                 game_mode="reconstruction",
                                                                 output_transformation=output_transformation)
        elif reward_type == "referential_log":
            self.reward_fn = lambda inputs, receiver_output, log_imitation=None: get_log_prob_given_index(receiver_output)
        elif reward_type == "accuracy_referential":
            self.reward_fn = lambda inputs, receiver_output, log_imitation=None: accuracy(inputs,
                                                                                          receiver_output,
                                                                                          game_mode="referential")
        elif reward_type == "imitation":
            self.reward_fn = lambda inputs, receiver_output, log_imitation: log_imitation
        else:
            raise "Set a known reward type"

        if baseline_type == "normalization_batch":
            self.baseline_fn = lambda reward: baseline_norm(reward)
        else:
            raise "Set a known baseline type"

    def compute(self,
                reward: th.Tensor,
                log_prob: th.Tensor,
                entropy: th.Tensor,
                message: th.Tensor = None,
                agent_type : str ="sender"
                ):

        if agent_type=="sender":
            return self.compute_sender(reward=reward,
                                       sender_log_prob = log_prob,
                                       sender_entropy = entropy,
                                       message = message)

        if agent_type=="receiver":
            return self.compute_receiver(reward=reward,
                                         receiver_log_prob = log_prob,
                                         receiver_entropy = entropy)


    def compute_sender(self,
                       reward: th.Tensor,
                       sender_log_prob: th.Tensor,
                       sender_entropy: th.Tensor,
                       message: th.Tensor):

        """
        TO DO :
        - implement KL regularization

        :param reward:
        :param neg_log_imit:
        :param receiver_output:
        :param sender_entropy:
        :param sender_log_prob:
        :param inputs:
        :param message:
        :return:
        """

        message_lengths = find_lengths(message)
        max_len = message.size(1)

        # Mask log_prob / entropy post EOS
        mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
                                           num_classes=max_len + 1), dim=1)[:, :-1]  # eg. [1,1,0] if length=2 and ml=3
        sender_log_prob = (sender_log_prob * mask_eos).sum(dim=1)  # [batch_size]
        sender_entropy = (sender_entropy * mask_eos).sum(dim=1) / message_lengths.float()  # [batch_size]

        # Policy gradient loss
        reward = self.baseline_fn(reward=reward)
        policy_loss = - (reward * sender_log_prob)  # [batch_size]

        # Entropy regularization
        entropy_regularization_penalty = self.entropy_reg_coef * sender_entropy  # [batch_size]

        # Message length regularization
        reward_message_length_penalty = - self.length_reg_coef * message_lengths.float()
        reward_message_length_penalty = self.baseline_fn(reward_message_length_penalty)
        policy_length_penalty = (reward_message_length_penalty * sender_log_prob)  # [batch_size]

        loss = policy_loss - policy_length_penalty - entropy_regularization_penalty  # [batch_size]

        return loss

    def compute_receiver(self,
                       reward: th.Tensor,
                       receiver_log_prob: th.Tensor,
                       receiver_entropy: th.Tensor):

        """
        TO DO :
        - implement KL regularization

        :param reward:
        :param neg_log_imit:
        :param receiver_output:
        :param sender_entropy:
        :param sender_log_prob:
        :param inputs:
        :param message:
        :return:
        """

        # Policy gradient loss
        reward = self.baseline_fn(reward=reward)
        policy_loss = - (reward * receiver_log_prob)  # [batch_size]

        # Entropy regularization
        entropy_regularization_penalty = self.entropy_reg_coef * receiver_entropy  # [batch_size]

        loss = policy_loss - entropy_regularization_penalty  # [batch_size]

        return loss



class CrossEntropyLoss:

    def __init__(self, multi_attribute: bool = False):
        self.multi_attribute = multi_attribute

    def compute(self,
                inputs,
                receiver_output,
                agent_type : str = "receiver"):
        return cross_entropy(inputs, receiver_output, multi_attribute=self.multi_attribute)  # [batch_size]


class ReferentialLoss:

    def __init__(self, id_correct_object: int = 0):
        self.id_correct_object = id_correct_object

    def compute(self, inputs: th.Tensor = None, receiver_output: th.Tensor = None):
        return -1 * get_log_prob_given_index(receiver_output, self.id_correct_object)


class SpeakerImitation:

    def __init__(self, reduce: bool = True):
        self.reduce = reduce

    def compute(self, sender_log_prob: th.Tensor = None, target_messages: th.Tensor = None):

        return cross_entropy_imitation(sender_log_prob=sender_log_prob,target_messages=target_messages)
