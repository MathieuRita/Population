import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'LSTM': nn.LSTMCell}

# Main sender class
class Sender(nn.Module):

    """
    Each sender is composed of:
    - an encoder: Input object --> embedding
    - a messsage generator : embedding --> message
    """

    def __init__(self,
                 message_generator: nn.Module):

        """
        :type encoder: nn.Module
        :type message_generator: nn.Module
        """

        super(Sender, self).__init__()

        self.message_generator = message_generator

    def forward(self,
                embedding,
                context=None,
                return_whole_log_probs=False):

        return  self.message_generator(embedding,return_whole_log_probs=return_whole_log_probs)


    def get_log_prob_m_given_x(self,embedding,messages, return_whole_log_probs : bool = False):

        return self.message_generator.get_log_prob_m_given_x(embedding,messages, return_whole_log_probs)

# Message generators

class RecurrentGenerator(nn.Module):

    def __init__(self,
                 sender_cell,
                 sender_embed_dim,
                 sender_num_layers,
                 sender_hidden_size,
                 voc_size,
                 max_len):

        super(RecurrentGenerator, self).__init__()

        # Network params
        self.sender_cell = sender_cell
        self.sender_num_layers = sender_num_layers
        self.sender_embed_dim = sender_embed_dim
        self.sender_hidden_size = sender_hidden_size
        self.sos_embedding = nn.Parameter(th.zeros(self.sender_embed_dim))

        # Communication channel params
        self.voc_size = voc_size
        self.max_len = max_len

        # Network init
        if sender_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {sender_cell}")
        cell_type = cell_types[sender_cell]
        self.sender_cells = nn.ModuleList([
            cell_type(input_size=self.sender_embed_dim, hidden_size=self.sender_hidden_size) if i == 0 else \
            cell_type(input_size=self.sender_hidden_size, hidden_size=self.sender_hidden_size) \
            for i in range(self.sender_num_layers)])

        self.sender_norm_h = nn.LayerNorm(self.sender_hidden_size)
        self.sender_norm_c = nn.LayerNorm(self.sender_hidden_size)
        self.hidden_to_output = nn.Linear(self.sender_hidden_size, self.voc_size)
        self.sender_embedding = nn.Embedding(self.voc_size, self.sender_embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 1.)

    def forward(self,embedding, return_whole_log_probs : bool = False):

        prev_hidden = [embedding]
        prev_hidden.extend([th.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers - 1)])
        prev_c = [th.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers)]  # only used for LSTM

        input = th.stack([self.sos_embedding] * embedding.size(0))

        sequence = []
        logits = []
        whole_logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.sender_cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.sender_norm_h(h_t)
                    c_t = self.sender_norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.sender_norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                #x = distr.sample()
                x = step_logits.argmax(dim=1)

            logits.append(distr.log_prob(x))

            if return_whole_log_probs: whole_logits.append(step_logits)
            input = self.sender_embedding(x)
            sequence.append(x)

        sequence = th.stack(sequence).permute(1, 0)
        logits = th.stack(logits).permute(1, 0)
        if return_whole_log_probs: whole_logits = th.stack(whole_logits).permute(1, 0, 2)
        entropy = th.stack(entropy).permute(1, 0)

        if not return_whole_log_probs:
            return sequence, logits, entropy
        else:
            return sequence, logits, whole_logits, entropy


    def get_log_prob_m_given_x(self,embedding,messages, return_whole_log_probs : bool = False):

        prev_hidden = [embedding]
        prev_hidden.extend([th.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers - 1)])
        prev_c = [th.zeros_like(prev_hidden[0]) for _ in range(self.sender_num_layers)]  # only used for LSTM

        input = th.stack([self.sos_embedding] * embedding.size(0))

        logits = []
        whole_logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.sender_cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    h_t = self.sender_norm_h(h_t)
                    c_t = self.sender_norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])
                    h_t = self.sender_norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            x = messages[:,step]

            logits.append(distr.log_prob(x))

            if return_whole_log_probs: whole_logits.append(step_logits)

            input = self.sender_embedding(x)

        logits = th.stack(logits).permute(1, 0)
        entropy = th.stack(entropy).permute(1, 0)
        if return_whole_log_probs: whole_logits = th.stack(whole_logits).permute(1, 0, 2)

        if not return_whole_log_probs:
            return logits
        else:
            return logits, whole_logits


# Build sender

def build_sender(sender_params,game_params):

    # Network params
    sender_type = sender_params["sender_type"]
    sender_cell = sender_params["sender_cell"]
    sender_embed_dim = sender_params["sender_embed_dim"]
    sender_num_layers = sender_params["sender_num_layers"]
    sender_hidden_size = sender_params["sender_hidden_size"]

    # Object params
    object_type = game_params["objects"]["object_type"]
    object_params = game_params["objects"]

    # Channel params
    voc_size=game_params["channel"]["voc_size"]
    max_len = game_params["channel"]["max_len"]

    if "lm_mode" in sender_params and sender_params["lm_mode"]==1:
        voc_size+=1
        max_len+=1

    # Message generator
    if sender_type=="recurrent":

        message_generator = RecurrentGenerator(sender_cell=sender_cell,
                                               sender_embed_dim=sender_embed_dim,
                                               sender_num_layers=sender_num_layers,
                                               sender_hidden_size=sender_hidden_size,
                                               voc_size=voc_size,
                                               max_len=max_len)

    else:
        raise "Specify a know sender type"

    sender = Sender(message_generator = message_generator)

    return sender
