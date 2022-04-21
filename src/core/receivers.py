import torch as th
import torch.nn as nn
from .utils import find_lengths

cell_types = {'rnn': nn.RNN, 'gru': nn.GRU, 'LSTM': nn.LSTM}


class Receiver(nn.Module):

    def __init__(self,
                 message_encoder: nn.Module):
        super(Receiver, self).__init__()

        self.message_encoder = message_encoder

    def forward(self, message, context=None):
        embedding = self.message_encoder(message)

        return embedding

    def reset_parameters(self):
        self.message_encoder.reset_parameters()


# MessageProcessor classes

class RecurrentProcessor(nn.Module):

    def __init__(self,
                 receiver_cell,
                 receiver_embed_dim,
                 receiver_num_layers,
                 receiver_hidden_size,
                 voc_size,
                 max_len
                 ):
        super(RecurrentProcessor, self).__init__()

        # Network parameters
        self.receiver_embed_dim = receiver_embed_dim
        self.receiver_num_layers = receiver_num_layers
        self.receiver_hidden_size = receiver_hidden_size
        self.sos_embedding = nn.Parameter(th.zeros(self.receiver_embed_dim))

        # Communication channel parameters
        self.voc_size = voc_size
        self.max_len = max_len

        # Network

        if receiver_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {receiver_cell}")

        self.receiver_cell = cell_types[receiver_cell](input_size=self.receiver_embed_dim,
                                                       batch_first=True,
                                                       hidden_size=self.receiver_hidden_size,
                                                       num_layers=self.receiver_num_layers)

        self.receiver_embedding = nn.Embedding(self.voc_size, self.receiver_embed_dim)
        self.receiver_norm_h = nn.LayerNorm(self.receiver_hidden_size)
        self.receiver_norm_c = nn.LayerNorm(self.receiver_hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 1.)

    def forward(self,
                message,
                message_lengths=None):

        emb = self.receiver_embedding(message)

        if message_lengths is None:
            message_lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, message_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, rnn_hidden = self.receiver_cell(packed)

        if isinstance(self.receiver_cell, nn.LSTM):
            rnn_hidden, _ = rnn_hidden

        encoded = rnn_hidden[-1]

        return encoded


class RecurrentProcessorLayerNorm(nn.Module):

    def __init__(self,
                 receiver_cell,
                 receiver_embed_dim,
                 receiver_num_layers,
                 receiver_hidden_size,
                 voc_size,
                 max_len,
                 dropout_rate : float = None
                 ):
        super(RecurrentProcessorLayerNorm, self).__init__()

        # Network parameters
        self.receiver_embed_dim = receiver_embed_dim
        self.receiver_num_layers = receiver_num_layers
        self.receiver_hidden_size = receiver_hidden_size
        self.sos_embedding = nn.Parameter(th.zeros(self.receiver_embed_dim))

        # Communication channel parameters
        self.voc_size = voc_size
        self.max_len = max_len

        # Network

        if receiver_cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {receiver_cell}")

        cell_type = nn.LSTMCell
        self.receiver_cells = nn.ModuleList([
            cell_type(input_size=self.receiver_embed_dim, hidden_size=self.receiver_hidden_size) if i == 0 else \
                cell_type(input_size=self.receiver_hidden_size, hidden_size=self.receiver_hidden_size) \
            for i in range(self.receiver_num_layers)])

        self.receiver_embedding = nn.Embedding(self.voc_size, self.receiver_embed_dim)
        self.receiver_norm_h = nn.LayerNorm(self.receiver_hidden_size)
        self.receiver_norm_c = nn.LayerNorm(self.receiver_hidden_size)

        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 1.)

    def forward(self,
                message,
                message_lengths=None):

        embedding = self.receiver_embedding(message)

        hidden_all_positions=[]
        prev_hidden = [th.zeros((embedding.size(0), self.receiver_hidden_size), device=embedding.device)
                       for _ in range(self.receiver_num_layers)]
        prev_c = [th.zeros_like(prev_hidden[0]) for _ in range(self.receiver_num_layers)]  # only used for LSTM

        if message_lengths is None:
            message_lengths = find_lengths(message)

        if self.dropout is not None:
            mask_dropout = self.dropout(th.ones((embedding.size(0),
                                                 self.receiver_hidden_size)
                                                ,device=embedding.device))
        else:
            mask_dropout = th.ones((embedding.size(0),self.receiver_hidden_size),device=embedding.device)

        for step in range(self.max_len):
            for i, layer in enumerate(self.receiver_cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(embedding[:, step], (prev_hidden[i], prev_c[i]))
                    h_t = h_t*mask_dropout
                    h_t = self.receiver_norm_h(h_t)
                    c_t = self.receiver_norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(embedding[:, step], prev_hidden[i])
                    h_t = self.receiver_norm_h(h_t)
                prev_hidden[i] = h_t
            hidden_all_positions.append(h_t)

        #encoded = prev_hidden[-1]
        hidden_all_positions = th.stack(hidden_all_positions).permute(1,0,2)
        encoded = hidden_all_positions.gather(dim=1,
                                              index=(message_lengths-1).view(-1,
                                                                             1,
                                                                             1).repeat(1,
                                                                                       1,
                                                                                       hidden_all_positions.size(-1)))

        encoded=encoded.squeeze(1)

        return encoded


def build_receiver(receiver_params, game_params):
    # Network params
    receiver_type = receiver_params["receiver_type"]
    receiver_cell = receiver_params["receiver_cell"]
    receiver_embed_dim = receiver_params["receiver_embed_dim"]
    receiver_num_layers = receiver_params["receiver_num_layers"]
    receiver_hidden_size = receiver_params["receiver_hidden_size"]
    dropout_rate = receiver_params["dropout_rate"] if "dropout_rate" in receiver_params else None

    # Channel params
    voc_size = game_params["channel"]["voc_size"]
    max_len = game_params["channel"]["max_len"]

    # Message processor
    if receiver_type == "recurrent":
        message_encoder = RecurrentProcessor(receiver_cell=receiver_cell,
                                             receiver_embed_dim=receiver_embed_dim,
                                             receiver_num_layers=receiver_num_layers,
                                             receiver_hidden_size=receiver_hidden_size,
                                             voc_size=voc_size,
                                             max_len=max_len)
    elif receiver_type == "recurrent_layernorm":
        message_encoder = RecurrentProcessorLayerNorm(receiver_cell=receiver_cell,
                                                      receiver_embed_dim=receiver_embed_dim,
                                                      receiver_num_layers=receiver_num_layers,
                                                      receiver_hidden_size=receiver_hidden_size,
                                                      voc_size=voc_size,
                                                      max_len=max_len,
                                                      dropout_rate = dropout_rate)
    else:
        raise "Specify a known receiver type"

    receiver = Receiver(message_encoder=message_encoder)

    return receiver
