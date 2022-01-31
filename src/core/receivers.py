import torch as th
import torch.nn as nn
from src.zoo.populations.utils import find_lengths

cell_types = {'rnn': nn.RNN, 'gru': nn.GRU, 'LSTM': nn.LSTM}

class Receiver(nn.Module):

    def __init__(self,
                 message_encoder : nn.Module):

        super(Receiver, self).__init__()

        self.message_encoder = message_encoder

    def forward(self,message,context=None):

        embedding = self.message_encoder(message)

        return embedding

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


def build_receiver(receiver_params,game_params):

    # Network params
    receiver_type = receiver_params["receiver_type"]
    receiver_cell = receiver_params["receiver_cell"]
    receiver_embed_dim = receiver_params["receiver_embed_dim"]
    receiver_num_layers = receiver_params["receiver_num_layers"]
    receiver_hidden_size = receiver_params["receiver_hidden_size"]

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
    else:
        raise "Specify a known receiver type"

    receiver = Receiver(message_encoder = message_encoder)

    return receiver