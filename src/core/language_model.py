import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'LSTM': nn.LSTM}
PAD_TOKEN = 0
START_TOKEN = 1
EOS_TOKEN = 2


def build_data_lm(messages):

    max_len = messages.size(1)

    messages = EOS_TOKEN + messages
    start_tokens = th.Tensor([START_TOKEN] * messages.size(0)).unsqueeze(1).to(messages.device)
    messages = th.cat((start_tokens, messages), dim=1)

    eos_mask = messages == EOS_TOKEN
    message_lengths = max_len + 1 - (eos_mask.cumsum(dim=1) > 0).sum(dim=1)
    message_lengths.add_(1).clamp_(max=max_len)

    pad_mask = 1 - th.cumsum(1 * eos_mask, dim=1).clamp_(max=1)
    messages = messages * pad_mask
    messages += EOS_TOKEN * eos_mask
    messages = messages.to(int)

    x = messages[:, :-1]
    y = messages[:, 1:]
    x_lengths = message_lengths - 1

    return x, y, x_lengths


class LanguageModel():

    def __init__(self,
                 model,
                 optimizer,
                 batch_size):

        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size

    def get_prob_messages(self, messages):

        # Gerer le pb descending order pour les lengths

        self.model.eval()

        with th.no_grad():
            x_test, y_test, x_lengths_test = build_data_lm(messages=messages)

            # Reorder by length
            y_hat = self.model(x_test, x_lengths_test)

            batch_size, seq_len, nb_vocab_words = y_hat.size()
            mask = nn.functional.one_hot(x_lengths_test, num_classes=seq_len + 1)
            mask = 1 - th.cumsum(mask, dim=1)[:, :-1]

            y_hat = y_hat * mask.unsqueeze(2)
            y_hat = th.exp(y_hat)
            y_hat = y_hat.contiguous()
            y_hat = y_hat.view(-1, self.model.voc_size)

            y_test = y_test.contiguous()
            y_hat = y_hat[range(y_hat.size(0)), y_test.view(-1)]
            y_hat = y_hat.resize(batch_size, seq_len)

        return y_hat.prod(1)

    def compute_loss(self, y_hat, y, x_lengths):

        # create a mask by filtering out all tokens that ARE NOT the padding token
        mask = F.one_hot(x_lengths, num_classes=y_hat.size(1) + 1)
        mask = 1 - th.cumsum(mask, dim=1)[:, :-1]
        mask = mask.to(y_hat.device)

        # flatten all the labels
        y = y.contiguous()
        y = y.view(-1)

        # flatten all predictions
        y_hat = y_hat.contiguous()
        y_hat = y_hat.view(-1, self.model.voc_size)

        # count how many tokens we have
        nb_tokens = int(th.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        y_hat = y_hat[range(y_hat.size(0)), y] * mask.view(-1)

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -th.sum(y_hat) / nb_tokens

        return ce_loss

    def train(self,
              messages,
              n_epochs: int = 200,
              threshold: float = 1e-4):

        x, y, x_lengths = build_data_lm(messages=messages)
        r = th.randperm(x.size(0))
        x = x[r]
        y = y[r]
        x_lengths = x_lengths[r]

        self.model.train()

        num_batches = int(x.size(0) / self.batch_size)

        prev_losses = []

        continue_training = True
        epoch = 0

        while continue_training:

            mean_loss=0.

            # Mini batches
            for i in range(num_batches):
                x_batch = x[i * self.batch_size: (i + 1) * self.batch_size]
                y_batch = y[i * self.batch_size: (i + 1) * self.batch_size]
                len_batch = x_lengths[i * self.batch_size: (i + 1) * self.batch_size]

                x_batch = x_batch.to(self.model.device)
                y_batch = y_batch.to(self.model.device)
                len_batch = len_batch.to(self.model.device)

                y_hat = self.model(x_batch, len_batch)
                loss = self.compute_loss(y_hat, y_batch, len_batch)

                self.optimizer.zero_grad()

                # Calculate gradients
                loss.backward()

                # Updated parameters
                self.optimizer.step()

                mean_loss+=loss.item()

            mean_loss/=num_batches

            if (len(prev_losses) > 4 or abs(mean_loss - np.mean(prev_losses)) < threshold) or epoch >= n_epochs:
                continue_training = False
            else:
                prev_losses.append(mean_loss)
                epoch += 1
                if len(prev_losses) > 5 : prev_losses.pop(0)


class LanguageModelNetwork(nn.Module):

    def __init__(self,
                 max_len: int,
                 voc_size: int,
                 num_layers: int = 1,
                 hidden_size: int = 128,
                 embedding_size: int = 20,
                 device : str ="cpu") -> None:
        super(LanguageModelNetwork, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.voc_size = voc_size
        self.max_len = max_len
        self.device=device

        self.word_embedding = nn.Embedding(
            num_embeddings=self.voc_size,
            embedding_dim=self.embedding_size,
            padding_idx=PAD_TOKEN
        )

        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.hidden_to_symbol = nn.Linear(self.hidden_size, self.voc_size)

    def init_hidden(self, batch_size):
        hidden_a = th.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        hidden_b = th.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        return hidden_a, hidden_b

    def forward(self, x, x_lengths):
        batch_size = x.size(0)

        # Prepare data
        hidden = self.init_hidden(batch_size)
        x = self.word_embedding(x)

        x = th.nn.utils.rnn.pack_padded_sequence(x, x_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # now run through LSTM
        x, hidden = self.lstm(x, hidden)

        # undo the packing operation
        x, _ = th.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if x.size(1) < self.max_len:
            dummy_tensor = th.zeros((batch_size,self.max_len - x.size(1), x.size(2)),device=self.device)
            x = th.cat([x, dummy_tensor], 1)

        x = x.contiguous()
        x = x.view(-1, x.size(2))

        # Pass through actual linear layer
        y_hat = self.hidden_to_symbol(x)
        y_hat = F.log_softmax(y_hat, dim=1)
        y_hat = y_hat.view(batch_size, self.max_len, self.voc_size)

        return y_hat


def get_language_model(lm_params:dict,game_params:dict,device:str):

    model = LanguageModelNetwork(max_len=game_params["channel"]["max_len"],
                                 voc_size=game_params["channel"]["voc_size"]+2,
                                 num_layers=lm_params["num_layers"],
                                 hidden_size=lm_params["hidden_size"],
                                 embedding_size=lm_params["embedding_size"],
                                 device=device)

    model.to(device)

    optimizer = th.optim.RMSprop(model.parameters(), lr=0.001)

    language_model = LanguageModel(model=model,optimizer=optimizer,batch_size=lm_params["batch_size"])

    return language_model