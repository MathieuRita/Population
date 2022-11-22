import torch as th
import torch.nn as nn
import torch.nn.functional as F


class OneHotEncoder(nn.Module):

    def __init__(self,
                 object_params: dict,
                 embedding_size: int,
                 ) -> None:
        super(OneHotEncoder, self).__init__()

        # Dims
        self.n_values = object_params["n_values"]
        self.n_attributes = object_params["n_attributes"]
        self.embedding_size = embedding_size
        self.sos_embedding = nn.Parameter(th.zeros(self.embedding_size))

        self.encoder = nn.Linear(self.n_values * self.n_attributes,
                                 self.embedding_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 1.)

    def forward(self,
                x,
                context=None):
        x = x.reshape(x.size(0), self.n_attributes * self.n_values).float()  # flatten objects
        embedding = self.encoder(x)

        return embedding


class OneHotDecoder(nn.Module):

    def __init__(self,
                 object_params: dict,
                 embedding_size: int):
        super(OneHotDecoder, self).__init__()

        # Network params
        self.embedding_size = embedding_size
        self.sos_embedding = nn.Parameter(th.zeros(embedding_size))

        # Object params
        self.n_values = object_params["n_values"]
        self.n_attributes = object_params["n_attributes"]

        # Network
        self.linear_output = nn.Linear(self.embedding_size, self.n_values * self.n_attributes)

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 1.)

    def forward(self, encoded):
        output = self.linear_output(encoded).reshape(encoded.size(0),
                                                     self.n_attributes,
                                                     self.n_values)

        output = F.log_softmax(output, dim=2)  # Softmax for each attribute

        return output


class ImageEncoder(nn.Module):

    def __init__(self):
        raise NotImplementedError

    def forward(self,
                x,
                context=None):
        raise NotImplementedError


def build_encoder(object_params: dict,
                  embedding_size: int):
    if object_params["object_type"] == "one_hot":

        encoder = OneHotEncoder(object_params=object_params,
                                embedding_size=embedding_size)

    elif object_params["object_type"] == "image_logit":
        encoder = nn.Linear(object_params["n_logits"],embedding_size)

    else:
        raise "Specify a known object type"

    return encoder


def build_decoder(object_params: dict,
                  embedding_size: int,
                  projection_size : int = 100):
    if object_params["object_type"] == "one_hot":

        decoder = OneHotDecoder(object_params=object_params,
                                embedding_size=embedding_size)

    elif object_params["object_type"] == "image_logit":
        if projection_size==-1:
            decoder = nn.Linear(embedding_size, object_params["n_logits"])
        else:
            decoder = nn.Linear(embedding_size, projection_size)

    else:
        raise "Specify a known object type"

    return decoder


def build_object_projector(object_params: dict,
                           projection_size: int):

    if object_params["object_type"] == "image_logit":
        if projection_size == -1:
            projector = nn.Identity()
        else:
            projector = nn.Linear(object_params["n_logits"], projection_size)
            #W1 = nn.Linear(object_params["n_logits"], projection_size)
            #A1 = nn.ReLU()
            #W2 = nn.Linear(projection_size, projection_size)
            #projector = nn.Sequential(W1, A1,W2

            #for param in projector.parameters():
            #    param.requires_grad = False
    else:
        raise "Specify a known object type"

    return projector
