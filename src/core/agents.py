import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity
import numpy as np

from .utils import find_lengths, move_to
from .senders import build_sender
from .receivers import build_receiver
from .object_encoders import build_encoder, build_decoder
from .losses import ReinforceLoss, CrossEntropyLoss, ReferentialLoss, SpeakerImitation


class Agent(object):

    def __init__(self,
                 agent_name: str,
                 sender:nn.Module ,
                 receiver:nn.Module,
                 object_encoder : nn.Module,
                 object_decoder : nn.Module,
                 sender_optimizer : th.optim.Optimizer,
                 receiver_optimizer : th.optim.Optimizer,
                 sender_loss_fn:object,
                 receiver_loss_fn:object,
                 sender_imitation_loss_fn:object,
                 p_step: float = 1.,
                 device:str="cpu"):

        self.agent_name = agent_name
        self.object_encoder = object_encoder
        self.sender = sender
        self.receiver = receiver
        self.object_decoder = object_decoder
        self.sender_optimizer = sender_optimizer
        self.receiver_optimizer = receiver_optimizer
        self.p_step = p_step
        self.sender_loss_fn = sender_loss_fn
        self.sender_imitation_loss_fn = sender_imitation_loss_fn
        self.receiver_loss_fn = receiver_loss_fn
        self.device = device

    def encode_object(self,x):

        return self.object_encoder(x)

    def send(self,
             x,
             context=None,
             return_whole_log_probs : bool = False):

        return self.sender(x,context,return_whole_log_probs)

    def get_log_prob_m_given_x(self,x,m):

        embedding = self.object_encoder(x)
        log_prob = self.sender.get_log_prob_m_given_x(embedding,m)

        return log_prob

    def receive(self,
                message,
                context=None):

        return self.receiver(message)

    def reconstruct_from_message_embedding(self,embedding):

        return self.object_decoder(embedding)

    def predict_referential_candidate(self,
                                      message_embedding:th.Tensor,
                                      inputs_embedding:th.Tensor,
                                      distractors_embedding:th.Tensor):

        # Expand dims across distractors axis
        inputs_embedding = inputs_embedding.reshape((inputs_embedding.size(0),1,inputs_embedding.size(1)))
        message_embedding = message_embedding.reshape((message_embedding.size(0), 1, message_embedding.size(1)))

        embeddings = th.concat((inputs_embedding,
                                distractors_embedding),dim=1)

        cos = CosineSimilarity(dim=2)
        message_object_scalar_product = cos(message_embedding,embeddings)

        output = F.log_softmax(message_object_scalar_product, dim=1)

        return output

    def compute_sender_loss(self,inputs,sender_log_prob,sender_entropy,messages,receiver_output):

        return self.sender_loss_fn.compute(inputs=inputs,
                                           message=messages,
                                           sender_log_prob=sender_log_prob,
                                           sender_entropy=sender_entropy,
                                           receiver_output=receiver_output)

    def compute_mutual_information(self,inputs):

        inputs_embeddings = self.encode_object(inputs)
        messages, log_prob_sender, entropy_sender = self.send(inputs_embeddings)
        batch_size = messages.size(0)

        id_sampled_messages = np.arange(batch_size)
        sampled_messages = messages[id_sampled_messages]
        sampled_messages = sampled_messages.unsqueeze(0)
        sampled_messages = sampled_messages.repeat(batch_size, 1, 1)
        sampled_messages = sampled_messages.permute(1, 0, 2)
        sampled_messages = sampled_messages.reshape([batch_size * batch_size, sampled_messages.size(-1)])
        sampled_x = inputs.repeat(batch_size, 1, 1, 1)
        sampled_x = sampled_x.reshape([batch_size * batch_size, *inputs.size()[1:]])

        sampled_x = move_to(sampled_x, self.device)
        sampled_messages = move_to(sampled_messages, self.device)
        log_probs = self.get_log_prob_m_given_x(sampled_x,sampled_messages)

        # log_probs -> pm1_x1,...,pm1_xn ; pm2_x1,...,pm2_xn,.....
        message_lengths = find_lengths(sampled_messages)
        max_len = sampled_messages.size(1)
        mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
                                           num_classes=max_len + 1), dim=1)[:, :-1]
        log_probs = (log_probs * mask_eos).sum(dim=1)

        log_pi_m_x = log_probs.reshape([batch_size, batch_size])
        pi_m_x = th.exp(log_pi_m_x)
        p_x = th.ones(batch_size) / batch_size  # Here we set p(x)=1/batch_size
        p_x = p_x.to(pi_m_x.device)  # Fix device issue

        log_p_x = th.log(p_x)
        log_pi_m = th.log((pi_m_x * p_x).sum(1))
        log_pi_m_x = th.log(pi_m_x.diagonal(0))

        mutual_information = log_pi_m_x + log_p_x - log_pi_m

        reward = mutual_information.detach()
        reward = (reward - reward.mean()) / reward.std()

        loss_mi = - log_pi_m_x * reward

        return loss_mi.mean()

    def compute_sender_imitation_loss(self,sender_log_prob,target_messages):

        return self.sender_imitation_loss_fn.compute(sender_log_prob=sender_log_prob,target_messages=target_messages)

    def compute_receiver_loss(self,inputs,output_receiver):

        return self.receiver_loss_fn.compute(inputs,output_receiver)


def build_agent(agent_name: str,
                agent_repertory : dict,
                game_params:dict,
                device : str = "cpu"):

    agent_params = agent_repertory[agent_name]
    pretrained_modules = agent_params["pretrained_modules"] if "pretrained_modules" in agent_params else {}


    if agent_params["sender"] and agent_params["receiver"]:

        assert agent_params["receiver_params"]["receiver_embed_dim"]==agent_params["sender_params"]["sender_embed_dim"]

        raise NotImplementedError # Should take into account lots of agents modellings

    elif agent_params["sender"] and not agent_params["receiver"]:

        # Models
        object_encoder = build_encoder(object_params=game_params["objects"],
                                       embedding_size = agent_params["sender_params"]["sender_embed_dim"])
        sender = build_sender(sender_params=agent_params["sender_params"], game_params = game_params)
        object_decoder = None
        receiver = None

        # Pretrained modules
        if "object_encoder" in pretrained_modules:
            object_encoder.load_state_dict(th.load(pretrained_modules["object_encoder"]))
        if "sender" in pretrained_modules:
            sender.load_state_dict(th.load(pretrained_modules["sender"]))


        # Send models to device
        sender.to(device)
        object_encoder.to(device)

        # Optimizer
        #sender_optimizer = build_agent_optimizer(model_parameters=list(sender.parameters()),
        #                                         optim_params=agent_params["sender_optim_params"])
        sender_optimizer = build_agent_optimizer(model_parameters=list(object_encoder.parameters()) \
                                                                  +list(sender.parameters()),
                                                 optim_params=agent_params["sender_optim_params"])
        p_step = agent_params["sender_optim_params"]["p_step"]
        receiver_optimizer = None

        # Losses
        sender_loss_fn = build_agent_loss_fn(optim_params=agent_params["sender_optim_params"])
        if game_params["game_type"]=="speaker_pretraining" or len(agent_params["sender_optim_params"]["imitation_loss"]):
            sender_imitation_loss_fn = build_agent_loss_fn(optim_params={"loss":"speaker_imitation"})
        else:
            sender_imitation_loss_fn = None
        receiver_loss_fn = None

    elif not agent_params["sender"] and agent_params["receiver"]:

        # Models
        object_encoder = build_encoder(object_params=game_params["objects"],
                                       embedding_size=agent_params["receiver_params"]["receiver_embed_dim"])
        object_decoder = build_decoder(object_params=game_params["objects"],
                                       embedding_size=agent_params["receiver_params"]["receiver_embed_dim"])
        sender = None
        receiver = build_receiver(receiver_params=agent_params["receiver_params"], game_params = game_params)

        # Pretrained modules
        if "object_encoder" in pretrained_modules:
            object_encoder.load_state_dict(th.load(pretrained_modules["object_encoder"]))
        if "object_decoder" in pretrained_modules:
            object_decoder.load_state_dict(th.load(pretrained_modules["object_decoder"]))
        if "receiver" in pretrained_modules:
            receiver.load_state_dict(th.load(pretrained_modules["receiver"]))

        # Send models to device
        receiver.to(device)
        object_encoder.to(device)
        object_decoder.to(device)

        # Optimizers
        sender_optimizer = None
        receiver_optimizer = build_agent_optimizer(model_parameters=list(receiver.parameters()) \
                                                                    + list(object_decoder.parameters()),
                                                   optim_params=agent_params["receiver_optim_params"])
        #receiver_optimizer = build_agent_optimizer(model_parameters=list(receiver.parameters()) \
        #                                                             + list(object_encoder.parameters())\
        #                                                             + list(object_decoder.parameters()),
        #                                         optim_params=agent_params["receiver_optim_params"])
        p_step = agent_params["receiver_optim_params"]["p_step"]

        # Losses
        sender_loss_fn = None
        sender_imitation_loss_fn = None
        receiver_loss_fn = build_agent_loss_fn(optim_params=agent_params["receiver_optim_params"])

    else:
        raise 'Agent should be at least a Sender or a Receiver'

    agent = Agent(agent_name = agent_name,
                  object_encoder = object_encoder,
                  object_decoder= object_decoder,
                  sender=sender,
                  receiver=receiver,
                  sender_optimizer = sender_optimizer,
                  receiver_optimizer = receiver_optimizer,
                  p_step = p_step,
                  sender_loss_fn = sender_loss_fn,
                  receiver_loss_fn = receiver_loss_fn,
                  sender_imitation_loss_fn=sender_imitation_loss_fn,
                  device = device)

    return agent


def build_agent_optimizer(model_parameters : nn.Module,
                          optim_params : dict) -> th.optim.Optimizer:

    optimizers = {'adam': th.optim.Adam,
                  'sgd': th.optim.SGD,
                  'adagrad': th.optim.Adagrad}

    optimizer_name = optim_params["optimizer"]
    lr = optim_params["lr"]

    optimizer = optimizers[optimizer_name](model_parameters,lr=lr)

    return optimizer

def build_agent_loss_fn(optim_params : dict):

    if optim_params["loss"]=="cross_entropy":
        agent_loss_fn = CrossEntropyLoss(multi_attribute=False)

    elif optim_params["loss"]=="referential_loss":
        agent_loss_fn = ReferentialLoss(id_correct_object=0)

    elif optim_params["loss"]=="REINFORCE":
        agent_loss_fn = ReinforceLoss(reward_type = optim_params["reward"],
                                      baseline_type = optim_params["baseline"],
                                      entropy_reg_coef = optim_params["entropy_reg_coef"])
    elif optim_params["loss"]=="speaker_imitation":
        agent_loss_fn = SpeakerImitation()
    else:
        raise Exception("Specify a known loss")


    return agent_loss_fn
