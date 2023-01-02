import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import CosineSimilarity
import copy
import numpy as np

from .utils import find_lengths, move_to
from .senders import build_sender
from .receivers import build_receiver
from .object_encoders import build_encoder, build_decoder, build_object_projector
from .losses import ReinforceLoss, CrossEntropyLoss, ReferentialLoss, SpeakerImitation
from .language_model import get_language_model


class Agent(object):

    def __init__(self,
                 agent_name: str,
                 sender: nn.Module,
                 receiver: nn.Module,
                 object_encoder: nn.Module,
                 object_decoder: nn.Module,
                 object_projector : nn.Module,
                 language_model,
                 tasks:dict,
                 weights:dict,
                 prob_reset : float = 0.,
                 optimal_listener : str = None,
                 optimal_lm: str = None,
                 device: str = "cpu")->None:
        self.agent_name = agent_name
        self.object_encoder = object_encoder
        self.sender = sender
        self.receiver = receiver
        self.language_model = language_model
        self.object_decoder = object_decoder
        self.object_projector = object_projector
        self.optimal_listener = optimal_listener
        self.optimal_lm = optimal_lm
        self.tasks = tasks
        self.weights = weights
        self.prob_reset = prob_reset
        self.device = device

    def encode_object(self, x):
        return self.object_encoder(x)

    def send(self,
             x,
             context=None,
             return_whole_log_probs: bool = False):
        return self.sender(x, context, return_whole_log_probs)

    def get_log_prob_m_given_x(self, x, m, return_whole_log_probs=False):
        embedding = self.object_encoder(x)

        return self.sender.get_log_prob_m_given_x(embedding, m, return_whole_log_probs=return_whole_log_probs)

    def receive(self,
                message,
                context=None):
        return self.receiver(message)

    def reconstruct_from_message_embedding(self, embedding):
        return self.object_decoder(embedding)

    def sample_candidates(self,output_receiver,sampling_mode="sample"):

        """
        Sample object based on logits
        """
        # Sample candidate objects
        batch_size,n_att,n_val = output_receiver.size(0),output_receiver.size(1),output_receiver.size(2)
        output_receiver = output_receiver.reshape((batch_size*n_att,n_val))

        distr = Categorical(logits= output_receiver)

        entropy_receiver = distr.entropy()
        entropy_receiver = entropy_receiver.reshape((batch_size,n_att)).sum(1)

        if sampling_mode=="sample":
            candidates = distr.sample()
        elif sampling_mode=="greedy":
            candidates = distr.argmax(dim=1)

        log_prob_receiver = distr.log_prob(candidates)
        candidates = candidates.reshape((batch_size, n_att))
        log_prob_receiver = log_prob_receiver.reshape((batch_size, n_att)).sum(1)

        return candidates, log_prob_receiver, entropy_receiver


    def project_object(self,object):
        out = self.object_projector(object)
        m = nn.ReLU()
        out = m(out)
        return out

    def compute_referential_scores(self,
                                  message_projection: th.Tensor,
                                  object_projection: th.Tensor,
                                  n_distractors : int):

        # Expand dims across distractors axis

        #cos = CosineSimilarity(dim=1)
        cos = lambda a,b : (a*b).sum(1)

        # Target
        target_cosine = cos(message_projection, object_projection)

        batch_size = message_projection.size(0)

        # Distractors
        distractor_ids = th.stack([(i + 1 + th.multinomial(th.ones(batch_size-1),
                                                           n_distractors,
                                                           replacement=False))%batch_size
                                   for i in range(batch_size)]) # sample n_distractors != target

        distractors_projection = object_projection[distractor_ids].reshape((batch_size*n_distractors,-1))
        #message_projection_repeated = message_projection.repeat((n_distractors, 1))

        message_projection_repeated = message_projection.unsqueeze(1)
        message_projection_repeated = message_projection_repeated.repeat((1, n_distractors, 1))
        message_projection_repeated = message_projection_repeated.reshape((batch_size * n_distractors, -1))

        distractors_cosine = cos(message_projection_repeated,
                                 distractors_projection).reshape((batch_size, n_distractors)) # working

        target_and_distractors_cosine = th.cat([target_cosine.unsqueeze(1), distractors_cosine], dim=1)

        temperature = 1. # Temperature
        target_and_distractors_cosine /= temperature

        probs = th.nn.functional.softmax(target_and_distractors_cosine, dim=1)[:,0]
        loss = - th.nn.functional.log_softmax(target_and_distractors_cosine, dim=1)[:,0]
        # loss = - th.log(th.exp(target_cosine)/th.exp(distractors_cosine).sum(1))
        accuracy = 1. * (target_and_distractors_cosine.argmax(1) == 0)

        return probs, loss, accuracy

    def compute_image_reconstruction_score(self,
                                          message_projection: th.Tensor,
                                          object_projection: th.Tensor):

        # Expand dims across distractors axis

        l2_dist = lambda a,b : ((a-b)**2).sum(1)

        # Target
        l2 = l2_dist(message_projection, object_projection)

        loss = l2

        return loss

    def compute_task_losses(self, inputs, sender_log_prob, sender_entropy, messages, receiver_output,
                            neg_log_imit=None):

        return self.sender_loss_fn.compute(inputs=inputs,
                                           message=messages,
                                           sender_log_prob=sender_log_prob,
                                           sender_entropy=sender_entropy,
                                           receiver_output=receiver_output,
                                           neg_log_imit=neg_log_imit)

    def train_language_model(self, messages, threshold=1e-3):

        self.language_model.train(messages,threshold=threshold)

    def reset_parameters(self):
        if self.object_encoder is not None: self.object_encoder.reset_parameters()
        if self.sender is not None: self.sender.reset_parameters()
        if self.receiver is not None: self.receiver.reset_parameters()
        if self.object_decoder is not None: self.object_decoder.reset_parameters()
        if self.object_projector is not None: self.object_projector.reset_parameters()

    def weight_noise(self):

        alpha=0.01

        if self.object_decoder is not None:
            reset_weights = self.object_decoder.state_dict().copy()

            for el in self.object_decoder.state_dict():
                if len(self.object_decoder.state_dict()[el].size()) > 1:
                    w = th.empty(self.object_decoder.state_dict()[el].size(),device=self.device)

                    reset_weights[el] = self.object_decoder.state_dict()[el] + \
                                        alpha*nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

            self.object_decoder.load_state_dict(reset_weights)

        if self.object_encoder is not None:
            reset_weights = self.object_encoder.state_dict().copy()

            for el in self.object_encoder.state_dict():

                if len(self.object_encoder.state_dict()[el].size()) > 1:
                    w = th.empty(self.object_encoder.state_dict()[el].size(),device=self.device)

                    reset_weights[el] = self.object_encoder.state_dict()[el] + \
                                        alpha*nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

            self.object_encoder.load_state_dict(reset_weights)

        if self.sender is not None:
            reset_weights = self.sender.state_dict().copy()

            for el in self.sender.state_dict():

                if len(self.sender.state_dict()[el].size()) > 1:
                    w = th.empty(self.sender.state_dict()[el].size(),device=self.device)

                    reset_weights[el] = self.sender.state_dict()[el] + \
                                        alpha*nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

            self.sender.load_state_dict(reset_weights)

        if self.receiver is not None:

            reset_weights = self.receiver.state_dict().copy()

            for el in self.receiver.state_dict():

                if len(self.receiver.state_dict()[el].size())>1:
                    w = th.empty(self.receiver.state_dict()[el].size(),device=self.device)

                    reset_weights[el] = self.receiver.state_dict()[el] + \
                                        alpha*nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')

            self.receiver.load_state_dict(reset_weights)


    def random_reset(self,reset_level: float = 0.5):

        if self.object_encoder is not None:
            reset_weights = self.object_encoder.state_dict().copy()

            for el in self.object_encoder.state_dict():
                probs = th.rand(size=self.object_encoder.state_dict()[el].size(),
                                device=self.object_encoder.state_dict()[el].device)
                M = 1 * (probs < reset_level)
                #reset_weights[el] = (1 - M) * self.object_encoder.state_dict()[el] + \
                ##                    M * th.normal(size=reset_weights[el].size(), mean=self.sender.state_dict()[el].mean().item(), std=1.,
                 #                                 device=self.object_encoder.state_dict()[el].device)
                reset_weights[el] = (1 - M) * self.object_encoder.state_dict()[el] + \
                                    M * th.normal(size=reset_weights[el].size(), mean=0.0, std=1.,
                                                  device=self.object_encoder.state_dict()[el].device)

            self.object_encoder.load_state_dict(reset_weights)

        #if self.object_decoder is not None:
        #    reset_weights = self.object_decoder.state_dict().copy()

        #    for el in self.object_decoder.state_dict():
        #        probs = th.rand(size=self.object_decoder.state_dict()[el].size(),
        #                        device=self.object_decoder.state_dict()[el].device)
        #        M = 1 * (probs < reset_level)
        #        reset_weights[el] = (1 - M) * self.object_decoder.state_dict()[el] + \
        #                            M * th.normal(size=reset_weights[el].size(), mean=0.0, std=1.,
        #                                          device=self.object_decoder.state_dict()[el].device)

            self.object_decoder.load_state_dict(reset_weights)

        if self.sender is not None:
            reset_weights = self.sender.state_dict().copy()

            for el in self.sender.state_dict():
                probs = th.rand(size=self.sender.state_dict()[el].size(),
                                device=self.sender.state_dict()[el].device)
                M = 1 * (probs < reset_level)
                #reset_weights[el] = (1 - M) * self.sender.state_dict()[el] + \
                #                    M * th.normal(size=reset_weights[el].size(), mean=self.sender.state_dict()[el].mean().item(), std=1.,
                                    #M * th.normal(size=reset_weights[el].size(), mean=0.0, std=1.,
                #                                  device=self.sender.state_dict()[el].device)
                reset_weights[el] = (1 - M) * self.sender.state_dict()[el] + \
                                    M * th.normal(size=reset_weights[el].size(), mean=0.0, std=1.,
                                                  device=self.sender.state_dict()[el].device)

            self.sender.load_state_dict(reset_weights)

        if self.receiver is not None:
            reset_weights = self.receiver.state_dict().copy()

            for el in self.receiver.state_dict():
                probs = th.rand(size=self.receiver.state_dict()[el].size(),
                                device=self.receiver.state_dict()[el].device)
                M = 1 * (probs < reset_level)
                #reset_weights[el] = (1 - M) * self.receiver.state_dict()[el] + \
                #                    M * th.normal(size=reset_weights[el].size(), mean=self.receiver.state_dict()[el].mean().item(), std=1.,
                                    #M * th.normal(size=reset_weights[el].size(), mean=0.0, std=1.,
                #                                  device=self.receiver.state_dict()[el].device)

                reset_weights[el] = (1 - M) * self.receiver.state_dict()[el] + \
                                    M * th.normal(size=reset_weights[el].size(), mean=0.0, std=1.,
                                                  device=self.receiver.state_dict()[el].device)

            self.receiver.load_state_dict(reset_weights)



def get_agent(agent_name: str,
              agent_repertory: dict,
              game_params: dict,
              pretrained_modules : dict = dict(),
              device: str = "cpu") -> Agent:

    agent_params = agent_repertory[agent_name]

    if not len(pretrained_modules) and "pretrained_modules" in agent_params:
        pretrained_modules = {k : th.load(path) for k,path in agent_params["pretrained_modules"].items()}
        load_state_dict_cond=True
    elif len(pretrained_modules):
        load_state_dict_cond=False
    else:
        load_state_dict_cond=True

    if agent_params["sender"] and agent_params["receiver"]:

        assert agent_params["receiver_params"]["receiver_embed_dim"] \
               == agent_params["sender_params"]["sender_embed_dim"]

        raise NotImplementedError  # Should take into account lots of agents modellings

    elif agent_params["sender"] and not agent_params["receiver"]:

        # Models
        object_encoder = build_encoder(object_params=game_params["objects"],
                                       embedding_size=agent_params["sender_params"]["sender_embed_dim"])
        sender = build_sender(sender_params=agent_params["sender_params"], game_params=game_params)
        if "language_model" in agent_params:
            language_model = get_language_model(lm_params = agent_repertory[agent_params["language_model"]],
                                                game_params=game_params,
                                                device=device)
        else:
            language_model = None

        object_decoder = None
        object_projector = None
        receiver = None

        # Pretrained modules
        if "object_encoder" in pretrained_modules and load_state_dict_cond:
            object_encoder.load_state_dict(pretrained_modules["object_encoder"])
        elif "object_encoder" in pretrained_modules and not load_state_dict_cond:
            object_encoder = pretrained_modules["object_encoder"]
        if "sender" in pretrained_modules and load_state_dict_cond:
            sender.load_state_dict(pretrained_modules["sender"])
        elif "sender" in pretrained_modules and not load_state_dict_cond:
            sender = pretrained_modules["sender"]
        if "language_model" in pretrained_modules and load_state_dict_cond:
            language_model.load_state_dict(pretrained_modules["language_model"])
        elif "language_model" in pretrained_modules and not load_state_dict_cond:
            language_model = pretrained_modules["language_model"]


        # Send models to device
        sender.to(device)
        object_encoder.to(device)

        # Model parameters
        model_parameters = list(object_encoder.parameters()) + list(sender.parameters())

    elif not agent_params["sender"] and agent_params["receiver"]:

        # Models
        if game_params["game_type"]=="referential" or game_params["game_type"] == "visual_reconstruction":
            object_encoder = build_encoder(object_params=game_params["objects"],
                                           embedding_size=agent_params["receiver_params"]["receiver_embed_dim"])
            object_decoder = build_decoder(object_params=game_params["objects"],
                                           embedding_size=agent_params["receiver_params"]["receiver_embed_dim"],
                                           projection_size=agent_params["receiver_params"]["projection_dim"])
            object_projector = build_object_projector(object_params=game_params["objects"],
                                                      projection_size=agent_params["receiver_params"]["projection_dim"])
        else:
            object_encoder = build_encoder(object_params=game_params["objects"],
                                           embedding_size=agent_params["receiver_params"]["receiver_embed_dim"])
            object_decoder = build_decoder(object_params=game_params["objects"],
                                           embedding_size=agent_params["receiver_params"]["receiver_embed_dim"])
            object_projector = None

        sender = None
        language_model = None
        receiver = build_receiver(receiver_params=agent_params["receiver_params"], game_params=game_params)

        # Pretrained modules
        if "object_encoder" in pretrained_modules and load_state_dict_cond:
            object_encoder.load_state_dict(pretrained_modules["object_encoder"])
        elif "object_encoder" in pretrained_modules and not load_state_dict_cond:
            object_encoder = pretrained_modules["object_encoder"]
        if "object_decoder" in pretrained_modules and load_state_dict_cond:
            object_decoder.load_state_dict(pretrained_modules["object_decoder"])
        elif "object_decoder" in pretrained_modules and not load_state_dict_cond:
            object_decoder = pretrained_modules["object_decoder"]
        if "receiver" in pretrained_modules and load_state_dict_cond:
            receiver.load_state_dict(pretrained_modules["receiver"])
        elif "receiver" in pretrained_modules and not load_state_dict_cond:
            receiver = pretrained_modules["receiver"]
        if "object_projector" in pretrained_modules and load_state_dict_cond:
            object_projector.load_state_dict(pretrained_modules["object_projector"])
        elif "object_projector" in pretrained_modules and not load_state_dict_cond:
            object_projector = pretrained_modules["object_projector"]

        # Send models to device
        receiver.to(device)
        object_encoder.to(device)
        object_decoder.to(device)
        if object_projector is not None:
            object_projector.to(device)

        # Model parameters
        if object_projector is not None:
            model_parameters = list(receiver.parameters()) + list(object_decoder.parameters()) + \
                               list(object_projector.parameters())
        else:
            model_parameters = list(receiver.parameters()) + list(object_decoder.parameters())

    else:
        raise 'Agent should be at least a Sender or a Receiver'

    # Tasks
    tasks = {}

    for task, task_infos in agent_params["tasks"].items():
        loss = get_loss(loss_infos=task_infos["loss"])

        if "weight_decay" in task_infos:
            weight_decay = task_infos["weight_decay"]
        else:
            weight_decay=0.

        optimizer = get_optimizer(model_parameters=model_parameters,
                                  optimizer_name=task_infos["optimizer"],
                                  weight_decay=weight_decay,
                                  lr=task_infos["lr"])

        p_step = task_infos["p_step"]

        tasks[task] = {"loss": loss, "optimizer": optimizer, "p_step": p_step}

        if 'lm_mode' in task_infos:
            tasks[task]["lm_mode"]=task_infos["lm_mode"]

    if "optimal_listener" in agent_params:
        optimal_listener = agent_params["optimal_listener"]
    else:
        optimal_listener = None

    if "optimal_lm" in agent_params:
        optimal_lm = agent_params["optimal_lm"]
    else:
        optimal_lm = None

    if "weights" in agent_params:
        weights = agent_params["weights"]
    else:
        weights = None

    if "prob_reset" in agent_params:
        prob_reset = agent_params["prob_reset"]
    else:
        prob_reset=0

    agent = Agent(agent_name=agent_name,
                  object_encoder=object_encoder,
                  object_decoder=object_decoder,
                  object_projector = object_projector,
                  sender=sender,
                  receiver=receiver,
                  language_model = language_model,
                  tasks=tasks,
                  weights = weights,
                  optimal_listener = optimal_listener,
                  optimal_lm = optimal_lm,
                  prob_reset = prob_reset,
                  device=device)

    return agent


def get_optimizer(model_parameters: list,
                  optimizer_name: str,
                  lr: float,
                  weight_decay : float = 0.):
    optimizers = {'adam': th.optim.Adam,
                  'sgd': th.optim.SGD,
                  'adagrad': th.optim.Adagrad}

    #optimizer = optimizers[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay)
    optimizer = optimizers[optimizer_name](filter(lambda p: p.requires_grad,model_parameters),
                                           lr=lr,
                                           weight_decay=weight_decay)

    return optimizer


def get_loss(loss_infos: dict):
    if loss_infos["type"] == "cross_entropy":
        agent_loss_fn = CrossEntropyLoss(multi_attribute=False)

    elif loss_infos["type"] == "REINFORCE":
        agent_loss_fn = ReinforceLoss(reward_type=loss_infos["reward"],
                                      baseline_type=loss_infos["baseline"],
                                      entropy_reg_coef=loss_infos["entropy_reg_coef"])
    elif loss_infos["type"] == "speaker_imitation":
        agent_loss_fn = SpeakerImitation()
    else:
        raise Exception("Specify a known loss")

    return agent_loss_fn

def copy_agent(agent : Agent,
               agent_repertory: dict,
               game_params: dict,
               device: str = "cpu") -> Agent:

    pretrained_modules = dict()

    if agent.object_encoder is not None:
        pretrained_modules["object_encoder"] = copy.deepcopy(agent.object_encoder)
    if agent.sender is not None:
        pretrained_modules["sender"] = copy.deepcopy(agent.sender)
    if agent.receiver is not None:
        pretrained_modules["receiver"] = copy.deepcopy(agent.receiver)
    if agent.language_model is not None:
        pretrained_modules["language_model"] = copy.deepcopy(agent.language_model)
    if agent.object_decoder is not None:
        pretrained_modules["object_decoder"] = copy.deepcopy(agent.object_decoder)

    agent_copy = get_agent(agent_name=agent.agent_name,
                          agent_repertory=agent_repertory,
                          game_params=game_params,
                          pretrained_modules=pretrained_modules,
                          device=device)

    return agent_copy