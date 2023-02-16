import torch as th
import copy
import numpy as np
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from .utils import _div_dict, move_to
from .agents import get_agent
from .receivers import build_receiver
from .object_encoders import build_decoder


class TrainerPopulation(object):

    def __init__(self,
                 game: object,
                 evaluator: object,
                 game_params: dict,
                 agent_repertory: dict,
                 train_loader: th.utils.data.DataLoader,
                 imitation_loader: th.utils.data.DataLoader = None,
                 mi_loader: th.utils.data.DataLoader = None,
                 val_loader: th.utils.data.DataLoader = None,
                 test_loader: th.utils.data.DataLoader = None,
                 logger: th.utils.tensorboard.SummaryWriter = None,
                 metrics_save_dir: str = "",
                 models_save_dir : str = "",
                 device: str = "cpu") -> None:

        self.game = game
        self.population = game.population
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.imitation_loader = imitation_loader
        self.game_params = game_params
        self.agent_repertory = agent_repertory
        self.mi_loader = mi_loader
        self.evaluator = evaluator
        self.start_epoch = 1
        self.device = th.device(device)
        self.writer = logger
        self.metrics_save_dir = metrics_save_dir
        self.models_save_dir = models_save_dir

    def train(self,
              n_epochs,
              train_communication_freq: int = 1000000,
              validation_freq: int = 1,
              train_imitation_freq: int = 100000,
              evaluator_freq: int = 1000000,
              save_models_freq : int = 100000,
              reset_agents_freq : int = 1000000,
              print_evolution: bool = True):

        for epoch in range(self.start_epoch, n_epochs):

            if print_evolution: print(f"Epoch {epoch}")
            
            # Reset agents
            if epoch % reset_agents_freq == 0 :
                self.reset_agents()

            # Train communication
            if epoch % train_communication_freq == 0:
                train_communication_loss_senders, train_communication_loss_receivers, train_metrics = \
                    self.train_communication(compute_metrics=True)  # dict,dict, dict
            else:
                train_communication_loss_senders, train_communication_loss_receivers, train_metrics = None, None, None

            # Train imitation
            if epoch % train_imitation_freq == 0:
                train_imitation_loss_senders, train_imitation_loss_imitators = \
                    self.train_imitation()  # dict,dict, dict
            else:
                train_imitation_loss_senders, train_imitation_loss_imitators = None, None

            # Validation
            if self.val_loader is not None and epoch % validation_freq == 0:
                val_loss_senders, val_loss_receivers, val_metrics = self.eval(compute_metrics=True)
            else:
                val_loss_senders, val_loss_receivers, val_metrics = None, None, None

            if self.writer is not None:
                self.log_metrics(epoch=epoch,
                                 train_communication_loss_senders=train_communication_loss_senders,
                                 train_communication_loss_receivers=train_communication_loss_receivers,
                                 train_imitation_loss_senders=train_imitation_loss_senders,
                                 train_imitation_loss_imitators=train_imitation_loss_imitators,
                                 train_metrics=train_metrics,
                                 val_loss_senders=val_loss_senders,
                                 val_loss_receivers=val_loss_receivers,
                                 val_metrics=val_metrics)

            if len(self.models_save_dir) and epoch % save_models_freq == 0:
                self.save_models(epoch=epoch)

            if self.evaluator is not None and epoch % evaluator_freq == 0:
                self.evaluator.step(epoch=epoch)

    def reset_optimizer(self):

        for agent_id in self.population.receiver_names:
            agent = self.population.agents[agent_id]
            model_parameters = list(agent.receiver.parameters()) + list(agent.object_decoder.parameters()) + \
                               list(agent.object_projector.parameters())

            agent.tasks["communication"]["optimizer"] = th.optim.Adam(model_parameters,
                                                                      lr=agent.tasks["communication"]["lr"])
            
        for agent_id in self.population.sender_names:
            agent = self.population.agents[agent_id]
            model_parameters = list(agent.sender.parameters()) + list(agent.object_encoder.parameters())

            agent.tasks["communication"]["optimizer"] = th.optim.Adam(model_parameters,
                                                                      lr=agent.tasks["communication"]["lr"])

    def reset_agents(self):

        for agent_id in self.population.sender_names+self.population.receiver_names:
            agent = self.population.agents[agent_id]
            if th.rand(1)[0] < agent.prob_reset:
                self.population.agents[agent_id] = get_agent(agent_name=agent_id,
                                                             agent_repertory=self.agent_repertory,
                                                             game_params=self.game_params,
                                                             device=self.device)

    def train_communication(self, compute_metrics: bool = False):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.train()

        for batch in self.train_loader:

            sender_id, receiver_id = batch.sender_id, batch.receiver_id
            agent_sender = self.population.agents[sender_id]
            agent_receiver = self.population.agents[receiver_id]

            task = "communication"

            if sender_id not in mean_loss_senders:
                mean_loss_senders[sender_id] = {task: 0.}
                n_batches[sender_id] = {task: 0}
            if receiver_id not in mean_loss_receivers:
                mean_loss_receivers[receiver_id] = {task: 0.}
                n_batches[receiver_id] = {task: 0}

            batch = move_to(batch, self.device)

            metrics = self.game(batch, compute_metrics=compute_metrics)

            # Sender
            if th.rand(1)[0] < agent_sender.tasks[task]["p_step"]:
                agent_sender.tasks[task]["optimizer"].zero_grad()
                agent_sender.tasks[task]["loss_value"].backward()
                agent_sender.tasks[task]["optimizer"].step()

            mean_loss_senders[sender_id][task] += agent_sender.tasks[task]["loss_value"].item()
            n_batches[sender_id][task] += 1

            # Receiver
            if th.rand(1)[0] < agent_receiver.tasks[task]["p_step"]:
                agent_receiver.tasks[task]["optimizer"].zero_grad()
                agent_receiver.tasks[task]["loss_value"].backward()
                agent_receiver.tasks[task]["optimizer"].step()

            mean_loss_receivers[receiver_id][task] += agent_receiver.tasks[task]["loss_value"].item()
            n_batches[receiver_id][task] += 1

            if compute_metrics:
                # Store metrics
                if sender_id not in mean_metrics:
                    mean_metrics[sender_id] = {"accuracy": 0.,
                                               "accuracy_tot": 0.,
                                               "sender_entropy": 0.,
                                               "sender_log_prob": 0.,
                                               "message_length": 0.}
                if receiver_id not in mean_metrics:
                    mean_metrics[receiver_id] = {"accuracy": 0.,"accuracy_tot":0.,"entropy":0.}

                mean_metrics[sender_id]["accuracy"] += metrics["accuracy"]
                mean_metrics[sender_id]["accuracy_tot"] += metrics["accuracy_tot"]
                mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"]
                mean_metrics[receiver_id]["accuracy_tot"] += metrics["accuracy_tot"]
                mean_metrics[receiver_id]["entropy"] += metrics["entropy_receiver"]

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                               for receiver_id in mean_loss_receivers}

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt][task])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def train_communication_broadcasting(self, compute_metrics=True):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.train()

        for batch in self.train_loader:

            sender_id, receiver_ids = batch.sender_id, batch.receiver_ids
            agent_sender = self.population.agents[sender_id]

            weights = {}
            for receiver_id in receiver_ids:
                weights[receiver_id] = 1

            task = "communication"

            if sender_id not in mean_loss_senders:
                mean_loss_senders[sender_id] = {task: 0.}
                n_batches[sender_id] = {task: 0}

            for receiver_id in receiver_ids:
                if receiver_id not in mean_loss_receivers:
                    mean_loss_receivers[receiver_id] = {task: 0.}
                    n_batches[receiver_id] = {task: 0}

            batch = move_to((batch.data, sender_id, receiver_ids, weights), self.device)

            metrics = self.game.communication_multi_listener_instance(*batch, compute_metrics=compute_metrics)

            # Sender
            if th.rand(1)[0] < agent_sender.tasks[task]["p_step"]:
                agent_sender.tasks[task]["optimizer"].zero_grad()
                agent_sender.tasks[task]["loss_value"].backward()
                agent_sender.tasks[task]["optimizer"].step()

            if compute_metrics:
                # Store metrics
                if sender_id not in mean_metrics:
                    mean_metrics[sender_id] = {"accuracy": 0.,
                                               "accuracy_tot":0.,
                                               "sender_entropy": 0.,
                                               "sender_log_prob": 0.,
                                               "message_length": 0.}

            # Receiver
            for receiver_id in receiver_ids:
                agent_receiver = self.population.agents[receiver_id]

                if th.rand(1)[0] < agent_receiver.tasks[task]["p_step"]:
                    agent_receiver.tasks[task]["optimizer"].zero_grad()
                    agent_receiver.tasks[task]["loss_value"].backward()
                    agent_receiver.tasks[task]["optimizer"].step()

                mean_loss_receivers[receiver_id][task] += agent_receiver.tasks[task]["loss_value"].item()
                n_batches[receiver_id][task] += 1

                mean_loss_senders[sender_id][task] += agent_sender.tasks[task]["loss_value"].item()
                n_batches[sender_id][task] += 1

                if compute_metrics:

                    if receiver_id not in mean_metrics:
                        mean_metrics[receiver_id] = {"accuracy": 0.,"accuracy_tot":0.}

                    mean_metrics[sender_id]["accuracy"] += metrics["accuracy"][receiver_id]
                    mean_metrics[sender_id]["accuracy_tot"] += metrics["accuracy_tot"][receiver_id]
                    mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                    mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                    mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                    mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"][receiver_id]
                    mean_metrics[receiver_id]["accuracy_tot"] += metrics["accuracy_tot"][receiver_id]

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                               for receiver_id in mean_loss_receivers}

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt][task])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def eval(self, compute_metrics: bool = False):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.eval()

        with th.no_grad():

            for batch in self.test_loader:

                sender_id, receiver_id = batch.sender_id, batch.receiver_id
                agent_sender, agent_receiver = self.population.agents[sender_id], self.population.agents[receiver_id]

                if sender_id not in mean_loss_senders:
                    mean_loss_senders[sender_id] = {"communication": 0.}
                    n_batches[sender_id] = {"communication": 0}
                if receiver_id not in mean_loss_receivers:
                    mean_loss_receivers[receiver_id] = {"communication": 0.}
                    n_batches[receiver_id] = {"communication": 0}

                batch = move_to(batch, self.device)
                metrics = self.game(batch, compute_metrics=compute_metrics)

                task = "communication"
                mean_loss_senders[sender_id][task] += agent_sender.tasks[task]["loss_value"]
                n_batches[sender_id][task] += 1
                mean_loss_receivers[receiver_id][task] += agent_receiver.tasks[task]["loss_value"]
                n_batches[receiver_id][task] += 1

                if compute_metrics:
                    # Store metrics
                    if sender_id not in mean_metrics:
                        mean_metrics[sender_id] = {"accuracy": 0.,
                                                   "accuracy_tot" : 0.,
                                                   "sender_entropy": 0.,
                                                   "message_length": 0.}
                    if receiver_id not in mean_metrics:
                        mean_metrics[receiver_id] = {"accuracy": 0.,"accuracy_tot":0.,"entropy":0.}
                    mean_metrics[sender_id]["accuracy"] += metrics["accuracy"]
                    mean_metrics[sender_id]["accuracy_tot"] += metrics["accuracy_tot"]
                    mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                    mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                    mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"]
                    mean_metrics[receiver_id]["accuracy_tot"] += metrics["accuracy_tot"]
                    mean_metrics[receiver_id]["entropy"] += metrics["entropy_receiver"]

            mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                                 for sender_id in mean_loss_senders}
            mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                                   for receiver_id in mean_loss_receivers}

            if compute_metrics:
                for agt in mean_metrics:
                    mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt]["communication"])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def log_metrics(self,
                    epoch: int,
                    train_communication_loss_senders: dict = None,
                    train_communication_loss_receivers: dict = None,
                    train_imitation_loss_senders: dict = None,
                    train_imitation_loss_imitators: dict = None,
                    train_communication_mi_loss_senders: dict = None,
                    train_metrics: dict = None,
                    val_loss_senders: dict = None,
                    val_loss_receivers: dict = None,
                    val_metrics: dict = None) -> None:

        # Train
        if train_communication_loss_senders is not None:
            for sender, tasks in train_communication_loss_senders.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{sender}/{task} (train)', l, epoch)

                    if task == "communication":
                        self.writer.add_scalar(f'{sender}/accuracy (train)',
                                               train_metrics[sender]['accuracy'], epoch)
                        self.writer.add_scalar(f'{sender}/accuracy tot (train)',
                                               train_metrics[sender]['accuracy_tot'], epoch)
                        self.writer.add_scalar(f'{sender}/Language entropy (train)',
                                               train_metrics[sender]['sender_entropy'], epoch)
                        self.writer.add_scalar(f'{sender}/Sender log prob',
                                               train_metrics[sender]['sender_log_prob'], epoch)
                        self.writer.add_scalar(f'{sender}/Messages length (train)',
                                               train_metrics[sender]['message_length'], epoch)

        if train_communication_loss_receivers is not None:

            for receiver, tasks in train_communication_loss_receivers.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{receiver}/{task}_train', l, epoch)

                    if task == "communication":
                        self.writer.add_scalar(f'{receiver}/accuracy (train)',
                                               train_metrics[receiver]['accuracy'], epoch)
                        self.writer.add_scalar(f'{receiver}/accuracy tot (train)',
                                               train_metrics[receiver]['accuracy_tot'], epoch)
                        if "entropy" in train_metrics[receiver]:
                            self.writer.add_scalar(f'{receiver}/entropy (train)',
                                                train_metrics[receiver]['entropy'], epoch)

        # Imitation
        if train_imitation_loss_senders is not None:
            for sender, tasks in train_imitation_loss_senders.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{sender}/{task} (train)', l, epoch)

        if train_imitation_loss_imitators is not None:
            for sender, tasks in train_imitation_loss_imitators.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{sender}/{task} (train)', l, epoch)

        # MI
        if train_communication_mi_loss_senders is not None:
            for sender, tasks in train_communication_mi_loss_senders.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{sender}/{task} (train)', l, epoch)

                    if task == "communication":
                        self.writer.add_scalar(f'{sender}/accuracy (train)',
                                               train_metrics[sender]['accuracy'], epoch)
                        self.writer.add_scalar(f'{sender}/accuracy tot (train)',
                                               train_metrics[sender]['accuracy_tot'], epoch)
                        self.writer.add_scalar(f'{sender}/Language entropy (train)',
                                               train_metrics[sender]['sender_entropy'], epoch)
                        self.writer.add_scalar(f'{sender}/Sender log prob',
                                               train_metrics[sender]['sender_log_prob'], epoch)
                        self.writer.add_scalar(f'{sender}/Messages length (train)',
                                               train_metrics[sender]['message_length'], epoch)

        # Val
        if val_loss_senders is not None:
            for sender, tasks in val_loss_senders.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{sender}/Loss val', l, epoch)

                self.writer.add_scalar(f'{sender}/accuracy (val)',
                                       val_metrics[sender]['accuracy'], epoch)
                self.writer.add_scalar(f'{sender}/accuracy tot (val)',
                                       val_metrics[sender]['accuracy_tot'], epoch)
                self.writer.add_scalar(f'{sender}/Language entropy (val)',
                                       val_metrics[sender]['sender_entropy'], epoch)
                self.writer.add_scalar(f'{sender}/Messages length (val)',
                                       val_metrics[sender]['message_length'], epoch)

            for receiver, tasks in val_loss_receivers.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{receiver}/Loss val', l, epoch)

                self.writer.add_scalar(f'{receiver}/accuracy (val)',
                                       val_metrics[receiver]['accuracy'], epoch)
                self.writer.add_scalar(f'{receiver}/accuracy tot (val)',
                                       val_metrics[receiver]['accuracy_tot'], epoch)
                if "entropy" in val_metrics[receiver]:
                    self.writer.add_scalar(f'{receiver}/entropy (val)',
                                           val_metrics[receiver]['entropy'], epoch)

    def save_models(self,epoch):
        self.population.save_models(save_dir=self.models_save_dir,
                                    add_info=str(epoch))


class PretrainingTrainer:

    def __init__(self,
                 game,
                 train_loader: th.utils.data.DataLoader,
                 log_dir: str = "",
                 compute_metrics: bool = False,
                 print_evolution: bool = True,
                 device: str = "cpu") -> None:

        self.game = game
        self.agent = game.agent
        self.train_loader = train_loader
        self.start_epoch = 0
        self.device = device
        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.compute_metrics = compute_metrics
        self.print_evolution = print_evolution

    def train(self, n_epochs):

        for epoch in range(self.start_epoch, n_epochs):

            if self.print_evolution: print(f"Epoch {epoch}")

            train_loss, metrics = \
                self.train_epoch(compute_metrics=self.compute_metrics)  # float, float

            if self.writer is not None:
                self.writer.add_scalar(f'Loss train', train_loss.item(), epoch)

                if self.compute_metrics:
                    self.writer.add_scalar(f'Language entropy (train)',
                                           metrics['sender_entropy'],
                                           epoch)
                    self.writer.add_scalar(f'Messages length (train)',
                                           metrics['message_length'],
                                           epoch)

    def train_epoch(self, compute_metrics: bool = False):
        mean_loss = 0.
        n_batches = 0
        mean_metrics = {"sender_entropy": 0., "message_length": 0.}

        self.game.train()
        for batch in self.train_loader:

            batch = move_to(batch, self.device)
            loss_sender, metrics = self.game(batch, compute_metrics=compute_metrics)

            self.agent.tasks["communication"]["optimizer"].zero_grad()
            loss_sender.backward()
            self.agent.tasks["communication"]["optimizer"].step()

            # Store losses
            mean_loss += loss_sender
            n_batches += 1

            if compute_metrics:
                # Store metrics
                mean_metrics["sender_entropy"] += metrics["sender_entropy"]
                mean_metrics["message_length"] += metrics["message_length"]

        mean_loss /= n_batches

        if compute_metrics:
            mean_metrics = _div_dict(mean_metrics, n_batches)

        return mean_loss, mean_metrics


class TrainerCustom(TrainerPopulation):

    def __init__(self,
                 game: object,
                 evaluator: object,
                 game_params: dict,
                 agent_repertory: dict,
                 train_loader: th.utils.data.DataLoader,
                 imitation_loader: th.utils.data.DataLoader = None,
                 mi_loader: th.utils.data.DataLoader = None,
                 val_loader: th.utils.data.DataLoader = None,
                 test_loader: th.utils.data.DataLoader = None,
                 logger: th.utils.tensorboard.SummaryWriter = None,
                 metrics_save_dir: str = "",
                 models_save_dir : str = "",
                 device: str = "cpu") -> None:

        self.mi_step = 0
        self.val_loss_optimal_listener = []
        self.step_without_opt_training = 0
        self.reward_distrib = []
        self.mi_distrib = []
        self.input_batch = []
        self.pi_x_m = []

        super().__init__(game=game,
                         evaluator=evaluator,
                         game_params=game_params,
                         agent_repertory=agent_repertory,
                         train_loader=train_loader,
                         imitation_loader=imitation_loader,
                         mi_loader=mi_loader,
                         val_loader=val_loader,
                         test_loader=test_loader,
                         logger=logger,
                         metrics_save_dir=metrics_save_dir,
                         models_save_dir=models_save_dir,
                         device=device)

    def train(self,
              n_epochs,
              train_communication_freq: int = 1000000,
              validation_freq: int = 1,
              train_imitation_freq: int = 100000,
              train_custom_freq: int = 100000,
              train_broadcasting_freq: int = 1000000,
              train_communication_and_mi_freq : int = 1000000,
              train_comm_and_check_gradient : int = 10000000,
              reset_agents_freq : int = 1000000,
              train_kl_freq : int = 10000000,
              evaluator_freq: int = 1000000,
              save_models_freq : int = 10000000,
              print_evolution: bool = True,
              custom_steps: int = 0,
              max_steps: int = 1000000,
              custom_early_stopping: bool = False):

        for epoch in range(self.start_epoch, n_epochs):

            if print_evolution: print(f"Epoch {epoch}")

            # Reset agents
            if epoch % reset_agents_freq == 0:
                self.reset_agents()

            # Train communication
            if epoch % train_communication_freq == 0:
                train_communication_loss_senders, train_communication_loss_receivers, train_metrics = \
                    self.train_communication(compute_metrics=True)  # dict,dict, dict
            else:
                train_communication_loss_senders, train_communication_loss_receivers, train_metrics = None, None, None

            # Train imitation
            if epoch % train_imitation_freq == 0:
                train_imitation_loss_senders, train_imitation_loss_imitators = \
                    self.train_imitation()  # dict,dict, dict
            else:
                train_imitation_loss_senders, train_imitation_loss_imitators = None, None

            # Train Mutual information
            if epoch % train_custom_freq == 0:

                self.reset_agents()
                #self.weigth_noise_listener()

                #self.pretrain_optimal_listener(epoch=epoch)

                self.custom_train_communication(epoch=epoch,
                                                custom_steps=custom_steps,
                                                max_steps=max_steps,
                                                early_stopping=custom_early_stopping)

                #self.save_error(epoch=epoch, save=False)

                #train_communication_mi_loss_senders, train_communication_loss_receivers, train_metrics = \
                #    self.train_communication_and_mutual_information()
                train_communication_loss_senders, train_communication_loss_receivers, train_metrics = \
                    self.train_communication(compute_metrics=True)
                train_communication_mi_loss_senders = None
                    #train_communication_mi_loss_senders, train_communication_loss_receivers, train_metrics = \
                #    self.train_communication_and_kl()

                #if epoch % 100 == 0:
                #    self.save_error(epoch=epoch, save=True)
                #else:
                #    self.save_error(epoch=epoch, save=False)

            if epoch % train_comm_and_check_gradient == 0:
                self.pretrain_optimal_listener(epoch=epoch)
                train_communication_mi_loss_senders, train_communication_loss_receivers, \
                mean_gradient_tot_senders, mean_gradient_fun_senders, mean_gradient_coo_senders,ps_gradient_senders, \
                train_metrics = \
                self.train_communication_and_keep_gradients(compute_metrics=True)

                for sender_id in mean_gradient_tot_senders:
                    self.writer.add_scalar(f'{sender_id}/grad_tot',
                                           mean_gradient_tot_senders[sender_id]["communication"], epoch)
                    self.writer.add_scalar(f'{sender_id}/grad_fun',
                                           mean_gradient_fun_senders[sender_id]["communication"], epoch)
                    self.writer.add_scalar(f'{sender_id}/grad_coo',
                                           mean_gradient_coo_senders[sender_id]["communication"], epoch)
                    self.writer.add_scalar(f'{sender_id}/ps_gradient',
                                           ps_gradient_senders[sender_id]["communication"], epoch)


            if epoch % train_communication_and_mi_freq == 0:
                self.pretrain_optimal_listener(epoch=epoch)
                train_communication_mi_loss_senders, train_communication_loss_receivers, train_metrics = \
                   self.train_communication_and_mutual_information()
            else:
                train_communication_mi_loss_senders = None


            if epoch % train_broadcasting_freq == 0:
                train_communication_loss_senders, train_communication_loss_receivers, train_metrics = \
                    self.train_communication_broadcasting(compute_metrics=True)  # dict,dict, dict

            # Train KL div
            if epoch % train_kl_freq == 0:
                self.pretrain_language_model(epoch=epoch)
                train_communication_mi_loss_senders, train_communication_loss_receivers, train_metrics = \
                    self.train_communication_and_kl()

            # Validation
            if self.val_loader is not None and epoch % validation_freq == 0:
                val_loss_senders, val_loss_receivers, val_metrics = self.eval(compute_metrics=True)
            else:
                val_loss_senders, val_loss_receivers, val_metrics = None, None, None

            if self.writer is not None:
                self.log_metrics(epoch=epoch,
                                 train_communication_loss_senders=train_communication_loss_senders,
                                 train_communication_loss_receivers=train_communication_loss_receivers,
                                 train_imitation_loss_senders=train_imitation_loss_senders,
                                 train_imitation_loss_imitators=train_imitation_loss_imitators,
                                 train_communication_mi_loss_senders=train_communication_mi_loss_senders,
                                 train_metrics=train_metrics,
                                 val_loss_senders=val_loss_senders,
                                 val_loss_receivers=val_loss_receivers,
                                 val_metrics=val_metrics)

            if len(self.models_save_dir) and epoch % save_models_freq == 0:
                self.save_models(epoch=epoch)

            if self.evaluator is not None and epoch % evaluator_freq == 0:
                self.evaluator.step(epoch=epoch)

    def custom_train_communication(self,
                                   compute_metrics: bool = False,
                                   early_stopping: bool = False,
                                   epoch: int = 0,
                                   custom_steps: int = 0,
                                   max_steps : int = 1000000):
        task = "communication"

        if early_stopping:

            continue_training = True
            early_stop_step = 0
            val_losses = []

            while continue_training:

                self.game.train()

                mean_loss = 0.
                n_batch = 0

                for batch in self.train_loader:
                    if batch.sender_id is not None:
                        inputs, sender_id, receiver_id = batch.data, batch.sender_id, batch.receiver_id
                        sender_id = "sender_default_reco"
                        inputs = inputs[th.randperm(inputs.size()[0])]
                        agent_receiver = self.population.agents[receiver_id]
                        batch = move_to((inputs, sender_id, receiver_id), self.device)

                        _ = self.game(batch)

                        agent_receiver.tasks[task]["optimizer"].zero_grad()
                        agent_receiver.tasks[task]["loss_value"].backward()
                        agent_receiver.tasks[task]["optimizer"].step()

                        mean_loss += agent_receiver.tasks[task]["loss_value"].item()
                        n_batch+=1

                self.writer.add_scalar(f'{receiver_id}_reset/loss',mean_loss/n_batch, self.mi_step)

                mean_val_acc = 0.
                mean_val_loss = 0.
                n_batch = 0

                self.game.eval()

                with th.no_grad():
                    for batch in self.val_loader:
                        if batch.sender_id is not None:
                            #batch = move_to(batch, self.device)
                            inputs, sender_id, receiver_id = batch.data, batch.sender_id, batch.receiver_id
                            sender_id = "sender_default_reco"
                            batch = move_to((inputs, sender_id, receiver_id), self.device)

                            metrics = self.game(batch, compute_metrics=True)

                            mean_val_acc += metrics["accuracy"].detach().item()
                            mean_val_loss += agent_receiver.tasks[task]["loss_value"].item()
                            n_batch += 1

                val_losses.append(mean_val_loss / n_batch)

                self.writer.add_scalar(f'{receiver_id}_reset/val_accuracy',
                                       mean_val_acc / n_batch, self.mi_step)
                self.writer.add_scalar(f'{receiver_id}_reset/val_loss',
                                       mean_val_loss / n_batch, self.mi_step)

                self.mi_step += 1
                early_stop_step += 1

                #cond = (len(val_losses) > 20 and (
                #            val_losses[-1] > np.mean(val_losses[-20:]) - 0.0001) or early_stop_step == 1000)

                cond = (len(val_losses) > 20 and (
                        val_losses[-1] > np.mean(val_losses[-10:]) - 0.0001) or early_stop_step == 1000)

                if cond or early_stop_step>max_steps:
                    continue_training = False

            self.writer.add_scalar(f'{receiver_id}_reset/early_stop_steps',
                                   early_stop_step, epoch)

        if custom_steps > 0:

            step = 0

            continue_training = True

            while continue_training:

                self.game.train()

                mean_loss = 0.
                n_batch = 0

                for batch in self.train_loader:

                    if batch.sender_id is not None:
                        inputs, sender_id, receiver_id = batch.data, batch.sender_id, batch.receiver_id
                        sender_id = "sender_default_reco"
                        inputs = inputs[th.randperm(inputs.size()[0])]
                        agent_receiver = self.population.agents[receiver_id]
                        batch = move_to((inputs, sender_id, receiver_id), self.device)

                        _ = self.game(batch)

                        agent_receiver.tasks[task]["optimizer"].zero_grad()
                        agent_receiver.tasks[task]["loss_value"].backward()
                        agent_receiver.tasks[task]["optimizer"].step()
                        mean_loss += agent_receiver.tasks[task]["loss_value"].item()
                        n_batch += 1

                self.writer.add_scalar(f'{receiver_id}_reset/loss',mean_loss/n_batch, self.mi_step)

                mean_val_acc = 0.
                mean_val_loss = 0.
                n_batch = 0

                with th.no_grad():
                    for batch in self.val_loader:
                        if batch.sender_id is not None:
                            batch = move_to(batch, self.device)

                            metrics = self.game(batch, compute_metrics=True)

                            mean_val_acc += metrics["accuracy"].detach().item()
                            mean_val_loss += agent_receiver.tasks[task]["loss_value"].item()
                            n_batch += 1

                self.writer.add_scalar(f'{receiver_id}_reset/val_accuracy',
                                       mean_val_acc / n_batch, self.mi_step)
                self.writer.add_scalar(f'{receiver_id}_reset/val_loss',
                                       mean_val_loss / n_batch, self.mi_step)

                self.mi_step += 1
                step += 1

                if step == custom_steps:
                    continue_training = False

        # Mean loss
        mean_train_loss = 0.
        self.game.train()

        with th.no_grad():

            n_batch=0

            for batch in self.train_loader:

                inputs, sender_id, receiver_id = batch.data, batch.sender_id, batch.receiver_id
                sender_id = "sender_default_reco"
                inputs = inputs[th.randperm(inputs.size()[0])]
                agent_receiver = self.population.agents[batch.receiver_id]
                batch = move_to((inputs, sender_id, receiver_id), self.device)

                _ = self.game(batch)

                mean_train_loss += agent_receiver.tasks[task]["loss_value"].item()
                n_batch+=1

        self.writer.add_scalar(f'{receiver_id}_reset/Loss sp',mean_train_loss / n_batch, epoch)

        # Mean val loss
        mean_val_loss = 0.
        self.game.eval()
        with th.no_grad():
            n_batch = 0
            for batch in self.val_loader:
                inputs, sender_id, receiver_id = batch.data, batch.sender_id, batch.receiver_id
                sender_id = "sender_default_reco"
                agent_receiver = self.population.agents[receiver_id]
                batch = move_to((inputs, sender_id, receiver_id), self.device)

                _ = self.game(batch)

                mean_val_loss += agent_receiver.tasks[task]["loss_value"].item()
                n_batch+=1

        self.writer.add_scalar(f'{receiver_id}_reset/Loss val sp',mean_val_loss/n_batch, epoch)

    def train_communication_and_keep_gradients(self, compute_metrics: bool = False):

        mean_loss_senders = {}
        mean_h_x_m_senders = {}
        mean_gradient_tot_senders = {}
        mean_gradient_fun_senders = {}
        mean_gradient_coo_senders = {}
        ps_gradient_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.train()

        for batch in self.train_loader:

            grads_opt=[]
            grads_tot=[]

            sender_id, receiver_id = batch.sender_id, batch.receiver_id
            agent_sender = self.population.agents[sender_id]
            agent_receiver = self.population.agents[receiver_id]
            optimal_receiver_id = agent_sender.optimal_listener["train"]

            task = "communication"

            if sender_id not in mean_loss_senders:
                mean_loss_senders[sender_id] = {task: 0.}
                mean_h_x_m_senders[sender_id] = {task : 0.}
                mean_gradient_tot_senders[sender_id] = {task: 0.}
                mean_gradient_fun_senders[sender_id] = {task: 0.}
                mean_gradient_coo_senders[sender_id] = {task: 0.}
                ps_gradient_senders[sender_id] = {task:0.}
                n_batches[sender_id] = {task: 0}
            if receiver_id not in mean_loss_receivers:
                mean_loss_receivers[receiver_id] = {task: 0.}
                n_batches[receiver_id] = {task: 0}

            p_sender = th.rand(1)[0]
            p_receiver = th.rand(1)[0]

            # Forward pass optimal listener
            batch_opt = move_to((batch.data,sender_id,optimal_receiver_id), self.device)

            metrics = self.game(batch_opt, compute_metrics=compute_metrics)

            # Sender
            if p_sender < agent_sender.tasks[task]["p_step"]:
                agent_sender.tasks[task]["optimizer"].zero_grad()
                agent_sender.tasks[task]["loss_value"].backward()
                agent_sender.tasks[task]["loss_value"].register_hook(lambda grad: grad)
                for index, weight in enumerate(agent_sender.sender.parameters(), start=1):
                    gradient, *_ = weight.grad.data.clone()
                    grads_opt.append(gradient)
                for index, weight in enumerate(agent_sender.object_encoder.encoder.parameters(), start=1):
                    gradient, *_ = weight.grad.data.clone()
                    grads_opt.append(gradient)

            mean_h_x_m_senders[sender_id][task] += agent_sender.tasks[task]["loss_value"].item()

            batch = move_to(batch, self.device)

            metrics = self.game(batch, compute_metrics=compute_metrics)

            # Sender
            if p_sender < agent_sender.tasks[task]["p_step"]:
                agent_sender.tasks[task]["optimizer"].zero_grad()
                agent_sender.tasks[task]["loss_value"].backward()
                agent_sender.tasks[task]["loss_value"].register_hook(lambda grad: grad)
                for index, weight in enumerate(agent_sender.sender.parameters(), start=1):
                    gradient, *_ = weight.grad.data.clone()
                    grads_tot.append(gradient)
                for index, weight in enumerate(agent_sender.object_encoder.encoder.parameters(), start=1):
                    gradient, *_ = weight.grad.data.clone()
                    grads_tot.append(gradient)
                agent_sender.tasks[task]["optimizer"].step()

            mean_loss_senders[sender_id][task] += agent_sender.tasks[task]["loss_value"].item()
            n_batches[sender_id][task] += 1

            # Receiver
            if p_receiver < agent_receiver.tasks[task]["p_step"]:
                agent_receiver.tasks[task]["optimizer"].zero_grad()
                agent_receiver.tasks[task]["loss_value"].backward()
                agent_receiver.tasks[task]["optimizer"].step()

            mean_loss_receivers[receiver_id][task] += agent_receiver.tasks[task]["loss_value"].item()
            n_batches[receiver_id][task] += 1

            grad_tot_value=0.
            grad_fun_value=0.
            ps_value = 0.
            grad_coo_value = 0.

            for i in range(len(grads_tot)):
                grad_tot_value += (grads_tot[i] ** 2).mean().item()
                grad_fun_value += (grads_opt[i] ** 2).mean().item()
                grad_coo_value += ((grads_tot[i]-grads_opt[i]) ** 2).mean().item()
                ps_value += ((grads_tot[i]*grads_opt[i])).sum().item()


            mean_gradient_tot_senders[sender_id][task] += grad_tot_value
            mean_gradient_fun_senders[sender_id][task] += grad_fun_value
            mean_gradient_coo_senders[sender_id][task] += grad_coo_value
            ps_gradient_senders[sender_id][task] += ps_value

            if compute_metrics:
                # Store metrics
                if sender_id not in mean_metrics:
                    mean_metrics[sender_id] = {"accuracy": 0.,
                                               "accuracy_tot": 0.,
                                               "sender_entropy": 0.,
                                               "sender_log_prob": 0.,
                                               "message_length": 0.}
                if receiver_id not in mean_metrics:
                    mean_metrics[receiver_id] = {"accuracy": 0.,"accuracy_tot":0.,"entropy":0.}

                mean_metrics[sender_id]["accuracy"] += metrics["accuracy"]
                mean_metrics[sender_id]["accuracy_tot"] += metrics["accuracy_tot"]
                mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"]
                mean_metrics[receiver_id]["accuracy_tot"] += metrics["accuracy_tot"]
                mean_metrics[receiver_id]["entropy"] += metrics["entropy_receiver"]

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                               for receiver_id in mean_loss_receivers}
        mean_gradient_tot_senders = {sender_id: _div_dict(mean_gradient_tot_senders[sender_id], n_batches[sender_id])
                                    for sender_id in mean_gradient_tot_senders}
        mean_gradient_fun_senders = {sender_id: _div_dict(mean_gradient_fun_senders[sender_id], n_batches[sender_id])
                                     for sender_id in mean_gradient_fun_senders}
        mean_gradient_coo_senders = {sender_id: _div_dict(mean_gradient_coo_senders[sender_id], n_batches[sender_id])
                                     for sender_id in mean_gradient_coo_senders}
        ps_gradient_senders = {sender_id: _div_dict(ps_gradient_senders[sender_id], n_batches[sender_id])
                                     for sender_id in ps_gradient_senders}


        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt][task])

        return mean_loss_senders, \
               mean_loss_receivers, \
               mean_gradient_tot_senders, \
               mean_gradient_fun_senders, \
               mean_gradient_coo_senders, \
               ps_gradient_senders, \
               mean_metrics

    def pretrain_optimal_listener(self, epoch: int, reset: bool = False, threshold=1e-3):

        if not reset:  # reset optimizer
            for sender_id in self.population.sender_names:
                agent_sender = self.population.agents[sender_id]
                optimal_listener_id = agent_sender.optimal_listener["train"]
                optimal_listener = self.population.agents[optimal_listener_id]

                model_parameters = list(optimal_listener.receiver.parameters()) + \
                                   list(optimal_listener.object_decoder.parameters())
                optimal_listener.tasks["communication"]["optimizer"] = th.optim.Adam(model_parameters, lr=0.0005)

        prev_loss_value = [0.]
        step = 0
        task = "communication"

        continue_optimal_listener_training = True

        while continue_optimal_listener_training:

            self.game.train()

            mean_loss_value = 0.
            n_batch = 0

            for batch in self.train_loader:

                inputs, sender_id = batch.data, batch.sender_id
                optimal_listener_id = agent_sender.optimal_listener["train"]
                optimal_listener = self.population.agents[optimal_listener_id]
                inputs = inputs[th.randperm(inputs.size()[0])]
                batch = move_to((inputs, sender_id, optimal_listener_id), self.device)

                _ = self.game(batch)

                optimal_listener.tasks[task]["optimizer"].zero_grad()
                optimal_listener.tasks[task]["loss_value"].backward()
                optimal_listener.tasks[task]["optimizer"].step()

                mean_loss_value += optimal_listener.tasks[task]["loss_value"].item()
                n_batch += 1

            mean_loss_value /= n_batch

            self.writer.add_scalar(f'{optimal_listener_id}/loss',
                                   mean_loss_value, self.mi_step)

            self.mi_step += 1
            step += 1

            if step == 200:
                # if abs(mean_val_loss / n_batch - np.mean(self.val_loss_optimal_listener[:-1])) < 10e-3 or \
                #        mean_val_loss / n_batch > np.mean(self.val_loss_optimal_listener[:-1]):
                continue_optimal_listener_training = False

    def pretrain_language_model(self, threshold=1e-2):

        self.game.train()

        messages_lm = []
        for sender_id in self.population.sender_names:

            agent_sender = self.population.agents[sender_id]

            for batch in self.train_loader:
                inputs = batch.data.to(self.device)
                inputs_embedding = agent_sender.encode_object(inputs)
                messages, _, _ = agent_sender.send(inputs_embedding)
                messages_lm.append(messages)

            messages_lm = torch.stack(messages_lm).view(-1, messages_lm[0].size(1))

            agent_sender.train_language_model(messages_lm, threshold=threshold)

    def train_communication_and_kl(self, compute_metrics=True):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.train()

        self.pretrain_language_model()

        for batch in self.train_loader:

            sender_id, receiver_id = batch.sender_id, batch.receiver_id
            agent_sender = self.population.agents[sender_id]
            agent_receiver = self.population.agents[receiver_id]

            weights = {"communication": agent_sender.weights["communication"],
                       "KL": agent_sender.weights["KL"]}

            task = "communication"

            if sender_id not in mean_loss_senders:
                mean_loss_senders[sender_id] = {task: 0.}
                n_batches[sender_id] = {task: 0}
            if receiver_id not in mean_loss_receivers:
                mean_loss_receivers[receiver_id] = {task: 0.}
                n_batches[receiver_id] = {task: 0}

            batch = move_to((batch.data, sender_id, receiver_id, weights), self.device)

            metrics = self.game.communication_and_kl_instance(*batch, compute_metrics=compute_metrics)

            # Sender
            if th.rand(1)[0] < agent_sender.tasks[task]["p_step"]:
                agent_sender.tasks[task]["optimizer"].zero_grad()
                agent_sender.tasks[task]["loss_value"].backward()
                agent_sender.tasks[task]["optimizer"].step()

            mean_loss_senders[sender_id][task] += agent_sender.tasks[task]["loss_value"].item()
            n_batches[sender_id][task] += 1

            # Receiver
            if th.rand(1)[0] < agent_receiver.tasks[task]["p_step"]:
                agent_receiver.tasks[task]["optimizer"].zero_grad()
                agent_receiver.tasks[task]["loss_value"].backward()
                agent_receiver.tasks[task]["optimizer"].step()

            mean_loss_receivers[receiver_id][task] += agent_receiver.tasks[task]["loss_value"].item()
            n_batches[receiver_id][task] += 1

            if compute_metrics:
                # Store metrics
                if sender_id not in mean_metrics:
                    mean_metrics[sender_id] = {"accuracy": 0.,
                                               "accuracy_tot": 0.,
                                               "sender_entropy": 0.,
                                               "sender_log_prob": 0.,
                                               "message_length": 0.}
                if receiver_id not in mean_metrics:
                    mean_metrics[receiver_id] = {"accuracy": 0.,"accuracy_tot": 0.}

                mean_metrics[sender_id]["accuracy"] += metrics["accuracy"]
                mean_metrics[sender_id]["accuracy_tot"] += metrics["accuracy_tot"]
                mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"]
                mean_metrics[sender_id]["accuracy_tot"] += metrics["accuracy_tot"]

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                               for receiver_id in mean_loss_receivers}

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt][task])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def train_communication_and_mutual_information(self, compute_metrics=True):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.train()

        for batch in self.train_loader:

            if batch.sender_id is not None:
                inputs = batch.data
                inputs = inputs[th.randperm(inputs.size()[0])]

                sender_id, receiver_id = batch.sender_id, batch.receiver_id
                agent_sender = self.population.agents[sender_id]
                agent_receiver = self.population.agents[receiver_id]

                optimal_listener_id = agent_sender.optimal_listener["train"]

                weights = {receiver_id: agent_sender.weights["communication"],
                           optimal_listener_id: agent_sender.weights["MI"]}

                task = "communication"

                if sender_id not in mean_loss_senders:
                    mean_loss_senders[sender_id] = {task: 0.}
                    n_batches[sender_id] = {task: 0}
                if receiver_id not in mean_loss_receivers:
                    mean_loss_receivers[receiver_id] = {task: 0.}
                    n_batches[receiver_id] = {task: 0}

                batch = move_to((inputs, sender_id, [receiver_id, optimal_listener_id], weights), self.device)

                metrics = self.game.communication_multi_listener_instance(*batch,
                                                                          compute_metrics=compute_metrics)

                # Sender
                if th.rand(1)[0] < agent_sender.tasks[task]["p_step"]:
                    agent_sender.tasks[task]["optimizer"].zero_grad()
                    agent_sender.tasks[task]["loss_value"].backward()
                    agent_sender.tasks[task]["optimizer"].step()

                mean_loss_senders[sender_id][task] += agent_sender.tasks[task]["loss_value"].item()
                n_batches[sender_id][task] += 1

                # Receiver
                if th.rand(1)[0] < agent_receiver.tasks[task]["p_step"]:
                    agent_receiver.tasks[task]["optimizer"].zero_grad()
                    agent_receiver.tasks[task]["loss_value"].backward()
                    agent_receiver.tasks[task]["optimizer"].step()

                mean_loss_receivers[receiver_id][task] += agent_receiver.tasks[task]["loss_value"].item()
                n_batches[receiver_id][task] += 1

                if compute_metrics:
                    # Store metrics
                    if sender_id not in mean_metrics:
                        mean_metrics[sender_id] = {"accuracy": 0.,
                                                   "accuracy_tot" : 0.,
                                                   "sender_entropy": 0.,
                                                   "sender_log_prob": 0.,
                                                   "message_length": 0.}
                    if receiver_id not in mean_metrics:
                        mean_metrics[receiver_id] = {"accuracy": 0.,"accuracy_tot" : 0.}

                    mean_metrics[sender_id]["accuracy"] += metrics["accuracy"][receiver_id]
                    mean_metrics[sender_id]["accuracy_tot"] += metrics["accuracy_tot"][receiver_id]
                    mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                    mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                    mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                    mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"][receiver_id]
                    mean_metrics[receiver_id]["accuracy_tot"] += metrics["accuracy_tot"][receiver_id]

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                               for receiver_id in mean_loss_receivers}

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt][task])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def weigth_noise_listener(self):

        for receiver_id in self.population.receiver_names:
            self.population.agents[receiver_id].weight_noise()


def build_trainer(game,
                  trainer_type : str,
                  evaluator,
                  train_loader: th.utils.data.DataLoader,
                  game_params: dict,
                  agent_repertory: dict,
                  mi_loader: th.utils.data.DataLoader = None,
                  val_loader: th.utils.data.DataLoader = None,
                  test_loader: th.utils.data.DataLoader = None,
                  imitation_loader: th.utils.data.DataLoader = None,
                  logger: th.utils.tensorboard.SummaryWriter = None,
                  compute_metrics: bool = False,
                  metrics_save_dir: str = "",
                  models_save_dir : str = "",
                  device: str = "cpu"):

    if trainer_type == "population":
        trainer = TrainerCustom(game=game,
                                evaluator=evaluator,
                                train_loader=train_loader,
                                mi_loader=mi_loader,
                                imitation_loader=imitation_loader,
                                val_loader=val_loader,
                                test_loader=test_loader,
                                game_params=game_params,
                                agent_repertory=agent_repertory,
                                logger=logger,
                                metrics_save_dir=metrics_save_dir,
                                models_save_dir=models_save_dir,
                                device=device)
    elif trainer_type=="custom":
        trainer = TrainerCustom(game=game,
                                evaluator=evaluator,
                                train_loader=train_loader,
                                mi_loader=mi_loader,
                                imitation_loader=imitation_loader,
                                val_loader=val_loader,
                                test_loader=test_loader,
                                game_params=game_params,
                                agent_repertory=agent_repertory,
                                logger=logger,
                                metrics_save_dir=metrics_save_dir,
                                models_save_dir=models_save_dir,
                                device=device)
    elif trainer_type=="pretraining":
        trainer = PretrainingTrainer(game=game,
                                     train_loader=train_loader,
                                     compute_metrics=compute_metrics,
                                     device=device)
    else:
        raise "Specify a know Trainer"

    return trainer
