import torch as th
import numpy as np
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from .utils import _div_dict, move_to


class TrainerBis:

    def __init__(self,
                 game,
                 evaluator,
                 train_loader: th.utils.data.DataLoader,
                 imitation_loader: th.utils.data.DataLoader = None,
                 mi_loader: th.utils.data.DataLoader = None,
                 val_loader: th.utils.data.DataLoader = None,
                 logger: th.utils.tensorboard.SummaryWriter = None,
                 device: str = "cpu") -> None:

        self.game = game
        self.population = game.population
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.imitation_loader = imitation_loader
        self.mi_loader = mi_loader
        self.evaluator = evaluator
        self.start_epoch = 1
        self.device = th.device(device)
        self.writer = logger
        self.mi_step=0

    def train(self,
              n_epochs,
              train_communication_freq: int = 1000000,
              validation_freq: int = 1,
              train_imitation_freq: int = 100000,
              train_mi_freq: int = 100000,
              train_kl_freq: int = 100000,
              train_broadcasting_freq:int = 1000000,
              evaluator_freq: int = 1000000,
              print_evolution: bool = True):

        for epoch in range(self.start_epoch, n_epochs):

            if print_evolution: print(f"Epoch {epoch}")

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
            if epoch % train_mi_freq == 0 and self.mi_loader is not None:
                self.pretrain_optimal_listener(epoch=epoch)
                train_communication_mi_loss_senders, train_communication_loss_receivers, train_metrics = \
                    self.train_communication_and_mutual_information()
            else:
                train_communication_mi_loss_senders, train_communication_loss_receivers, train_metrics = \
                    None, None, None

            if epoch % train_broadcasting_freq == 0:
                train_communication_loss_senders, train_communication_loss_receivers, train_metrics = \
                    self.train_communication_broadcasting(compute_metrics=True)  # dict,dict, dict

            # Train KL div
            if epoch % train_kl_freq == 0 and self.mi_loader is not None:
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

            if self.evaluator is not None and epoch % evaluator_freq == 0:
                self.evaluator.step(epoch=epoch)

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
                mean_loss_senders[sender_id] = {task:0.}
                n_batches[sender_id] = {task:0}
            if receiver_id not in mean_loss_receivers:
                mean_loss_receivers[receiver_id] = {task:0.}
                n_batches[receiver_id] = {task:0}

            batch = move_to(batch, self.device)

            metrics = self.game(batch, compute_metrics=compute_metrics)

            # Sender
            if th.rand(1)[0] < agent_sender.tasks[task]["p_step"]:
                agent_sender.tasks[task]["optimizer"].zero_grad()
                agent_sender.tasks[task]["loss_value"].backward(retain_graph=True)
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
                                               "sender_entropy": 0.,
                                               "sender_log_prob": 0.,
                                               "message_length": 0.}
                if receiver_id not in mean_metrics:
                    mean_metrics[receiver_id] = {"accuracy": 0.}

                mean_metrics[sender_id]["accuracy"] += metrics["accuracy"]
                mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"]

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                               for receiver_id in mean_loss_receivers}

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt][task])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def train_communication_broadcasting(self,compute_metrics=True):

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
                weights[receiver_id]=1

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
                        mean_metrics[receiver_id] = {"accuracy": 0.}

                    mean_metrics[sender_id]["accuracy"] += metrics["accuracy"][receiver_id]
                    mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                    mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                    mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                    mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"][receiver_id]

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                               for receiver_id in mean_loss_receivers}

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt][task])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def pretrain_optimal_listener(self,epoch:int,threshold=1e-2):

        self.game.train()

        prev_loss_value = [0.]
        continue_optimal_listener_training = True

        task = "communication"

        while continue_optimal_listener_training:

            batch = next(iter(self.mi_loader))
            inputs, sender_id = batch.data, batch.sender_id
            agent_sender = self.population.agents[sender_id]
            optimal_listener_id = agent_sender.optimal_listener
            optimal_listener = self.population.agents[optimal_listener_id]
            batch = move_to((inputs,sender_id,optimal_listener_id), self.device)

            _ = self.game(batch)

            optimal_listener.tasks[task]["optimizer"].zero_grad()
            optimal_listener.tasks[task]["loss_value"].backward()
            optimal_listener.tasks[task]["optimizer"].step()

            if len(prev_loss_value) > 9 and \
                    abs(optimal_listener.tasks[task]["loss_value"].item() - np.mean(prev_loss_value)) < threshold:
                continue_optimal_listener_training = False
            else:
                prev_loss_value.append(optimal_listener.tasks[task]["loss_value"].item())
                if len(prev_loss_value) > 10:
                    prev_loss_value.pop(0)

            self.writer.add_scalar(f'{optimal_listener_id}/loss',
                                   optimal_listener.tasks[task]["loss_value"].item(), self.mi_step)

            self.mi_step += 1

        self.writer.add_scalar(f'{optimal_listener_id}/MI',
                               optimal_listener.tasks[task]["loss_value"].item(), epoch)

    def pretrain_language_model(self,epoch:int,threshold=1e-2):

        self.game.train()

        messages_lm = []
        for sender_id in self.population.sender_names:

            agent_sender = self.population.agents[sender_id]

            for batch in self.mi_loader:
                inputs = batch.data.to(self.device)
                inputs_embedding = agent_sender.encode_object(inputs)
                messages, _, _ = agent_sender.send(inputs_embedding)
                messages_lm.append(messages)

            messages_lm = torch.stack(messages_lm).view(-1, messages_lm[0].size(1))

            agent_sender.train_language_model(messages_lm,threshold=threshold)

    def train_communication_and_kl(self,compute_metrics=True):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.train()

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

            batch = move_to((batch.data,sender_id,receiver_id,weights), self.device)

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
                                               "sender_entropy": 0.,
                                               "sender_log_prob": 0.,
                                               "message_length": 0.}
                if receiver_id not in mean_metrics:
                    mean_metrics[receiver_id] = {"accuracy": 0.}

                mean_metrics[sender_id]["accuracy"] += metrics["accuracy"]
                mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"]

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                               for receiver_id in mean_loss_receivers}

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt][task])

        return mean_loss_senders, mean_loss_receivers, mean_metrics


    def train_communication_and_mutual_information(self,compute_metrics=True):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.train()

        for batch in self.train_loader:

            sender_id, receiver_id = batch.sender_id, batch.receiver_id
            agent_sender = self.population.agents[sender_id]
            agent_receiver = self.population.agents[receiver_id]
            optimal_listener_id = agent_sender.optimal_listener

            weights = {receiver_id:agent_sender.weights["communication"],optimal_listener_id:agent_sender.weights["MI"]}

            task = "communication"

            if sender_id not in mean_loss_senders:
                mean_loss_senders[sender_id] = {task: 0.}
                n_batches[sender_id] = {task: 0}
            if receiver_id not in mean_loss_receivers:
                mean_loss_receivers[receiver_id] = {task: 0.}
                n_batches[receiver_id] = {task: 0}

            batch = move_to((batch.data,sender_id,[receiver_id,optimal_listener_id],weights), self.device)

            metrics = self.game.communication_multi_listener_instance(*batch, compute_metrics=compute_metrics)

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
                                               "sender_entropy": 0.,
                                               "sender_log_prob": 0.,
                                               "message_length": 0.}
                if receiver_id not in mean_metrics:
                    mean_metrics[receiver_id] = {"accuracy": 0.}

                mean_metrics[sender_id]["accuracy"] += metrics["accuracy"][receiver_id]
                mean_metrics[sender_id]["accuracy (optimal listener)"] += metrics["accuracy"][optimal_listener_id]
                mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"][receiver_id]

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

            for batch in self.val_loader:

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
                        mean_metrics[sender_id] = {"accuracy": 0., "sender_entropy": 0., "message_length": 0.}
                    if receiver_id not in mean_metrics:
                        mean_metrics[receiver_id] = {"accuracy": 0.}
                    mean_metrics[sender_id]["accuracy"] += metrics["accuracy"]
                    mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                    mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                    mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"]

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
                    train_communication_loss_senders: dict,
                    train_communication_loss_receivers: dict,
                    train_imitation_loss_senders: dict,
                    train_imitation_loss_imitators: dict,
                    train_communication_mi_loss_senders: dict,
                    train_metrics: dict,
                    val_loss_senders: dict,
                    val_loss_receivers: dict,
                    val_metrics: dict) -> None:

        # Train
        if train_communication_loss_senders is not None:
            for sender, tasks in train_communication_loss_senders.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{sender}/{task} (train)', l, epoch)

                    if task == "communication":
                        self.writer.add_scalar(f'{sender}/accuracy (train)',
                                               train_metrics[sender]['accuracy'], epoch)
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
                self.writer.add_scalar(f'{sender}/Language entropy (val)',
                                       val_metrics[sender]['sender_entropy'], epoch)
                self.writer.add_scalar(f'{sender}/Messages length (val)',
                                       val_metrics[sender]['message_length'], epoch)

            for receiver, tasks in val_loss_receivers.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{receiver}/Loss val', l, epoch)

                self.writer.add_scalar(f'{receiver}/accuracy (val)',
                                       val_metrics[receiver]['accuracy'], epoch)



class Trainer:

    def __init__(self,
                 game,
                 evaluator,
                 train_loader: th.utils.data.DataLoader,
                 imitation_loader: th.utils.data.DataLoader = None,
                 mi_loader: th.utils.data.DataLoader = None,
                 val_loader: th.utils.data.DataLoader = None,
                 logger: th.utils.tensorboard.SummaryWriter = None,
                 device: str = "cpu") -> None:

        self.game = game
        self.population = game.population
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.imitation_loader = imitation_loader
        self.mi_loader = mi_loader
        self.evaluator = evaluator
        self.start_epoch = 1
        self.device = th.device(device)
        self.writer = logger
        self.mi_step=0

    def train(self,
              n_epochs,
              train_communication_freq: int = 1,
              validation_freq: int = 1,
              train_imitation_freq: int = 100000,
              train_mi_freq: int = 100000,
              train_mi_with_lm_freq: int = 100000,
              evaluator_freq: int = 1000000,
              print_evolution: bool = True):

        for epoch in range(self.start_epoch, n_epochs):

            if print_evolution: print(f"Epoch {epoch}")

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
            if epoch % train_mi_freq == 0 and self.mi_loader is not None:
                train_mi_loss_senders = self.train_mutual_information()
            else:
                train_mi_loss_senders = None

            # Train Mutual information
            if epoch % train_mi_with_lm_freq == 0 and self.mi_loader is not None:
                train_mi_loss_senders = self.train_mutual_information_with_lm()
            else:
                train_mi_loss_senders = None

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
                                 train_mi_loss_senders=train_mi_loss_senders,
                                 train_metrics=train_metrics,
                                 val_loss_senders=val_loss_senders,
                                 val_loss_receivers=val_loss_receivers,
                                 val_metrics=val_metrics)

            if self.evaluator is not None and epoch % evaluator_freq == 0:
                self.evaluator.step(epoch=epoch)

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
                mean_loss_senders[sender_id] = {task:0.}
                n_batches[sender_id] = {task:0}
            if receiver_id not in mean_loss_receivers:
                mean_loss_receivers[receiver_id] = {task:0.}
                n_batches[receiver_id] = {task:0}

            batch = move_to(batch, self.device)

            metrics = self.game(batch, compute_metrics=compute_metrics)

            # Sender
            if th.rand(1)[0] < agent_sender.tasks[task]["p_step"]:
                agent_sender.tasks[task]["optimizer"].zero_grad()
                agent_sender.tasks[task]["loss_value"].backward(retain_graph=True)
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
                                               "sender_entropy": 0.,
                                               "sender_log_prob": 0.,
                                               "message_length": 0.}
                if receiver_id not in mean_metrics:
                    mean_metrics[receiver_id] = {"accuracy": 0.}

                mean_metrics[sender_id]["accuracy"] += metrics["accuracy"]
                mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                mean_metrics[sender_id]["sender_log_prob"] += metrics["sender_log_prob"].sum(1).mean().item()
                mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"]

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[receiver_id])
                               for receiver_id in mean_loss_receivers}

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt][task])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def train_imitation(self):

        mean_loss_senders = {}
        mean_loss_imitators = {}
        n_batches = {}

        self.game.train()

        for batch in self.imitation_loader:

            sender_id, imitator_id = batch.sender_id, batch.imitator_id
            agent_sender, agent_imitator = self.population.agents[sender_id], self.population.agents[imitator_id]

            batch = move_to(batch, self.device)

            self.game.imitation_instance(*batch)

            if sender_id not in mean_loss_senders:
                mean_loss_senders[sender_id] = {}
                n_batches[sender_id] = {}
            if imitator_id not in mean_loss_imitators:
                mean_loss_imitators[imitator_id] = {}
                n_batches[imitator_id] = {}

            task = "imitation"

            # Sender
            if th.rand(1)[0] < agent_sender.tasks[task]["p_step"]:

                if task not in mean_loss_senders[sender_id]:
                    mean_loss_senders[sender_id][task] = 0.
                    n_batches[sender_id][task] = 0

                agent_sender.tasks[task]["optimizer"].zero_grad()
                agent_sender.tasks[task]["loss_value"].backward()
                agent_sender.tasks[task]["optimizer"].step()

                mean_loss_senders[sender_id][task] += agent_sender.tasks[task]["loss_value"].item()
                n_batches[sender_id][task] += 1

            # Imitator
            if th.rand(1)[0] < agent_imitator.tasks[task]["p_step"]:

                if task not in mean_loss_imitators[imitator_id]:
                    mean_loss_imitators[imitator_id][task] = 0.
                    n_batches[imitator_id][task] = 0

                agent_imitator.tasks[task]["optimizer"].zero_grad()
                agent_imitator.tasks[task]["loss_value"].backward()
                agent_imitator.tasks[task]["optimizer"].step()

                mean_loss_imitators[imitator_id][task] += agent_imitator.tasks[task]["loss_value"].item()
                n_batches[imitator_id][task] += 1

        mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                             for sender_id in mean_loss_senders}
        mean_loss_imitators = {imitator_id: _div_dict(mean_loss_imitators[imitator_id], n_batches[imitator_id])
                               for imitator_id in mean_loss_imitators}

        return mean_loss_senders, mean_loss_imitators

    def train_mutual_information(self,threshold=1e-3):

        self.game.train()

        prev_loss_value = [0.]
        continue_optimal_listener_training = True

        task = "communication"

        while continue_optimal_listener_training:

            batch = next(iter(self.mi_loader))
            inputs, sender_id = batch.data, batch.sender_id
            agent_sender = self.population.agents[sender_id]
            optimal_listener_id = agent_sender.optimal_listener
            optimal_listener = self.population.agents[optimal_listener_id]
            batch = move_to((inputs,sender_id,optimal_listener_id), self.device)

            _ = self.game(batch)

            optimal_listener.tasks[task]["optimizer"].zero_grad()
            optimal_listener.tasks[task]["loss_value"].backward()
            optimal_listener.tasks[task]["optimizer"].step()

            if len(prev_loss_value) > 9 and \
                    abs(optimal_listener.tasks[task]["loss_value"].item() - np.mean(prev_loss_value)) < threshold:
                continue_optimal_listener_training = False
            else:
                prev_loss_value.append(optimal_listener.tasks[task]["loss_value"].item())
                if len(prev_loss_value) > 10:
                    prev_loss_value.pop(0)

            self.writer.add_scalar(f'{optimal_listener_id}/loss',
                                   optimal_listener.tasks[task]["loss_value"].item(), self.mi_step)

            self.mi_step += 1

        agent_sender.tasks["mutual_information"]["optimizer"].zero_grad()
        agent_sender.tasks[task]["loss_value"].backward()
        agent_sender.tasks["mutual_information"]["optimizer"].step()

        return {sender_id:agent_sender.tasks[task]["loss_value"].item()}

    def train_mutual_information_with_lm(self,threshold=1e-3):

        self.game.train()

        # Language model training
        #messages_lm = []
        #for sender_id in self.population.sender_names:

        #    agent_sender = self.population.agents[sender_id]

        #    for batch in self.mi_loader:
        #        inputs = batch.data.to(self.device)
        #        inputs_embedding = agent_sender.encode_object(inputs)
        #        messages, _, _ = agent_sender.send(inputs_embedding)
        #        messages_lm.append(messages)
        #    messages_lm = torch.stack(messages_lm).view(-1, messages_lm[0].size(1))

        #    agent_sender.train_language_model(messages_lm)

        task = "mutual_information"

        batch = next(iter(self.train_loader))
        inputs, sender_id = batch.data, batch.sender_id
        agent_sender = self.population.agents[sender_id]
        batch = move_to((inputs, sender_id), self.device)

        mutual_information = self.game.mi_instance(*batch)

        agent_sender.tasks[task]["optimizer"].zero_grad()
        agent_sender.tasks["mutual_information"]["loss_value"].backward()
        agent_sender.tasks[task]["optimizer"].step()

        self.writer.add_scalar(f'{sender_id}/reward_mi',
                               mutual_information.mean().item(), self.mi_step)

        self.mi_step+=1

        return {sender_id:agent_sender.tasks["mutual_information"]["loss_value"].item()}

    def train_mutual_information_direct(self):

        mean_mi_senders = {}
        n_batches = {}

        self.game.train()

        for batch in self.mi_loader:

            batch = move_to(batch, self.device)
            inputs, sender_id = batch[0], batch[1]

            if sender_id not in mean_mi_senders:
                mean_mi_senders[sender_id] = 0.
                n_batches[sender_id] = 0

            agent = self.population.agents[sender_id]

            loss_m_i = agent.compute_mutual_information(inputs)

            self.population.agents[sender_id].sender_optimizer.zero_grad()
            loss_m_i.backward()
            self.population.agents[sender_id].sender_optimizer.step()

            # Store losses
            mean_mi_senders[sender_id] += loss_m_i
            n_batches[sender_id] += 1


        mean_mi_senders = _div_dict(mean_mi_senders, n_batches)

        return mean_mi_senders

    def eval(self, compute_metrics: bool = False):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.eval()

        with th.no_grad():

            for batch in self.val_loader:

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
                        mean_metrics[sender_id] = {"accuracy": 0., "sender_entropy": 0., "message_length": 0.}
                    if receiver_id not in mean_metrics:
                        mean_metrics[receiver_id] = {"accuracy": 0.}
                    mean_metrics[sender_id]["accuracy"] += metrics["accuracy"]
                    mean_metrics[sender_id]["sender_entropy"] += metrics["sender_entropy"]
                    mean_metrics[sender_id]["message_length"] += metrics["message_length"]
                    mean_metrics[receiver_id]["accuracy"] += metrics["accuracy"]

            mean_loss_senders = {sender_id: _div_dict(mean_loss_senders[sender_id], n_batches[sender_id])
                                 for sender_id in mean_loss_senders}
            mean_loss_receivers = {receiver_id: _div_dict(mean_loss_receivers[receiver_id], n_batches[sender_id])
                                   for receiver_id in mean_loss_receivers}

            if compute_metrics:
                for agt in mean_metrics:
                    mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt]["communication"])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def log_metrics(self,
                    epoch: int,
                    train_communication_loss_senders: dict,
                    train_communication_loss_receivers: dict,
                    train_imitation_loss_senders: dict,
                    train_imitation_loss_imitators: dict,
                    train_mi_loss_senders: dict,
                    train_metrics: dict,
                    val_loss_senders: dict,
                    val_loss_receivers: dict,
                    val_metrics: dict) -> None:

        # Train
        if train_communication_loss_senders is not None:
            for sender, tasks in train_communication_loss_senders.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{sender}/{task} (train)', l, epoch)

                    if task == "communication":
                        self.writer.add_scalar(f'{sender}/accuracy (train)',
                                               train_metrics[sender]['accuracy'], epoch)
                        self.writer.add_scalar(f'{sender}/Language entropy (train)',
                                               train_metrics[sender]['sender_entropy'], epoch)
                        self.writer.add_scalar(f'{sender}/Sender log prob',
                                               train_metrics[sender]['sender_log_prob'], epoch)
                        self.writer.add_scalar(f'{sender}/Messages length (train)',
                                               train_metrics[sender]['message_length'], epoch)

            for receiver, tasks in train_communication_loss_receivers.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{receiver}/{task}_train', l, epoch)

                    if task == "communication":
                        self.writer.add_scalar(f'{receiver}/accuracy (train)',
                                               train_metrics[receiver]['accuracy'], epoch)

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
        if train_mi_loss_senders is not None:
            for sender, l in train_mi_loss_senders.items():
                self.writer.add_scalar(f'{sender}/Loss Mutual information', l, epoch)

        # Val
        if val_loss_senders is not None:
            for sender, tasks in val_loss_senders.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{sender}/Loss val', l, epoch)

                self.writer.add_scalar(f'{sender}/accuracy (val)',
                                       val_metrics[sender]['accuracy'], epoch)
                self.writer.add_scalar(f'{sender}/Language entropy (val)',
                                       val_metrics[sender]['sender_entropy'], epoch)
                self.writer.add_scalar(f'{sender}/Messages length (val)',
                                       val_metrics[sender]['message_length'], epoch)

            for receiver, tasks in val_loss_receivers.items():
                for task, l in tasks.items():
                    self.writer.add_scalar(f'{receiver}/Loss val', l, epoch)

                self.writer.add_scalar(f'{receiver}/accuracy (val)',
                                       val_metrics[receiver]['accuracy'], epoch)


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

            self.agent.sender_optimizer.zero_grad()
            loss_sender.backward()
            self.agent.sender_optimizer.step()

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


def build_trainer(game,
                  evaluator,
                  train_loader: th.utils.data.DataLoader,
                  mi_loader: th.utils.data.DataLoader = None,
                  val_loader: th.utils.data.DataLoader = None,
                  imitation_loader: th.utils.data.DataLoader = None,
                  logger: th.utils.tensorboard.SummaryWriter = None,
                  compute_metrics: bool = False,
                  pretraining: bool = False,
                  device: str = "cpu"):
    if not pretraining:
        trainer = TrainerBis(game=game,
                              evaluator=evaluator,
                              train_loader=train_loader,
                              mi_loader=mi_loader,
                              imitation_loader=imitation_loader,
                              val_loader=val_loader,
                              logger=logger,
                              device=device)
    else:
        trainer = PretrainingTrainer(game=game,
                                     train_loader=train_loader,
                                     compute_metrics=compute_metrics,
                                     device=device)

    return trainer
