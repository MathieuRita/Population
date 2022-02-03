import torch as th
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from .utils import _div_dict, move_to


class Trainer:

    def __init__(self,
                 game,
                 evaluator,
                 train_loader: th.utils.data.DataLoader,
                 val_loader: th.utils.data.DataLoader = None,
                 mi_loader: th.utils.data.DataLoader = None,
                 logger: th.utils.tensorboard.SummaryWriter = None,
                 device: str = "cpu") -> None:

        self.game = game
        self.population = game.population
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mi_loader = mi_loader
        self.evaluator = evaluator
        self.start_epoch = 1
        self.device = th.device(device)
        self.writer = logger

    def train(self,
              n_epochs,
              train_freq : int = 1,
              validation_freq: int = 1,
              mi_freq : int = 1000,
              evaluator_freq: int = 1000,
              print_evolution: bool = True):

        for epoch in range(self.start_epoch, n_epochs):

            if print_evolution: print(f"Epoch {epoch}")

            # Mutual information
            if self.mi_loader is not None and epoch % mi_freq == 0:
                mi_loss_senders = self.train_mutual_information()
            else:
                mi_loss_senders = None

            # Train
            if epoch % train_freq ==0:
                train_loss_senders, train_loss_receivers, train_metrics = \
                    self.train_epoch(compute_metrics=True)  # dict,dict, dict
            else:
                train_loss_senders, train_loss_receivers, train_metrics = None, None, None

            # Validation
            if self.val_loader is not None and epoch % validation_freq == 0:
                val_loss_senders, val_loss_receivers, val_metrics = self.eval(compute_metrics=True)
            else:
                val_loss_senders, val_loss_receivers, val_metrics = None, None, None

            if self.writer is not None:
                self.log_metrics(epoch=epoch,
                                 train_loss_senders=train_loss_senders,
                                 train_loss_receivers=train_loss_receivers,
                                 mi_loss_senders = mi_loss_senders,
                                 train_metrics=train_metrics,
                                 val_loss_senders=val_loss_senders,
                                 val_loss_receivers=val_loss_receivers,
                                 val_metrics=val_metrics)

            if self.evaluator is not None and epoch % evaluator_freq == 0:
                self.evaluator.step(epoch=epoch)

    def train_epoch(self, compute_metrics: bool = False):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.train()

        for batch in self.train_loader:

            sender_id, receiver_id = batch.sender_id, batch.receiver_id

            if sender_id not in mean_loss_senders:
                mean_loss_senders[sender_id] = 0.
                n_batches[sender_id] = 0
            if receiver_id not in mean_loss_receivers:
                mean_loss_receivers[receiver_id] = 0.
                n_batches[receiver_id] = 0

            batch = move_to(batch, self.device)

            loss_sender, loss_receiver, metrics = self.game(batch, compute_metrics=compute_metrics)

            if th.rand(1)[0] < self.population.agents[sender_id].p_step:
                self.population.agents[sender_id].sender_optimizer.zero_grad()
                loss_sender.backward()
                self.population.agents[sender_id].sender_optimizer.step()

            if th.rand(1)[0] < self.population.agents[receiver_id].p_step:
                self.population.agents[receiver_id].receiver_optimizer.zero_grad()
                loss_receiver.backward()
                self.population.agents[receiver_id].receiver_optimizer.step()

            # Store losses
            mean_loss_senders[sender_id] += loss_sender
            n_batches[sender_id] += 1
            mean_loss_receivers[receiver_id] += loss_receiver
            n_batches[receiver_id] += 1

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

        mean_loss_senders = _div_dict(mean_loss_senders, n_batches)
        mean_loss_receivers = _div_dict(mean_loss_receivers, n_batches)

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def train_mutual_information(self):

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

                sender_id, receiver_id = batch[-2], batch[-1]

                if sender_id not in mean_loss_senders:
                    mean_loss_senders[sender_id] = 0.
                    n_batches[sender_id] = 0
                if receiver_id not in mean_loss_receivers:
                    mean_loss_receivers[receiver_id] = 0.
                    n_batches[receiver_id] = 0

                batch = move_to(batch, self.device)
                loss_sender, loss_receiver, metrics = self.game(batch, compute_metrics=compute_metrics)

                mean_loss_senders[sender_id] += loss_sender
                n_batches[sender_id] += 1
                mean_loss_receivers[receiver_id] += loss_receiver
                n_batches[receiver_id] += 1

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

        mean_loss_senders = _div_dict(mean_loss_senders, n_batches)
        mean_loss_receivers = _div_dict(mean_loss_receivers, n_batches)

        if compute_metrics:
            for agt in mean_metrics:
                mean_metrics[agt] = _div_dict(mean_metrics[agt], n_batches[agt])

        return mean_loss_senders, mean_loss_receivers, mean_metrics

    def log_metrics(self,
                    epoch: int,
                    train_loss_senders: dict,
                    train_loss_receivers: dict,
                    mi_loss_senders : dict,
                    train_metrics: dict,
                    val_loss_senders: dict,
                    val_loss_receivers: dict,
                    val_metrics: dict) -> None:

        # Train
        if train_loss_senders is not None:
            for sender, l in train_loss_senders.items():
                self.writer.add_scalar(f'{sender}/Loss train', l.item(),
                                       int(self.population.agents[sender].p_step*epoch))

                self.writer.add_scalar(f'{sender}/accuracy (train)',
                                       train_metrics[sender]['accuracy'],
                                       int(self.population.agents[sender].p_step*epoch))
                self.writer.add_scalar(f'{sender}/Language entropy (train)',
                                       train_metrics[sender]['sender_entropy'],
                                       int(self.population.agents[sender].p_step*epoch))
                self.writer.add_scalar(f'{sender}/Sender log prob',
                                       train_metrics[sender]['sender_log_prob'],
                                       int(self.population.agents[sender].p_step*epoch))
                self.writer.add_scalar(f'{sender}/Messages length (train)',
                                       train_metrics[sender]['message_length'],
                                       int(self.population.agents[sender].p_step*epoch))

            for receiver, l in train_loss_receivers.items():
                self.writer.add_scalar(f'{receiver}/Loss train ', l.item(),
                                       int(self.population.agents[receiver].p_step*epoch))

                self.writer.add_scalar(f'{receiver}/accuracy (train)',
                                       train_metrics[receiver]['accuracy'],
                                       int(self.population.agents[receiver].p_step*epoch))

        # MI
        if mi_loss_senders is not None:
            for sender, l in mi_loss_senders.items():
                self.writer.add_scalar(f'{sender}/Loss Mutual information', l.item(),
                                       int(self.population.agents[sender].p_step*epoch))

        # Val
        if val_loss_senders is not None:
            for sender, l in val_loss_senders.items():
                self.writer.add_scalar(f'{sender}/Loss val', l.item(),
                                       int(self.population.agents[sender].p_step*epoch))

                self.writer.add_scalar(f'{sender}/accuracy (val)',
                                       val_metrics[sender]['accuracy'],
                                       int(self.population.agents[sender].p_step*epoch))
                self.writer.add_scalar(f'{sender}/Language entropy (val)',
                                       val_metrics[sender]['sender_entropy'],
                                       int(self.population.agents[sender].p_step*epoch))
                self.writer.add_scalar(f'{sender}/Messages length (val)',
                                       val_metrics[sender]['message_length'],
                                       int(self.population.agents[sender].p_step*epoch))

            for receiver, l in val_loss_receivers.items():
                self.writer.add_scalar(f'{receiver}/Loss val', l.item(),
                                       int(self.population.agents[receiver].p_step*epoch))

                self.writer.add_scalar(f'{receiver}/accuracy (val)',
                                       val_metrics[receiver]['accuracy'],
                                       int(self.population.agents[receiver].p_step*epoch))


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
                  logger: th.utils.tensorboard.SummaryWriter = None,
                  compute_metrics: bool = False,
                  pretraining: bool = False,
                  device: str = "cpu"):
    if not pretraining:
        trainer = Trainer(game=game,
                          evaluator=evaluator,
                          train_loader=train_loader,
                          mi_loader=mi_loader,
                          val_loader=val_loader,
                          logger=logger,
                          device=device)
    else:
        trainer = PretrainingTrainer(game=game,
                                     train_loader=train_loader,
                                     compute_metrics=compute_metrics,
                                     device=device)

    return trainer
