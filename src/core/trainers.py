import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from .utils import _add_dicts, _div_dict, move_to, find_lengths
from .language_metrics import compute_language_similarity


class Trainer:

    def __init__(self,
                 game,
                 train_loader: th.utils.data.DataLoader,
                 val_loader: th.utils.data.DataLoader = None,
                 log_dir: str = "",
                 compute_metrics: bool = False,
                 print_evolution: bool = True,
                 device: str = "cpu") -> None:

        self.game = game
        self.population = game.population
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_epoch = 0
        self.device = device
        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.compute_metrics = compute_metrics
        self.print_evolution = print_evolution
        self.past_loss_receivers = []
        self.r = []
        self.r_int = []
        self.r_co = []
        self.language_similarity = []
        self.init_languages = None
        self.similarity_to_init_languages = []
        self.divergence_to_untrained_speakers = []

    def init_metrics(self,
                     dump_batch=None,
                     mutual_information:bool=True,
                     synchronization:bool=False,
                     distance_from_init:bool=False,
                     KL_div:bool=True) -> None :


        if dump_batch is not None:
            if mutual_information : self.evaluate_MI(dump_batch, 0)
            if synchronization : self.evaluate_synchronization(dump_batch, n_samples=10, iter=0)
            if distance_from_init : self.evaluate_distance_from_init_language(dump_batch, n_samples=10, iter=0)
            if KL_div : self.evaluate_KL_divergence_with_untrained_speakers(dump_batch,n_samples=50,iter=0)

    def train(self,
              n_epochs,
              dump_freq: int = 10000,
              validation_freq: int = 1000,
              MI_freq: int = 1000,
              l_analysis_freq: int = 1000,
              div_analysis_freq : int = 1000,
              dump_batch=None):

        for epoch in range(self.start_epoch, n_epochs):

            if self.print_evolution: print(f"Epoch {epoch}")

            train_loss_senders, train_loss_receivers, metrics = \
                self.train_epoch(compute_metrics=self.compute_metrics)  # dict,dict, dict

            if self.writer is not None:
                for sender, l in train_loss_senders.items():
                    self.writer.add_scalar(f'{sender}/Loss train', l.item(), epoch)

                    if self.compute_metrics:
                        self.writer.add_scalar(f'{sender}/accuracy (train)',
                                               metrics[sender]['accuracy'],
                                               epoch)
                        self.writer.add_scalar(f'{sender}/Language entropy (train)',
                                               metrics[sender]['sender_entropy'],
                                               epoch)
                        self.writer.add_scalar(f'{sender}/Sender log prob',
                                               metrics[sender]['sender_log_prob'],
                                               epoch)
                        self.writer.add_scalar(f'{sender}/Messages length (train)',
                                               metrics[sender]['message_length'],
                                               epoch)

                for receiver, l in train_loss_receivers.items():
                    self.writer.add_scalar(f'{receiver}/Loss train ', l.item(), epoch)

                    if self.compute_metrics:
                        self.writer.add_scalar(f'{receiver}/accuracy (train)',
                                               metrics[receiver]['accuracy'],
                                               epoch)

            if self.val_loader is not None and epoch+1 % validation_freq == 0:

                val_loss_senders, val_loss_receivers, metrics = self.eval(compute_metrics=self.compute_metrics)

                if self.writer is not None:
                    for sender, l in val_loss_senders.items():
                        self.writer.add_scalar(f'{sender}/Loss val', l.item(), epoch)

                        if self.compute_metrics:
                            self.writer.add_scalar(f'{sender}/accuracy (val)',
                                                   metrics[sender]['accuracy'],
                                                   epoch)
                            self.writer.add_scalar(f'{sender}/Language entropy (val)',
                                                   metrics[sender]['sender_entropy'],
                                                   epoch)
                            self.writer.add_scalar(f'{sender}/Messages length (val)',
                                                   metrics[sender]['message_length'],
                                                   epoch)

                    for receiver, l in val_loss_receivers.items():
                        self.writer.add_scalar(f'{receiver}/Loss val', l.item(), epoch)

                        if self.compute_metrics:
                            self.writer.add_scalar(f'{receiver}/accuracy (train)',
                                                   metrics[receiver]['accuracy'],
                                                   epoch)

            if (epoch+1) % dump_freq == 0:
                if dump_batch is not None:
                    self.dump(dump_batch)

            if (epoch+1) % MI_freq == 0:
                if dump_batch is not None:
                    self.evaluate_MI(dump_batch, iter=(epoch+1) // MI_freq)

            if (epoch+1) % l_analysis_freq == 0 :
                if dump_batch is not None:
                    self.evaluate_synchronization(dump_batch, n_samples=10, iter=(epoch+1)//l_analysis_freq)
                    self.evaluate_distance_from_init_language(dump_batch, n_samples=10, iter=(epoch+1)//l_analysis_freq)

            if (epoch + 1) % div_analysis_freq == 0:
                if dump_batch is not None:
                    self.evaluate_KL_divergence_with_untrained_speakers(dump_batch,
                                                                       n_samples=50,
                                                                       iter=(epoch+1)//div_analysis_freq)

    def train_epoch(self, compute_metrics: bool = False):

        mean_loss_senders = {}
        mean_loss_receivers = {}
        n_batches = {}
        mean_metrics = {}

        self.game.train()

        for batch in self.train_loader:

            sender_id, receiver_id = batch[-2], batch[-1]

            if sender_id not in mean_loss_senders:
                mean_loss_senders[sender_id] = 0.
                n_batches[sender_id] = 0
            if receiver_id not in mean_loss_receivers:
                mean_loss_receivers[receiver_id] = 0.
                n_batches[receiver_id] = 0

            batch = move_to(batch, self.device)
            loss_sender, loss_receiver, metrics = self.game(batch, compute_metrics=compute_metrics)

            self.past_loss_receivers.append(loss_receiver.item())

            if th.rand(1)[0] < self.population.agents[sender_id].p_step:
                # print("speaker update")
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
                                               "sender_log_prob":0.,
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

    def dump(self, batch):

        self.game.eval()

        with th.no_grad():
            batch = move_to(batch, self.device)
            _, _, metrics = self.game(batch, compute_metrics=True)
            for m in metrics["messages"]:
                print(m)

    def evaluate_MI(self, batch, iter: int)->None:

        self.game.train()

        with th.no_grad():

            n_x, n_att, n_val = batch[0].size(0), batch[0].size(1), batch[0].size(2)
            id_sender = batch[1]
            batch = move_to(batch, self.device)
            _, loss_receiver, metrics = self.game(batch, compute_metrics=True)

            id_sampled_messages = np.arange(256)
            n_m = 256
            #id_sampled_messages = np.random.choice(metrics["messages"].size(0), n_m)
            sampled_messages = metrics["messages"][id_sampled_messages]
            sampled_messages = sampled_messages.unsqueeze(0)
            sampled_messages = th.tile(sampled_messages, (n_x, 1,1))
            sampled_messages = sampled_messages.permute(1,0,2)
            sampled_messages = sampled_messages.reshape([n_x*n_m,sampled_messages.size(-1)])
            sampled_x = th.tile(batch[0], (n_m, 1, 1, 1))
            sampled_x = sampled_x.reshape([n_m * n_x, n_att, n_val])

            sampled_x = move_to(sampled_x, self.device)
            sampled_messages = move_to(sampled_messages, self.device)
            log_probs = self.game.population.agents[id_sender].get_log_prob_m_given_x(sampled_x,
                                                                                      sampled_messages)

            # log_probs -> pm1_x1,...,pm1_xn ; pm2_x1,...,pm2_xn,.....
            message_lengths = find_lengths(sampled_messages)
            max_len = sampled_messages.size(1)
            mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
                                               num_classes=max_len + 1), dim=1)[:, :-1]
            log_probs = (log_probs * mask_eos).sum(dim=1)



            log_pi_m_x = log_probs.reshape([n_m, n_x])
            pi_m_x = th.exp(log_pi_m_x)
            p_x = th.ones(n_x) / n_x  # Ici on considère p(x)=1/batch_size

            log_pi_m = th.log((pi_m_x * p_x).sum(1))


            f_pi_m = (th.log(pi_m_x.diagonal(0)) + th.log(p_x)  - log_pi_m)
            pi_l = th.exp(-1*loss_receiver)
            f_pi_l = th.log(pi_l)

            self.r.append(f_pi_l.mean())
            self.r_int.append(-(f_pi_m - f_pi_l).mean())
            self.r_co.append(f_pi_m.mean())


            if self.writer is not None:
                self.writer.add_scalar(f'{id_sender}/f(pi_m_x*p_x / pi_m)', f_pi_m.mean().item(), iter)
            if self.writer is not None:
                self.writer.add_scalar(f'{id_sender}/f(pi_m_x*p_x / pi_m) - f(pi_l)',
                                       -(f_pi_m - f_pi_l).mean().item(), iter)
            if self.writer is not None:
                self.writer.add_scalar(f'{id_sender}/f(pi_l)', f_pi_l.mean().item(), iter)

    def evaluate_synchronization(self,
                                 batch:tuple,
                                 iter: int,
                                 n_samples:int=50,
                                 method:str="edit_distance")->None:

        self.game.train()

        with th.no_grad():

            messages_per_agents=[]
            message_lengths_per_agents = []

            inputs = batch[0]
            inputs = th.tile(inputs,[n_samples]+[1]*(len(inputs.size())-1))
            inputs = move_to(inputs,device=self.device)
            for agent_id in self.population.sender_names:
                inputs_embedding = self.population.agents[agent_id].encode_object(inputs)
                messages, _, _ = self.population.agents[agent_id].send(inputs_embedding)
                message_lengths = find_lengths(messages)
                messages_per_agents.append(messages)
                message_lengths_per_agents.append(message_lengths)

            similarity_matrix = np.identity(len(self.population.sender_names))

            for i in range(len(messages_per_agents)-1):
                for j in range(i+1,len(messages_per_agents)):
                    if method=="edit_distance":
                        similarity_val = compute_language_similarity(messages_1=messages_per_agents[i],
                                                                     messages_2=messages_per_agents[j],
                                                                     len_messages_1=message_lengths_per_agents[i],
                                                                     len_messages_2=message_lengths_per_agents[j]).mean()

                    else:
                        raise Exception("Specify known method")

                    similarity_matrix[i, j] = similarity_val
                    similarity_matrix[j, i] = similarity_val

            self.language_similarity.append(similarity_matrix)

            if self.writer is not None:
                unique_sim = [similarity_matrix[i,j] for i in range(len(similarity_matrix)-1) \
                              for j in range(i+1,len(similarity_matrix))]
                self.writer.add_scalar(f'average_similarity', np.mean(unique_sim), iter)

    def evaluate_distance_from_init_language(self,
                                             batch:tuple,
                                             iter: int,
                                             n_samples:int=50)->None:

        self.game.train()

        with th.no_grad():

            inputs = batch[0]
            inputs = th.tile(inputs, [n_samples] + [1] * (len(inputs.size()) - 1))
            inputs = move_to(inputs, device=self.device)

            if self.init_languages is None:

                self.init_languages=[]
                self.init_languages_lengths = []
                for agent_id in self.population.sender_names:
                    inputs_embedding = self.population.agents[agent_id].encode_object(inputs)
                    messages, _, _ = self.population.agents[agent_id].send(inputs_embedding)
                    message_lengths = find_lengths(messages)
                    self.init_languages.append(messages)
                    self.init_languages_lengths.append(message_lengths)

            messages_per_agents = []
            message_lengths_per_agents = []

            for agent_id in self.population.sender_names:
                inputs_embedding = self.population.agents[agent_id].encode_object(inputs)
                messages, _, _ = self.population.agents[agent_id].send(inputs_embedding)
                message_lengths = find_lengths(messages)
                messages_per_agents.append(messages)
                message_lengths_per_agents.append(message_lengths)

            similarity_to_init_languages = np.zeros((len(self.population.sender_names),
                                                     len(self.population.sender_names)))

            for i in range(len(messages_per_agents)):
                for j in range(len(self.init_languages)):
                    similarity_val = compute_language_similarity(messages_1=messages_per_agents[i],
                                                                 messages_2=self.init_languages[j],
                                                                 len_messages_1=message_lengths_per_agents[i],
                                                                 len_messages_2=self.init_languages_lengths[j]).mean()
                    similarity_to_init_languages[i, j] = similarity_val

            self.similarity_to_init_languages.append(similarity_to_init_languages)

            if self.writer is not None:
                for i,agent_id in enumerate(self.population.sender_names):
                    for j, agent_id_init in enumerate(self.population.sender_names):
                        self.writer.add_scalar(f'{agent_id}/similarity_to_{agent_id_init}',
                                               similarity_to_init_languages[i,j],
                                               iter)

    def evaluate_KL_divergence_with_untrained_speakers(self,
                                                     batch:tuple,
                                                     iter: int,
                                                     n_samples:int=50)->None:

        self.game.train()

        with th.no_grad():
            inputs = batch[0]
            inputs = th.tile(inputs, [n_samples] + [1] * (len(inputs.size()) - 1))
            inputs = move_to(inputs, device=self.device)

            divergence_matrix=np.zeros((len(self.population.sender_names),
                                        len(self.population.untrained_sender_names)))

            for i, agent_id_p in enumerate(self.population.sender_names):
                inputs_embedding = self.population.agents[agent_id_p].encode_object(inputs)
                messages_1, log_prob_1, _ = self.population.agents[agent_id_p].send(inputs_embedding)
                message_lengths_1 = find_lengths(messages_1)
                max_len = messages_1.size(1)
                mask_eos = 1 - th.cumsum(F.one_hot(message_lengths_1.to(th.int64),
                                                   num_classes=max_len + 1), dim=1)[:, :-1]
                log_prob_1 = (log_prob_1 * mask_eos).sum(dim=1)

                for j, agent_id_q in enumerate(self.population.untrained_sender_names):
                    log_prob_2 = self.game.population.agents[agent_id_q].get_log_prob_m_given_x(inputs,
                                                                                                messages_1)

                    log_prob_2 = (log_prob_2 * mask_eos).sum(dim=1)

                    divergence_matrix[i,j] = th.sum(th.exp(log_prob_1)*(log_prob_1 - log_prob_2)).item()/inputs.size(0)

        self.divergence_to_untrained_speakers.append(divergence_matrix)

        if self.writer is not None:
            for i, agent_id in enumerate(self.population.sender_names):
                for j, agent_id_untrained in enumerate(self.population.untrained_sender_names):
                    self.writer.add_scalar(f'{agent_id}/div_to_{agent_id_untrained}',
                                           divergence_matrix[i, j],
                                           iter)

                    self.writer.add_scalar(f'div/div_to_{agent_id_untrained}',
                                           divergence_matrix[i, j],
                                           iter)


    def save_metrics(self,save_dir):

        np.save(f"{save_dir}/reward.npy",self.r)
        np.save(f"{save_dir}/reward_inf.npy", self.r_int)
        np.save(f"{save_dir}/reward_co.npy", self.r_co)
        np.save(f"{save_dir}/similarity_matrix.npy", np.stack(self.language_similarity))
        np.save(f"{save_dir}/similarity_to_init_languages.npy", np.stack(self.similarity_to_init_languages))
        np.save(f"{save_dir}/divergence_to_untrained_speakers.npy", np.stack(self.divergence_to_untrained_speakers))

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
                  train_loader: th.utils.data.DataLoader,
                  val_loader: th.utils.data.DataLoader,
                  log_dir: str = "",
                  compute_metrics: bool = False,
                  pretraining: bool = False,
                  device: str = "cpu"):
    if not pretraining:
        trainer = Trainer(game=game,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          log_dir=log_dir,
                          compute_metrics=compute_metrics,
                          device=device)
    else:
        trainer = PretrainingTrainer(game=game,
                                     train_loader=train_loader,
                                     log_dir=log_dir,
                                     compute_metrics=compute_metrics,
                                     device=device)

    return trainer
