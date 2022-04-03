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
        self.mi_step = 0

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

    def train_mutual_information(self, threshold=1e-3):

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
            batch = move_to((inputs, sender_id, optimal_listener_id), self.device)

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

        return {sender_id: agent_sender.tasks[task]["loss_value"].item()}

    def train_mutual_information_with_lm(self, threshold=1e-3):

        self.game.train()

        # Language model training
        # messages_lm = []
        # for sender_id in self.population.sender_names:

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

        self.mi_step += 1

        return {sender_id: agent_sender.tasks["mutual_information"]["loss_value"].item()}

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

    def save_error(self, epoch: int, save: bool = False):

        task = "communication"

        self.game.train()

        with th.no_grad():
            batch = next(iter(self.train_loader))
            inputs, sender_id, receiver_id = batch.data, batch.sender_id, batch.receiver_id
            agent_sender = self.population.agents[sender_id]
            agent_receiver = self.population.agents[batch.receiver_id]
            optimal_listener_id = agent_sender.optimal_listener
            optimal_listener = self.population.agents[optimal_listener_id]
            batch = move_to((inputs, sender_id, receiver_id), self.device)

            _ = self.game(batch,
                          reduce=False,
                          compute_metrics=True)

            reward_distrib = -1 * agent_receiver.tasks[task]["loss_value"].cpu()

            batch = move_to((inputs, sender_id, optimal_listener_id), self.device)

            _ = self.game(batch,
                          reduce=False,
                          compute_metrics=True)

            mi_distrib = -1 * optimal_listener.tasks[task]["loss_value"].cpu()

        self.reward_distrib.append(reward_distrib)
        self.mi_distrib.append(mi_distrib)
        self.input_batch.append(inputs)
        self.pi_x_m.append(metrics["sender_log_prob"].cpu())

        if save:
            np.save("{}/reward_distrib_{}".format(self.metrics_save_dir, epoch), th.stack(self.reward_distrib))
            np.save("{}/mi_distrib_{}".format(self.metrics_save_dir, epoch), th.stack(self.mi_distrib))
            np.save("{}/inputs_{}".format(self.metrics_save_dir, epoch), th.stack(self.input_batch))
            np.save("{}/pi_x_m_{}".format(self.metrics_save_dir, epoch), th.stack(self.pi_x_m))

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



# Formerly in agents


    def compute_mutual_information(self,inputs):

        raise NotImplementedError

    #def compute_mutual_information(self, inputs):
    #    inputs_embeddings = self.encode_object(inputs)
    #    messages, log_prob_sender, entropy_sender = self.send(inputs_embeddings)
    #    batch_size = messages.size(0)

    #    id_sampled_messages = np.arange(batch_size)
    #    sampled_messages = messages[id_sampled_messages]
    #    sampled_messages = sampled_messages.unsqueeze(0)
    #    sampled_messages = sampled_messages.repeat(batch_size, 1, 1)
    #    sampled_messages = sampled_messages.permute(1, 0, 2)
    #    sampled_messages = sampled_messages.reshape([batch_size * batch_size, sampled_messages.size(-1)])
    #    sampled_x = inputs.repeat(batch_size, 1, 1, 1)
    #    sampled_x = sampled_x.reshape([batch_size * batch_size, *inputs.size()[1:]])

    #    sampled_x = move_to(sampled_x, self.device)
    #    sampled_messages = move_to(sampled_messages, self.device)
    #    log_probs = self.get_log_prob_m_given_x(sampled_x, sampled_messages)

        # log_probs -> pm1_x1,...,pm1_xn ; pm2_x1,...,pm2_xn,.....
    #    message_lengths = find_lengths(sampled_messages)
    #    max_len = sampled_messages.size(1)
    #    mask_eos = 1 - th.cumsum(F.one_hot(message_lengths.to(th.int64),
    #                                       num_classes=max_len + 1), dim=1)[:, :-1]
    #    log_probs = (log_probs * mask_eos).sum(dim=1)

    #    log_pi_m_x = log_probs.reshape([batch_size, batch_size])
    #    pi_m_x = th.exp(log_pi_m_x)
    #    p_x = th.ones(batch_size) / batch_size  # Here we set p(x)=1/batch_size
    #    p_x = p_x.to(pi_m_x.device)  # Fix device issue

    #    log_p_x = th.log(p_x)
    #    log_pi_m = th.log((pi_m_x * p_x).sum(1))
    #    log_pi_m_x = th.log(pi_m_x.diagonal(0))

    #    mutual_information = log_pi_m_x + log_p_x - log_pi_m

    #    reward = mutual_information.detach()
    #    reward = (reward - reward.mean()) / reward.std()

    #    loss_mi = - log_pi_m_x * reward

    #    return loss_mi.mean()