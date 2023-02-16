import torch as th
import numpy as np
import itertools
import collections

CommunicationBatch = collections.namedtuple("CommunicationBatch", ["data", "sender_id", "receiver_id"])
ImitationBatch = collections.namedtuple("ImitationBatch", ["data", "sender_id", "imitator_id"])
MIBatch = collections.namedtuple("MIBatch", ["data", "sender_id"])
BroadcastingBatch = collections.namedtuple("BroadcastingBatch", ["data", "sender_id", "receiver_ids"])


class ReconstructionDataLoader(th.utils.data.DataLoader):

    def __init__(self,
                 data,
                 agent_names,
                 population_probs: th.Tensor,
                 population_split: dict,
                 batches_per_epoch: int,
                 batch_size: int,
                 imitation_probs: th.Tensor = None,
                 task: str = "communication",
                 broadcasting: bool = False,
                 mode: str = "train",
                 seed: int = None):

        self.data = data
        self.agent_names = agent_names
        self.population_probs = population_probs
        self.imitation_probs = imitation_probs
        self.population_split = population_split
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.task = task
        self.broadcasting = broadcasting
        self.mode = mode
        self.seed = seed
        if seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

    def __iter__(self):

        return _ReconstructionIterator(data=self.data,
                                       agent_names=self.agent_names,
                                       population_probs=self.population_probs,
                                       imitation_probs=self.imitation_probs,
                                       population_split=self.population_split,
                                       n_batches_per_epoch=self.batches_per_epoch,
                                       batch_size=self.batch_size,
                                       task=self.task,
                                       broadcasting=self.broadcasting,
                                       mode=self.mode,
                                       random_state=self.random_state)


class _ReconstructionIterator():

    def __init__(self,
                 data,
                 agent_names,
                 population_probs,
                 population_split,
                 n_batches_per_epoch,
                 batch_size,
                 imitation_probs: th.Tensor = None,
                 task: str = "communication",
                 broadcasting: bool = False,
                 mode: str = "train",
                 random_state=None):

        self.data = data
        self.agent_names = agent_names
        self.grid_names = [(agent_names[i], agent_names[j]) for i in range(len(agent_names)) \
                           for j in range(len(agent_names))]
        self.population_probs = population_probs.flatten()
        self.imitation_probs = imitation_probs
        self.population_split = population_split
        self.n_batches_per_epoch = n_batches_per_epoch
        self.broadcasting = broadcasting
        self.batch_size = batch_size
        self.batches_generated = 0
        self.mode =  mode
        self.task = task
        self.random_state = random_state

    def __iter__(self):
        return self

    def __next__(self):

        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        # Sample pair sender_id, receiver_id
        sampled_pair_id = th.multinomial(self.population_probs, 1)
        sender_id, receiver_id = self.grid_names[sampled_pair_id]
        if self.imitation_probs is not None:
            imitator_id = self.agent_names[th.multinomial(self.imitation_probs, 1)[0]]
        else:
            imitator_id = None

        # Sample batch from sender_id's split
        split_ids = self.population_split[sender_id]["{}_split".format(self.mode)]
        batch_ids = self.random_state.choice(len(split_ids),
                                             size=min(self.batch_size, len(split_ids)),
                                             replace=False)

        batch_data = self.data[split_ids[batch_ids]]

        self.batches_generated += 1

        if self.task == "communication":

            if self.broadcasting and self.mode=="train":
                receiver_ids = [pair[1] for j, pair in enumerate(self.grid_names)
                                if pair[0] == sender_id and self.population_probs[j] > 0]

                batch = BroadcastingBatch(data=batch_data,
                                          sender_id=sender_id,
                                          receiver_ids=receiver_ids)
            else:
                batch = CommunicationBatch(data=batch_data,
                                           sender_id=sender_id,
                                           receiver_id=receiver_id)
        elif self.task == "imitation":
            batch = ImitationBatch(data=batch_data,
                                   sender_id=sender_id,
                                   imitator_id=imitator_id)
        elif self.task == "MI":
            batch = MIBatch(data=batch_data,
                            sender_id=sender_id)

        return batch


class ReferentialDataLoader(th.utils.data.DataLoader):

    def __init__(self,
                 data,
                 agent_names,
                 population_probs: th.Tensor,
                 population_split: dict,
                 batches_per_epoch: int,
                 batch_size: int,
                 n_distractors: int,
                 mode: str = "train",
                 seed: int = None):

        self.data = data
        self.agent_names = agent_names
        self.population_probs = population_probs
        self.population_split = population_split
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.n_distractors = n_distractors
        self.mode = mode
        self.seed = seed

    def __iter__(self):

        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed

        return _ReferentialIterator(data=self.data,
                                    agent_names=self.agent_names,
                                    population_probs=self.population_probs,
                                    population_split=self.population_split,
                                    n_batches_per_epoch=self.batches_per_epoch,
                                    batch_size=self.batch_size,
                                    n_distractors=self.n_distractors,
                                    mode=self.mode,
                                    seed=seed)


class _ReferentialIterator():

    def __init__(self,
                 data,
                 agent_names,
                 population_probs,
                 population_split: dict,
                 n_batches_per_epoch: int,
                 batch_size: int,
                 n_distractors: int,
                 mode: str = "train",
                 seed: int = 10):
        self.data = data
        self.agent_names = agent_names
        self.grid_names = [(agent_names[i], agent_names[j]) for i in range(len(agent_names)) \
                           for j in range(len(agent_names))]
        self.population_probs = population_probs.flatten()
        self.population_split = population_split
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batch_size = batch_size
        self.n_distractors = n_distractors
        self.batches_generated = 0
        self.mode = mode
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        # Sample pair sender_id, receiver_id
        sampled_pair_id = th.multinomial(self.population_probs, 1)
        sender_id, receiver_id = self.grid_names[sampled_pair_id]

        # Sample batch of 1 object to communicate from sender_id's split
        split_ids_sender = self.population_split[sender_id]["{}_split".format(self.mode)]
        batch_ids_sender = self.random_state.choice(len(split_ids_sender),
                                                    size=self.batch_size,
                                                    replace=True)

        batch_data = self.data[split_ids_sender[batch_ids_sender]]

        # Sample batch of n_distractors objects to from receiver_id's split
        split_ids_receiver = self.population_split[receiver_id]["{}_split".format(self.mode)]
        batch_distractors_ids_receiver = self.random_state.choice(len(split_ids_receiver),
                                                                  size=self.batch_size * self.n_distractors,
                                                                  replace=True)

        distractors_data = self.data[split_ids_receiver[batch_distractors_ids_receiver]]

        distractors_data = distractors_data.reshape((self.batch_size,
                                                     self.n_distractors,
                                                     distractors_data.size(-2),
                                                     distractors_data.size(-1)))

        self.batches_generated += 1

        return batch_data, distractors_data, sender_id, receiver_id


class UnidirectionalDataLoader(th.utils.data.DataLoader):

    def __init__(self,
                 data: th.Tensor,
                 target_messages: th.Tensor,
                 batches_per_epoch: int,
                 batch_size: int,
                 mode: str = "train",
                 seed: int = None):

        self.data = data
        self.target_messages = target_messages
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.mode = mode
        self.seed = seed

    def __iter__(self):

        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed

        return _UnidirectionalIterator(data=self.data,
                                       target_messages=self.target_messages,
                                       n_batches_per_epoch=self.batches_per_epoch,
                                       batch_size=self.batch_size,
                                       mode=self.mode,
                                       seed=seed)


class _UnidirectionalIterator():

    def __init__(self,
                 data,
                 target_messages,
                 n_batches_per_epoch,
                 batch_size,
                 mode: str = "train",
                 seed: int = 10):
        self.data = data
        self.target_messages = target_messages
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batch_size = batch_size
        self.batches_generated = 0
        self.mode = mode
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        batch_ids = self.random_state.choice(len(self.data),
                                             size=self.batch_size,
                                             replace=True)

        batch_data = self.data[batch_ids]
        target_messages = self.target_messages[batch_ids]

        self.batches_generated += 1

        return batch_data, target_messages

class ReconstructionDataLoaderSpecificDistribution(th.utils.data.DataLoader):

    def __init__(self,
                 object_params,
                 agent_names,
                 population_probs: th.Tensor,
                 population_split: dict,
                 batches_per_epoch: int,
                 batch_size: int,
                 imitation_probs: th.Tensor = None,
                 task: str = "communication",
                 broadcasting: bool = False,
                 mode: str = "train",
                 seed: int = None):

        self.object_params = object_params
        self.agent_names = agent_names
        self.population_probs = population_probs
        self.imitation_probs = imitation_probs
        self.population_split = population_split
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.task = task
        self.broadcasting = broadcasting
        self.mode = mode
        self.seed = seed
        if seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        n_attributes = object_params["n_attributes"]
        n_values_per_attribute = object_params["n_values_per_attribute"]
        attribute_distributions = object_params["attribute_distributions"]
        max_values = np.max(n_values_per_attribute)

        # Building probability over values for each attribute

        attribute_probs = []

        for idx_attribute in range(n_attributes):
            if attribute_distributions[idx_attribute] == "uniform":
                probs = [1] * n_values_per_attribute[idx_attribute]
                probs += [0] * (max_values - n_values_per_attribute[idx_attribute])
                probs = np.array(probs, dtype=np.float32)
            elif attribute_distributions[idx_attribute] == "powerlaw":
                probs = 1 / np.arange(1, n_values_per_attribute[idx_attribute] + 1, dtype=np.float32)
                probs = np.concatenate((probs, [0.] * (max_values - n_values_per_attribute[idx_attribute])))
            else:
                raise "Specify a know distribution"

            probs /= np.sum(probs)

            attribute_probs.append(probs)

        attribute_probs = th.Tensor(np.stack(attribute_probs, axis=0))

        self.attribute_probs = attribute_probs
        self.max_values = max_values


    def __iter__(self):

        return _ReconstructionIteratorSpecificDistribution(attribute_probs=self.attribute_probs,
                                                           max_values=self.max_values,
                                                           agent_names=self.agent_names,
                                                           population_probs=self.population_probs,
                                                           imitation_probs=self.imitation_probs,
                                                           population_split=self.population_split,
                                                           n_batches_per_epoch=self.batches_per_epoch,
                                                           batch_size=self.batch_size,
                                                           task=self.task,
                                                           broadcasting=self.broadcasting,
                                                           mode=self.mode,
                                                           random_state=self.random_state)


class _ReconstructionIteratorSpecificDistribution():

    def __init__(self,
                 attribute_probs,
                 max_values,
                 agent_names,
                 population_probs,
                 population_split,
                 n_batches_per_epoch,
                 batch_size,
                 imitation_probs: th.Tensor = None,
                 task: str = "communication",
                 broadcasting: bool = False,
                 mode: str = "train",
                 random_state=None):

        self.attribute_probs = attribute_probs
        self.max_values = max_values
        self.agent_names = agent_names
        self.grid_names = [(agent_names[i], agent_names[j]) for i in range(len(agent_names)) \
                           for j in range(len(agent_names))]
        self.population_probs = population_probs.flatten()
        self.imitation_probs = imitation_probs
        self.population_split = population_split
        self.n_batches_per_epoch = n_batches_per_epoch
        self.broadcasting = broadcasting
        self.batch_size = batch_size
        self.batches_generated = 0
        self.mode =  mode
        self.task = task
        self.random_state = random_state

    def __iter__(self):
        return self

    def __next__(self):

        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        # Build the one-hot dataset by sampling over the probs

        dataset = th.stack([th.multinomial(attribute_probs[i], num_samples=n_elements, replacement=True) \
                            for i in range(n_attributes)], dim=1).to(th.int64)

        dataset = th.nn.functional.one_hot(dataset, num_classes=n_elements)




        # Sample pair sender_id, receiver_id
        sampled_pair_id = th.multinomial(self.population_probs, 1)
        sender_id, receiver_id = self.grid_names[sampled_pair_id]
        if self.imitation_probs is not None:
            imitator_id = self.agent_names[th.multinomial(self.imitation_probs, 1)[0]]
        else:
            imitator_id = None


        self.batches_generated += 1

        if self.task == "communication":
            batch = CommunicationBatch(data=batch_data,
                                       sender_id=sender_id,
                                       receiver_id=receiver_id)
        elif self.task == "imitation":
            batch = ImitationBatch(data=batch_data,
                                   sender_id=sender_id,
                                   imitator_id=imitator_id)
        elif self.task == "MI":
            batch = MIBatch(data=batch_data,
                            sender_id=sender_id)

        if self.broadcasting:
            receiver_ids = [pair[1] for j, pair in enumerate(self.grid_names)
                            if pair[0] == sender_id and self.population_probs[j] > 0]

            batch = BroadcastingBatch(data=batch_data,
                                      sender_id=sender_id,
                                      receiver_ids=receiver_ids)

        return batch

def build_one_hot_dataset(object_params: dict, n_elements: int) -> th.Tensor:
    n_attributes = object_params["n_attributes"]
    n_values = object_params["n_values"]

    if object_params["n_attributes"] ** object_params["n_values"] == n_elements:

        dataset = th.Tensor(list(itertools.product(th.arange(n_values), repeat=n_attributes))).to(th.int64)
        dataset = th.nn.functional.one_hot(dataset, num_classes=n_values)

    else:
        dataset = []

        count = 0
        while count < n_elements:
            el = [np.random.choice(n_values) for _ in range(n_attributes)]
            if el not in dataset:
                dataset.append(el)
                count += 1

        dataset = th.stack([th.Tensor(data) for data in dataset]).to(th.int64)

        dataset = th.nn.functional.one_hot(dataset, num_classes=n_values)

    return dataset

def build_one_hot_dataset_with_specific_distribution(object_params: dict, n_elements: int) -> th.Tensor:

    n_attributes = object_params["n_attributes"]
    n_values_per_attribute = object_params["n_values_per_attribute"]
    attribute_distributions = object_params["attribute_distributions"]
    max_values = np.max(n_values_per_attribute)

    # Building probability over values for each attribute

    attribute_probs = []

    for idx_attribute in range(n_attributes):
        if attribute_distributions[idx_attribute] == "uniform":
            probs = [1] * n_values_per_attribute[idx_attribute]
            probs += [0] * (max_values - n_values_per_attribute[idx_attribute])
            probs = np.array(probs, dtype=np.float32)
        elif attribute_distributions[idx_attribute] == "powerlaw":
            probs = 1 / np.arange(1, n_values_per_attribute[idx_attribute] + 1, dtype=np.float32)
            probs = np.concatenate((probs, [0.] * (max_values - n_values_per_attribute[idx_attribute])))
        elif attribute_distributions[idx_attribute] == "ramp":
            probs = np.arange(1, n_values_per_attribute[idx_attribute] + 1, dtype=np.float32)[::-1]
            probs = np.concatenate((probs, [0.] * (max_values - n_values_per_attribute[idx_attribute])))
        else:
            raise "Specify a know distribution"

        probs /= np.sum(probs)

        attribute_probs.append(probs)

    attribute_probs = th.Tensor(np.stack(attribute_probs, axis=0))

    # Build the one-hot dataset by sampling over the probs

    dataset = th.stack([th.multinomial(attribute_probs[i], num_samples=n_elements, replacement=True) \
                        for i in range(n_attributes)], dim=1).to(th.int64)

    dataset = th.nn.functional.one_hot(dataset, num_classes=max_values)

    return dataset

def get_all_one_hot_elements(object_params: dict) -> th.Tensor:
    n_attributes = object_params["n_attributes"]
    n_values_per_attribute = object_params["n_values_per_attribute"]
    max_values = np.max(n_values_per_attribute)

    product_list=(np.arange(n_values_per_attribute[i]) for i in range(n_attributes))

    dataset = th.Tensor(list(itertools.product(*product_list))).to(th.int64)

    dataset = th.nn.functional.one_hot(dataset, num_classes=max_values)

    return dataset



def build_target_messages(n_elements: int, pretrained_language: str, channel_params: dict) -> th.Tensor:
    if pretrained_language is None:
        target_messages = th.zeros((n_elements, channel_params["max_len"]), dtype=th.int64)

    else:
        raise Exception("Specify a known pretrained language")

    return target_messages


def split_data_into_population(dataset_size: int,
                               n_elements: int,
                               agent_names: list,
                               split_proportion: float = 0.8,
                               population_dataset_type: str = "unique",
                               total_number_elements : int = None,
                               seed: int = 19) -> dict:
    data_split = {}

    if population_dataset_type == "unique":
        random_permut = np.random.RandomState(seed).choice(dataset_size, size=n_elements, replace=False)
        N_element_train = int(split_proportion * n_elements)
        N_element_test = N_element_val = int(split_proportion + 0.5 * (1 - split_proportion) * n_elements)
        train_split, val_split, test_split = random_permut[:N_element_train], \
                                             random_permut[N_element_train:N_element_train + N_element_val], \
                                             random_permut[N_element_train + N_element_val:]

        for agent_name in agent_names:
            data_split[agent_name] = {}
            data_split[agent_name]["train_split"] = train_split
            data_split[agent_name]["val_split"] = val_split
            data_split[agent_name]["test_split"] = test_split
            data_split[agent_name]["MI_split"] = random_permut

    elif population_dataset_type=="expe_non_uniform":
        
        for agent_name in agent_names:
            data_split[agent_name] = {}
            data_split[agent_name]["train_split"] = np.arange(dataset_size)
            data_split[agent_name]["val_split"] = np.arange(total_number_elements)
            data_split[agent_name]["test_split"] = np.arange(total_number_elements)

    else:
        raise "Specify a known population dataset type"

    return data_split


def save_dataset(dataset_save_dir: str,
                 full_dataset: th.Tensor,
                 population_split: dict) -> None:

    th.save(full_dataset, f"{dataset_save_dir}/full_dataset.pt")

    for agent_name in population_split:
        if "train_split" in population_split[agent_name]:
            th.save(population_split[agent_name]["train_split"], f"{dataset_save_dir}/{agent_name}_train_split.pt")
        if "val_split" in population_split[agent_name]:
            th.save(population_split[agent_name]["val_split"], f"{dataset_save_dir}/{agent_name}_val_split.pt")
        if "test_split" in population_split[agent_name]:
            th.save(population_split[agent_name]["test_split"], f"{dataset_save_dir}/{agent_name}_test_split.pt")
        if "MI_split" in population_split[agent_name]:
            th.save(population_split[agent_name]["MI_split"], f"{dataset_save_dir}/{agent_name}_MI_split.pt")


def build_one_hot_dataloader(game_type: str,
                             dataset: th.Tensor,
                             training_params: dict,
                             agent_names: list = None,
                             population_split: dict = None,
                             population_probs=None,
                             imitation_probs=None,
                             task: str = "communication",
                             mode: str = "train",
                             target_messages: th.Tensor = None,  # If pretraining mode
                             ) -> th.utils.data.DataLoader:
    if game_type in ["reconstruction","reconstruction_reinforce"]:

        if "broadcasting" in training_params and mode != "val":
            broadcasting = training_params["broadcasting"]
        else:
            broadcasting = False

        if task == "communication":

            loader = ReconstructionDataLoader(data=dataset,
                                              agent_names=agent_names,
                                              population_split=population_split,
                                              population_probs=population_probs,
                                              batch_size=training_params["batch_size"],
                                              batches_per_epoch=training_params[
                                                      "{}_batches_per_epoch".format(mode)],
                                              task=task,
                                              mode=mode,
                                              broadcasting=broadcasting,
                                              seed=training_params["seed"])

        elif task == "imitation":
            loader = ReconstructionDataLoader(data=dataset,
                                              agent_names=agent_names,
                                              population_split=population_split,
                                              population_probs=population_probs,
                                              imitation_probs=imitation_probs,
                                              batch_size=training_params["batch_size"],
                                              batches_per_epoch=training_params["batches_per_epoch"],
                                              task=task,
                                              mode=mode,
                                              seed=training_params["seed"])

        elif task == "MI":
            loader = ReconstructionDataLoader(data=dataset,
                                              agent_names=agent_names,
                                              population_split=population_split,
                                              population_probs=population_probs,
                                              batch_size=training_params["MI_batch_size"],
                                              batches_per_epoch=training_params["MI_batches_per_epoch"],
                                              task=task,
                                              mode=mode,
                                              seed=training_params["seed"])
        else:
            raise BaseException("Specify a known task")


    elif game_type == "referential":

        loader = ReferentialDataLoader(data=dataset,
                                       agent_names=agent_names,
                                       population_split=population_split,
                                       population_probs=population_probs,
                                       batch_size=training_params["batch_size"],
                                       batches_per_epoch=training_params["batches_per_epoch"],
                                       n_distractors=training_params["n_distractors"],
                                       mode=mode,
                                       seed=training_params["seed"])

    elif game_type == "speaker_pretraining":

        loader = UnidirectionalDataLoader(data=dataset,
                                          target_messages=target_messages,
                                          batch_size=training_params["batch_size"],
                                          batches_per_epoch=training_params["batches_per_epoch"],
                                          mode=mode,
                                          seed=training_params["seed"])

    else:
        raise "Specify a known game type"

    return loader
