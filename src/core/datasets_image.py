import torch as th
import os
import numpy as np
import itertools
import collections

CommunicationBatch = collections.namedtuple("CommunicationBatch", ["data", "sender_id", "receiver_id"])
CommunicationBatchWithAttributes = collections.namedtuple("CommunicationBatchWithAttributes",
                                                          ["data","attributes", "sender_id", "receiver_id"])
ImitationBatch = collections.namedtuple("ImitationBatch", ["data", "sender_id", "imitator_id"])
MIBatch = collections.namedtuple("MIBatch", ["data", "sender_id"])
BroadcastingBatch = collections.namedtuple("BroadcastingBatch", ["data", "sender_id", "receiver_ids"])


class _ReferentialIterator():

    def __init__(self,
                 dataset_dir: str,
                 agent_names: list,
                 files: list,
                 n_files: int,
                 population_probs: th.Tensor,
                 n_batches_per_epoch: int,
                 batch_size: int,
                 task: str = "communication",
                 broadcasting: bool = False,
                 mode: str = "train",
                 random_state=None) -> None:

        self.dataset_dir = dataset_dir
        self.agent_names = agent_names
        self.grid_names = [(agent_names[i], agent_names[j]) for i in range(len(agent_names)) \
                           for j in range(len(agent_names))]
        self.population_probs = population_probs.flatten()
        self.n_batches_per_epoch = n_batches_per_epoch
        self.broadcasting = broadcasting
        self.batch_size = batch_size
        self.batches_generated = 0
        self.mode = mode
        self.files = files
        self.n_files = n_files
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

        # Sample batch from sender_id's split

        # Sample random file
        random_file_id = np.random.choice(self.n_files)
        random_file = th.load(f"{self.dataset_dir}/{self.files[random_file_id]}")

        # Select random split inside the file
        random_samples_ids = np.random.choice(len(random_file), self.batch_size, replace=False)
        batch_data = th.Tensor(np.array([sample["logit"] for sample in np.array(random_file)[random_samples_ids]]))

        self.batches_generated += 1

        if self.task == "communication":
            batch = CommunicationBatch(data=batch_data,
                                       sender_id=sender_id,
                                       receiver_id=receiver_id)
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


class ReferentialDataLoader(th.utils.data.DataLoader):

    def __init__(self,
                 dataset_dir: str,
                 agent_names: list,
                 population_probs: th.Tensor,
                 batches_per_epoch: int,
                 batch_size: int,
                 mode: str = "train",
                 seed: int = None):

        self.dataset_dir = dataset_dir
        self.agent_names = agent_names
        self.population_probs = population_probs
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.mode = mode
        self.files = [f for f in os.listdir(dataset_dir) if mode in f]
        self.n_files = len(self.files)
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

    def __iter__(self):
        return _ReferentialIterator(dataset_dir=self.dataset_dir,
                                    agent_names=self.agent_names,
                                    files=self.files,
                                    n_files=self.n_files,
                                    population_probs=self.population_probs,
                                    n_batches_per_epoch=self.batches_per_epoch,
                                    batch_size=self.batch_size,
                                    mode=self.mode,
                                    random_state=self.random_state)


class _ReferentialIteratorMemory():

    def __init__(self,
                 dataset_dir: str,
                 agent_names: list,
                 files: list,
                 n_files: int,
                 population_probs: th.Tensor,
                 n_batches_per_epoch: int,
                 batch_size: int,
                 task: str = "communication",
                 broadcasting: bool = False,
                 mode: str = "train",
                 return_attributes : bool = False,
                 random_state=None) -> None:

        self.dataset_dir = dataset_dir
        self.agent_names = agent_names
        self.grid_names = [(agent_names[i], agent_names[j]) for i in range(len(agent_names)) \
                           for j in range(len(agent_names))]
        self.population_probs = population_probs.flatten()
        self.n_batches_per_epoch = n_batches_per_epoch
        self.broadcasting = broadcasting
        self.batch_size = batch_size
        self.batches_generated = 0
        self.mode = mode
        self.files = files
        self.n_files = n_files
        self.task = task
        self.return_attributes = return_attributes
        self.random_state = random_state

    def __iter__(self):
        return self

    def __next__(self):

        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        # Sample pair sender_id, receiver_id
        sampled_pair_id = th.multinomial(self.population_probs, 1)
        sender_id, receiver_id = self.grid_names[sampled_pair_id]

        # Sample batch from sender_id's split

        # Sample random file
        random_file_id = np.random.choice(self.n_files)
        random_file = self.files[random_file_id]

        # Select random split inside the file
        random_samples_ids = np.random.choice(len(random_file), self.batch_size, replace=False)
        batch_data = th.Tensor(np.array([sample["logit"] for sample in np.array(random_file)[random_samples_ids]]))
        if self.return_attributes:
            attributes = th.Tensor(np.array([sample["attributes"] for sample in np.array(random_file)[random_samples_ids]]))

        self.batches_generated += 1

        if self.task == "communication":
            if self.return_attributes:
                batch = CommunicationBatchWithAttributes(data=batch_data,
                                                         attributes=attributes,
                                                         sender_id=sender_id,
                                                         receiver_id=receiver_id)
            else:
                batch = CommunicationBatch(data=batch_data,
                                           sender_id=sender_id,
                                           receiver_id=receiver_id)
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


class ReferentialDataLoaderMemory(th.utils.data.DataLoader):

    def __init__(self,
                 dataset_dir: str,
                 agent_names: list,
                 population_probs: th.Tensor,
                 batches_per_epoch: int,
                 batch_size: int,
                 n_files : int = None,
                 mode: str = "train",
                 seed: int = None,
                 return_attributes : bool = False):

        self.dataset_dir = dataset_dir
        self.agent_names = agent_names
        self.population_probs = population_probs
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.return_attributes = return_attributes
        self.mode = mode
        if n_files is not None:
            self.n_files = n_files
            self.files=[]
            for f in os.listdir(dataset_dir):
                if mode in f and len(self.files)<self.n_files:
                    self.files.append(th.load(f"{dataset_dir}/{f}"))

        else:
            self.files = [th.load(f"{dataset_dir}/{f}") for f in os.listdir(dataset_dir) if mode in f]
            self.n_files = len(self.files)

        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

    def __iter__(self):
        return _ReferentialIteratorMemory(dataset_dir=self.dataset_dir,
                                            agent_names=self.agent_names,
                                            files=self.files,
                                            n_files=self.n_files,
                                            population_probs=self.population_probs,
                                            n_batches_per_epoch=self.batches_per_epoch,
                                            batch_size=self.batch_size,
                                            mode=self.mode,
                                            random_state=self.random_state,
                                            return_attributes=self.return_attributes)


def build_image_dataloader(game_type: str,
                           dataset_dir: str,
                           training_params: dict,
                           agent_names: list = None,
                           population_probs=None,
                           n_files: int = None,
                           mode: str = "train",
                           ) -> th.utils.data.DataLoader:
    if game_type == "referential" or game_type =="visual_reconstruction":

        loader = ReferentialDataLoaderMemory(dataset_dir=dataset_dir,
                                               agent_names=agent_names,
                                               population_probs=population_probs,
                                               batch_size=training_params["batch_size"],
                                               batches_per_epoch=training_params[f"{mode}_batches_per_epoch"],
                                               n_files=n_files,
                                               mode=mode,
                                               seed=training_params["seed"])

    else:
        raise BaseException("Specify a game type")

    return loader
