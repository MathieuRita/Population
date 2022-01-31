from typing import Sequence
import torch as th
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np

class OneHotDistribution(object):

    def __init__(self, n_attributes : int, n_values : int, probs : th.Tensor = None) -> None :

        self.n_attributes = n_attributes
        self.n_values = n_values
        if probs is None :
            self.probs = th.ones((self.n_attributes,self.n_values))/self.n_values
        else:
            raise "Not implemented alternative attributes/values distributions"

    def sample(self, n : int = 1, replacement : bool = True, one_hot_format : bool = False) -> th.Tensor:

        """
        :param n: number of samples
        :return: One hot object composed of n_attributes that can take n_values each

        If one_hot_format = False : output_dim = [n,n_attributes] ; else : output_dim = [n,n_attributes,n_values]
        """

        samples = th.multinomial(self.probs, n, replacement=replacement).t() # [n,n_attributes]

        if one_hot_format:
            samples = F.one_hot(samples,num_classes = self.n_values)
            sampling_probs = (samples*self.probs).sum(dim=2).prod(dim=1)
            return samples, sampling_probs # [n,n_attributes,n_values] [n]
        else:
            sampling_probs = (F.one_hot(samples,num_classes = self.n_values)*self.probs).sum(dim=2).prod(dim=1)
            return samples, sampling_probs   # [n,n_attributes]


class OneHotIterableDataset(th.utils.data.IterableDataset):

    def __init__(self,
                 distribution : OneHotDistribution,
                 n_elements : int = None,
                 one_hot_format : bool = False,
                 replacement : bool = True):

        super(OneHotIterableDataset).__init__()
        self.distribution = distribution
        self.one_hot_format = one_hot_format
        self.n_elements = n_elements

        if self.n_elements is not None:
            self._data, self._probs = self.distribution.sample(n=self.n_elements,
                                                              one_hot_format = self.one_hot_format,
                                                              replacement = replacement)
        else:
            self._data, self._probs = None, None

    def __len__(self):
        return self.n_elements

    def __getitem__(self, index) -> th.Tensor:

        if self._data is None :
            sample, prob = self.distribution.sample(one_hot_format = self.one_hot_format)
        else:
            sample, prob = self._data[index], self._probs[index]

        return sample,prob

    def __iter__(self):
        idx = 0
        while self.n_elements is None or idx < self.n_elements:
            idx += 1
            yield self[idx - 1]


class _ReconstructionIterator():

    def __init__(self,
                 data,
                 data_distribution,
                 n_batches_per_epoch,
                 batch_size,
                 seed=None):

        self.data = data
        self.data_distribution = data_distribution
        self.n_batches_per_epoch = n_batches_per_epoch
        self.batch_size = batch_size
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)


    def __iter__(self):
        return self

    def __next__(self):

        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()

        ids = th.multinomial(self.data_distribution,self.batch_size,replacement=True)

        batch_data = th.stack([self.data[i] for i in ids])

        self.batches_generated += 1

        return batch_data


class ReconstructionDataLoader(th.utils.data.DataLoader):
    
    def __init__(self,
                 data,
                 data_distribution,
                 batches_per_epoch :int,
                 batch_size : int,
                 seed:int=None):

        self.data = data
        self.data_distribution = data_distribution
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.seed = seed

        
    def __iter__(self):

        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed

        return _ReconstructionIterator(data=self.data,
                                       data_distribution = self.data_distribution,
                                       n_batches_per_epoch=self.batches_per_epoch,
                                       batch_size=self.batch_size,
                                       seed=seed)


def build_dataset(game_type : str,
                  object_params : dict,
                  n_elements : int = None,
                  split_proportion : float = 0.8) -> (th.utils.data.IterableDataset, object):

    # TO DO: séparer le cas reconstruction / referential

    if object_params["object_type"]=="one_hot":

        if n_elements is None:
            n_elements = object_params["n_values"] ** object_params["n_attributes"]

        distribution = OneHotDistribution(n_attributes=object_params["n_attributes"],
                                          n_values=object_params["n_values"])

        iterable_dataset = OneHotIterableDataset(distribution = distribution,
                                                 n_elements = n_elements,
                                                 one_hot_format = True)

    else:
        raise "Set a known object type"

    n_train, n_val = int(split_proportion * iterable_dataset.n_elements), \
                     iterable_dataset.n_elements - int(split_proportion * iterable_dataset.n_elements)

    train_dataset, val_dataset = random_split(iterable_dataset, [n_train, n_val])

    return train_dataset, val_dataset

def build_dataloader(game_type : str,
                     dataset : th.utils.data.Dataset,
                     batch_size : int = 1024,
                     batches_per_epoch : int = None,
                     seed : int = 19,
                     ) -> th.utils.data.DataLoader:

    if game_type == "reconstruction":

        if batches_per_epoch is None:
            batches_per_epoch = len(dataset)//batch_size +1

        data = []
        data_distribution = []

        for d , prob in dataset:
            data.append(d)
            data_distribution.append(prob)

        data=th.stack(data)
        data_distribution = th.stack(data_distribution)

        loader = ReconstructionDataLoader(data = data,
                                          data_distribution = data_distribution,
                                          batches_per_epoch= batches_per_epoch,
                                          batch_size = batch_size,
                                          seed = seed)

    return loader