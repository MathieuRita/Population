from typing import Dict
import numpy as np
import torch as th
import editdistance #pip install editdistance==0.3.1

def compute_language_similarity(messages_1 : th.Tensor,
                                messages_2 : th.Tensor,
                                len_messages_1 : th.Tensor = None,
                                len_messages_2 : th.Tensor = None) -> th.Tensor:


    if not type(messages_1)==np.ndarray:
        messages_1 = messages_1.cpu().numpy()
    if not type(messages_2) == np.ndarray:
        messages_2 = messages_2.cpu().numpy()
    if len_messages_1 is not None:
        if not type(len_messages_1)==np.ndarray:
            len_messages_1=len_messages_1.cpu().numpy()
    else:
        len_messages_1 = [messages_1.size(1)]*messages_1.size(0)
    if len_messages_2 is not None:
        if not type(len_messages_2) == np.ndarray:
            len_messages_2=len_messages_2.cpu().numpy()
    else:
        len_messages_2 = [messages_2.size(1)] * messages_2.size(0)


    distances = []
    for i in range(len(messages_1)):
        m_1 = messages_1[i]
        m_2 = messages_2[i]
        len_m_1 = len_messages_1[i]
        len_m_2 = len_messages_2[i]
        distances.append(editdistance.eval(m_1[:len_m_1],m_2[:len_m_2])/max(len_m_1,len_m_2))

    similarity = 1 - th.Tensor(distances)

    return similarity

def mutual_information_with_samples(X: np.array,
                                    M: np.array):
    # ATTENTION : only implemented for one hot vectors

    counts_m: Dict[str] = {}
    counts_m_x: Dict[str] = {}
    counts_x: Dict[str] = {}

    # Counts

    for i in range(len(X)):
        x_i: str = "".join([str(sym) for sym in X[i]])
        m_i: str = "".join([str(sym) for sym in M[i]])

        if x_i in counts_m_x:
            if m_i in counts_m_x[x_i]:
                counts_m_x[x_i][m_i] += 1
            else:
                counts_m_x[x_i][m_i] = 1
        else:
            counts_m_x[x_i] = {}
            counts_m_x[x_i][m_i] = 1

        if m_i in counts_m:
            counts_m[m_i] += 1
        else:
            counts_m[m_i] = 1

        if x_i in counts_x:
            counts_x[x_i] += 1
        else:
            counts_x[x_i] = 1

    # Normalization
    x_tot = np.sum(list(counts_x.values()))
    for x in counts_x:
        counts_x[x] /= x_tot

    m_tot = np.sum(list(counts_m.values()))
    for m in counts_m:
        counts_m[m] /= m_tot

    for x in counts_m_x:
        m_x_tot = np.sum(list(counts_m_x[x].values()))
        for m in counts_m_x[x]:
            counts_m_x[x][m] /= m_x_tot

    # Compute MI
    MI = 0.

    for x in counts_m_x:
        for m in counts_m:
            if m in counts_m_x[x]:
                pi_m_x = counts_m_x[x][m] / np.sum(list(counts_m_x[x].values()))
                pi_m = counts_m[m] / np.sum(list(counts_m.values()))

                MI += counts_m_x[x][m] / counts_x[x] * np.log(counts_m_x[x][m] / counts_m[m])

    return MI
