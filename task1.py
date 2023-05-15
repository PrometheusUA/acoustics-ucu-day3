import numpy as np
import torch


def levenstein(str_a, str_b) -> int:
    levenstein_matrix = np.zeros((len(str_a) + 1, len(str_b) + 1))
    levenstein_matrix[0, :] = np.arange(len(str_b) + 1)
    levenstein_matrix[:, 0] = np.arange(len(str_a) + 1)

    for i in range(len(str_a)):
        for j in range(len(str_b)):
            subst_cost = 1 if str_a[i] != str_b[j] else 0

            levenstein_matrix[i + 1, j + 1] = min(
                levenstein_matrix[i, j + 1] + 1, 
                levenstein_matrix[i + 1, j] + 1, 
                levenstein_matrix[i, j] + subst_cost)

    return levenstein_matrix[-1, -1]

def cer(str_a, str_b) -> float:
    levenstein_distance = levenstein(str_a, str_b)

    len_ref = len(str_b)
    return levenstein_distance/len_ref

def wer(str_a, str_b) -> float:
    str_a_words = str_a.split(" ")
    str_b_words = str_b.split(" ")
    levenstein_distance = levenstein(str_a_words, str_b_words)

    len_ref = len(str_b_words)
    return levenstein_distance/len_ref

def levenstein_torch(str_a, str_b) -> int:
    levenstein_matrix = torch.zeros((len(str_a) + 1, len(str_b) + 1), 
                                    requires_grad=True)

    levenstein_matrix[0, :] = torch.arange(len(str_b) + 1)
    levenstein_matrix[:, 0] = torch.arange(len(str_a) + 1)

    for i in range(len(str_a)):
        for j in range(len(str_b)):
            subst_cost = 1 if str_a[i] != str_b[j] else 0

            levenstein_matrix[i + 1, j + 1] = torch.min(torch.tensor([
                levenstein_matrix[i, j + 1] + 1, 
                levenstein_matrix[i + 1, j] + 1, 
                levenstein_matrix[i, j] + subst_cost]))

    return levenstein_matrix[-1, -1]

def wer_loss(pred: torch.Tensor, labels: torch.Tensor):
    batch_size, sentence_size = pred.size()
    losses = torch.FloatTensor((batch_size), requires_grad=True)
    for i in range(batch_size):
        distance = levenstein_torch(pred[i], labels[i])
        losses[i] = distance / sentence_size
    return torch.mean(losses)


if __name__ == '__main__':
    print(levenstein('sence', 'nonsence'))
    print(cer('sence', 'nonsence'))
    print(wer('Bread is bad', "Not so bad"))