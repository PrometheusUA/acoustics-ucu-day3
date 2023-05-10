import numpy as np


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


if __name__ == '__main__':
    print(levenstein('sence', 'nonsence'))
    print(cer('sence', 'nonsence'))
    print(wer('Bread is bad', "Not so bad"))