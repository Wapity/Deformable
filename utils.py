
import torch

RANDOM = True
NUMBER = 10  # Number of pairs = NUMBER**2


def dataset(random=RANDOM, batch_size):
    images = []
    data_pairs = []
    for i in range(NUMBER):
        if random:
            images.append(torch.normal(1, 0.5, size=(1, 64, 192, 192)))
            print("Image_{}_created".format(i))

    for i in range(NUMBER):
        for j in range(NUMBER):
            data_pairs.append((images[i], images[j]))

    return data_pairs
