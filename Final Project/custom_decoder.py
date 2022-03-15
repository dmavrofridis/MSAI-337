# Write the custom code here
from math import log
from numpy import array
from numpy import argmax

import numpy as np


def greedy_search_decoder(predictions):
    # select token with the maximum probability for each prediction
    output_sequence = [np.argmax(prediction) for prediction in predictions]

    # storing token probabilities
    token_probabilities = [np.max(prediction) for prediction in predictions]

    # multiply individaul token-level probabilities to get overall sequence probability
    sequence_probability = np.product(token_probabilities)

    return output_sequence, sequence_probability


# greedy decoder
def greedy_decoder(data):
    # index for largest probability each row
    return [argmax(s) for s in data]


# beam search
def beam_search_decoder(data, num_beams=3):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:num_beams]
    return sequences
