'''
Helper Function for Generating Labels given a Sentence
'''
import numpy as np

MAX_PHONEME_SEQUENCE_LENGTH = 116

def get_labels(sentence):
    vocab = open('/Users/padmanabhankrishnamurthy/Desktop/lrs3/arpabet_vocab.txt', 'r').readlines()
    vocab = [element[:element.find('\n')] for element in vocab]

    labels = [vocab.index(element) for element in sentence.split()]
    unpadded_length = len(labels)
    #add padding if necessary
    if len(labels) < MAX_PHONEME_SEQUENCE_LENGTH:
        labels = labels + ([-1.0] * (MAX_PHONEME_SEQUENCE_LENGTH - len(labels)))

    # print(labels)
    return labels, unpadded_length

def sequence_from_labels(labels:list):
    vocab = open('/Users/padmanabhankrishnamurthy/Desktop/lrs3/arpabet_vocab.txt', 'r').readlines()
    vocab = [element[:element.find('\n')] for element in vocab]
    sequence = ''
    for element in labels:
        sequence+= vocab[element] + ' '
    return sequence

sentence = 'W IY1 D   L AH1 V   T UW1   HH EH1 L P'
get_labels(sentence)
