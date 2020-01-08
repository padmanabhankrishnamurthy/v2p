'''
Helper Function for Generating Labels given a Sentence
'''
import numpy as np
from pprint import pprint

MAX_PHONEME_SEQUENCE_LENGTH = 116 #actually 116

def get_labels(sentence):
    vocab = open('/Users/padmanabhankrishnamurthy/Desktop/lrs3/arpabet_vocab.txt', 'r').readlines()
    vocab = [element[:element.find('\n')] for element in vocab]
    # print(vocab)

    split = sentence.split('  ')
    for index,element in enumerate(split):
        if element[0] == ' ':
            split[index] = ' sp' + element
    split = ''.join(split).split()
    # print(split)


    # for i in range(len(split)-1):
    #     # print(i, split[i], len(split))
    #     if i == len(split) - 1:
    #         break
    #     if split[i] != '':
    #         if split[i+1] == '':
    #             split[i+2] = ' '
    #             del split[i+1]
    #             # print(split)
    #             # break

    # print(split)
    labels = [vocab.index(element) for element in split]
    print(labels)
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
