'''
Helper Function for Generating Labels given a Sentence
'''

def get_labels(sentence):
    vocab = open('/Users/padmanabhankrishnamurthy/Desktop/lrs3/arpabet_vocab.txt', 'r').readlines()
    vocab = [element[:element.find('\n')] for element in vocab]
    labels = [vocab.index(element) for element in sentence.split()]
    print(labels)
    return labels

sentence = 'W IY1 D   L AH1 V   T UW1   HH EH1 L P'
get_labels(sentence)
