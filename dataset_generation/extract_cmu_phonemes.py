from g2p_en import G2p #https://github.com/Kyubyong/g2p
import ssl
import os
import nltk
import inflect #for spelling out numbers
from colorama import Back, Style
from pprint import pprint

'''
TODO:
G2P is not being used since inversion is difficult (due to G2P generating its own phoneme sequences for homophones and unknown words, and not handling apostrophes well)
Therefore, currently a trivial mapping as given by the CMUdict will be used. 
    For homophones, the first phoneme sequence as given by the CMUdict will be used
    This creates ambiguity and inaccuracies for homophones - NEEDS TO BE WORKED ON IN FUTURE
        g2p = G2p()
        print(g2p("we don't really walk anymore"))
'''

#READ
#https://www.nltk.org/book/ch02.html

#resolve SSL Certificate Verification Failure
ssl._create_default_https_context = ssl._create_unverified_context

dataset_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3'
phoneme_annotations_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/lrs3_annotations/test_phonemes/'
vocab_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/'

#load dictionaries
cmudict = nltk.corpus.cmudict.dict()
inverse_dict = []
for key,value in cmudict.items():
    inverse_dict.append((value, key))
vocab = []

def word_to_phoneme(word):
    phoneme_seq = []

    try:
        phoneme_seq = cmudict[word][0]

        #add phonemes to vocabulary
        for element in phoneme_seq:
            if element not in vocab:
                vocab.append(element)

    except KeyError:
        key, remaining = get_longest_vaid_key(word, '')
        # print(key, remaining)
        seq_part_1 = cmudict[key][0]
        if seq_part_1[-1] == ' ':
            seq_part_1.pop() #pop the unnecessary space before concatenation
        # print(seq_part_1)
        seq_part_2 = word_to_phoneme(remaining)
        # print(seq_part_2)
        phoneme_seq = seq_part_1 + seq_part_2

    # print(phoneme_seq)

    if phoneme_seq[-1] != ' ':
        phoneme_seq.append(' ')

    return phoneme_seq

def get_longest_vaid_key(word, remaining):
    end = len(word)
    if word[:end] in cmudict.keys():
        return(word, remaining)
    else:
        end-=1
        remaining = word[end:] + remaining
        return get_longest_vaid_key(word[:end], remaining)

def phoneme_to_word(key:list):
    for tup in inverse_dict:
        if tup[0] == key:
            print(tup[1])
            return tup[1]

def generate_phoneme_files():

    for speaker in os.listdir(dataset_path):
        speaker_name = speaker
        speaker = os.path.join(dataset_path, speaker_name)

        if not os.path.isdir(speaker): #ignore .DS_Store
            continue

        for file in os.listdir(speaker):
            if '.txt' in file and 'phoneme' not in file:
                file_name = file
                file = os.path.join(speaker, file_name)

                with open(file, 'r') as txtfile:
                    first_line = txtfile.readline()
                    sentence = first_line[first_line.find(':') + 3 : ]
                    sentence = sentence.split()

                    phoneme_sentence = []
                    for word in sentence:
                        word = word.lower()

                        #HANDLING NUMBERS:
                        # numbers aren't in CMUdict. So, spell them out and pass them
                        if word.isdigit():
                            converter = inflect.engine()
                            spelt = converter.number_to_words(int(word)).replace('-', ' ').split()
                            for spelt_word in spelt:
                                phoneme_seq = word_to_phoneme(spelt_word)
                                phoneme_sentence.extend(phoneme_seq)
                            continue

                        try:
                            phoneme_seq = word_to_phoneme(word)
                        except:
                            print(Back.RED + " ".join(sentence), Back.GREEN + word,  Style.RESET_ALL) #easy to debug

                        phoneme_sentence.extend(phoneme_seq)

                    phoneme_sentence.pop() #remove the extra added space at the end
                    phoneme_sentence_string = ' '.join(phoneme_sentence)
                    # print(first_line, phoneme_sentence, '\n', phoneme_sentence_string, '\n')

                    #write files
                    # with open(os.path.join(speaker, file_name[:file_name.find('.txt')] + '_phoneme.txt'), 'w') as phoneme_file:
                    #     phoneme_file.write(phoneme_sentence_string)

def save_vocab(path):
    global vocab

    vocab = sorted(vocab)
    print(len(vocab))
    pprint(vocab)

    # write files
    # with open(os.path.join(path, 'arpabet_vocab.txt'), 'w') as vocab_file:
    #     for element in vocab:
    #         vocab_file.write(element + '\n')

# generate_phoneme_files() #UNCOMMENT TO CALL
# save_vocab(vocab_path)


