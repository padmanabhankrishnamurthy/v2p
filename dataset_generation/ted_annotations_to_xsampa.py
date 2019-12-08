'''
Convert ground truth words to their XSAMPA phoneme equivalents
'''


from pprint import pprint
import ssl
import os
from pywiktionary import Wiktionary

#resolve SSL Certificate Verification Failure
ssl._create_default_https_context = ssl._create_unverified_context

wikt = Wiktionary(XSAMPA=True)
# word = wikt.lookup("apple")
# pprint(word)

dataset_path = '/Users/padmanabhankrishnamurthy/PycharmProjects/helen_v2p/lrs3_annotations/test/'
for directory in os.listdir(dataset_path):
    directory = os.path.join(dataset_path, directory)
    if '.DS_Store' not in directory:
        for txt_file in os.listdir(directory):
            txt_file = os.path.join(directory, txt_file)
            with open(txt_file, 'r') as file:
                line = file.readline() #reads only first line
                print(line[line.find(':')+2:], end=': ')
                with open(txt_file[:txt_file.find('.txt')] + '_phoneme' + '.txt', 'w') as new_file:
                    phoneme_translation = ''
                    for word in line.split()[1:]:
                        word = word.lower()
                        wikt_word = wikt.lookup(word)
                        phonemes = []
                        for dict in wikt_word['English']:
                            if dict['X-SAMPA'] != 'en':
                                phonemes.append(dict['X-SAMPA'])
                        #if '@' figures in any of the phoneme transcriptions, that transcription is probably the accurate one; ignore others
                        phoneme = phonemes[0]
                        for element in phonemes:
                            if '@' in element:
                                phoneme = element
                                break
                        phoneme_translation+=phoneme
                        print(phoneme)
                    new_file.write(phoneme_translation)
            print('===============')






