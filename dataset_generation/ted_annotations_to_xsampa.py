'''
Convert ground truth words to their XSAMPA phoneme equivalents
'''


from pprint import pprint
import ssl
import os
from os import path
from pywiktionary import Wiktionary #wikt2pron

#resolve SSL Certificate Verification Failure
ssl._create_default_https_context = ssl._create_unverified_context

wikt = Wiktionary(XSAMPA=True) #wikt2pron, not pywikitionary
# word = wikt.lookup("apple")
# pprint(word)

dataset_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/lrs3_annotations/test/'
phoneme_annotations_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/lrs3_annotations/test_phonemes/'

def generate_phoneme_files():
    for directory in os.listdir(dataset_path):
        directory_name = directory
        if not path.exists(phoneme_annotations_path + directory_name):
            os.mkdir(phoneme_annotations_path + directory_name)
        directory = os.path.join(dataset_path, directory)
        if '.DS_Store' not in directory:
            for txt_file in os.listdir(directory):
                txt_file_name = txt_file[:txt_file.find('.txt')]
                txt_file = os.path.join(directory, txt_file)
                print("Processing {}".format(txt_file))
                if '.DS_Store' not in txt_file:
                    with open(txt_file, 'r') as file:
                        line = file.readline() #reads only first line
                        print(line[line.find(':')+2:], end=': ')
                        with open(phoneme_annotations_path + directory_name + '/' + txt_file_name + '.txt', 'a') as new_file:
                            #to avoid duplicate processing - process only if phoneme file size == 0 -> file is blank
                            if os.stat(phoneme_annotations_path + directory_name + '/' + txt_file_name + '.txt').st_size == 0:
                                phoneme_translation = ''
                                for word in line.split()[1:]:
                                    word = word.lower()
                                    wikt_word = wikt.lookup(word)
                                    phonemes = []
                                    if type(wikt_word) == dict and 'English' in wikt_word.keys():
                                        for phoneme_dict in wikt_word['English']:
                                            if type(phoneme_dict) == dict and 'X-SAMPA' in phoneme_dict.keys() and phoneme_dict['X-SAMPA'] != 'en':
                                                phonemes.append(phoneme_dict['X-SAMPA'])
                                        #if '@' figures in any of the phoneme transcriptions, that transcription is probably the accurate one; ignore others
                                        if phonemes:
                                            phoneme = phonemes[0]
                                            for element in phonemes:
                                                if '@' in element:
                                                    phoneme = element
                                                    break
                                            phoneme_translation+=phoneme
                                            print(phoneme)
                                # new_file.write(phoneme_translation)
                            else:
                                print('Phoneme transcription already exists at {}'.format(phoneme_annotations_path + directory_name + '/' + txt_file_name + '.txt'))
                    print('===============')

'''
move phoneme text files to directory containing videos
'''
def move_phoneme_files():
    video_dataset_path = '/Users/padmanabhankrishnamurthy/Desktop/lrs3/test-3/'
    for directory in os.listdir(phoneme_annotations_path):
        directory_name = directory
        directory = os.path.join(phoneme_annotations_path, directory_name)
        if os.path.isdir(directory):
            for txt_file in os.listdir(directory):
                txt_file_name = txt_file
                if '.DS_Store' not in txt_file_name:
                    txt_file = os.path.join(directory, txt_file_name)
                    new_txt_file_name = txt_file_name[:txt_file_name.find('.txt')] + '_phoneme.txt'
                    os.rename(txt_file, video_dataset_path + directory_name + '/' + new_txt_file_name)

generate_phoneme_files()





