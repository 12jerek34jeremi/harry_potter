#   Author: Jedrzej Chmiel

import matplotlib.pyplot as plt
import matplotlib
import os
from os import listdir
from nltk.tokenize import word_tokenize
import re
import pickle
import pandas as pd
from hpcw.corpus import Corpus
from hpcw.models.embedding import Embedding
import torch
from typing import Dict


def create_dictionary(directory: str,
                      save_file: str = None,
                      min_documents: int = 1) -> Dict[str, int]:
    """
This method creates a dictionary of all words from all files in passed dictionary. (Almost) each word is given an unique
id (natural number) (see min_documnets argument description find out when some words doesn't get unique id). The lowest
given id is 0 and the highest given id is number of unique words in all files - 1. Each id from interval
<0; number of unique words - 1> is assigned to some word. (There are no gpas in id's).
To divide text to words this function uses word_tokenize() function form nltk package.

Parameters:
    directory (str) :
        Path to the directory with files to create dictionary from words. In the passed directory there must be 7 files
        named 'harry_potter_n_prepared.txt', where n belogns to {1,2,3,4,5,6,7}. If at least one file is missing
        function print information on the screen and returns None. Function ignores all other files in directory.
        Note: This function assumes that files are prepared. (Things like page signs, chapters names and some strange
        characters were removed from the text.)

    save_file (str) :
        The path (directory/file_name) to the file in which created dictionary will be saved using pickle module.
        If the file doesn't exist (but directory exists) the file will be created.
        If save_file is None (default value) function doesn't save dictionary in file.
        Default value is None.

    min_documents (str) :
        The minimum number of files in which each word must appear to get unique id. For example if word "fence" appears
        in file "harry_potter_1_prepared" and min_documents is set to 3, then word 'fence' must appear in at least two
        other files to get unique id. If that's not the case (like word "fence" is only in one file) the word is given
        id which is one of the numbers: 0,1,2:
            0 if it is proper word (like Dumbledore or hogwart);
            1 if it is a number (integer or floating-point)
            2 in every other case.
        If min_documents is greater than 1, then unique ids starts from 3.
        If min_documents is equal 1, then unique ids starts from 0.
        Default value is 1.

return: the created dictionary in case of success, None in case of errors.
        Examplatory return:{'cat': 0, 'wizard': 1, 'magic': 2, ..., 'orange': 21370}
    """

    pattern = re.compile(r'^-?[0-9]*[,.]?[0-9]*$')
    sets = []
    rejected = 0
    dictionary = {}
    rejected_words = []
    for i in range(1, 8):
        file_path = directory + '/harry_potter_' + str(i) + '_prepared.txt'
        if not os.path.exists(file_path):
            print("Could not find file ", file_path)
            return None
        with open(file_path, 'rt', encoding='utf-8') as file:
            sets.append(set(word_tokenize(file.read())))

    if len(sets) < min_documents:
        print(
            f"There are only {len(sets)} documents in this directory and you require from each word to be in at lest"
            f" {min_documents} documents!")
        return None
    min_documents -= 1
    # if word needs to be in at least 3 documents, I need to check if it appears in at least 2 otherdocu ments
    i = 0
    if min_documents != 0:
        i = 3
        for i, my_set in enumerate(sets):
            to_remove = set()
            for word in my_set:
                sets_with_the_word = []
                n = 0

                # start of checking if word is in at least min_documents nr of documents
                for other_set in sets[:i] + sets[i + 1:]:
                    if word in other_set:
                        n += 1
                        sets_with_the_word.append(other_set)
                        if n == min_documents:
                            break
                # end of checking if word is in at least min_documents nr of documents

                if n < min_documents:
                    for other_set in sets_with_the_word:
                        other_set.remove(word)
                    to_remove.add(word)
                    if word.istitle():
                        dictionary[word] = 0
                    elif pattern.match(word) is not None:
                        dictionary[word] = 1
                    else:
                        dictionary[word] = 2
                    rejected += 1
                    rejected_words.append(word)
            my_set -= to_remove

        print(f"Rejected {rejected} words.")
        print("Rejected words:")
        print(rejected_words)

    whole_set = set()
    for my_set in sets:
        whole_set.update(my_set)

    del sets

    for word in whole_set:
        dictionary[word] = i
        i += 1

    if save_file is not None:
        with open(save_file, 'wb') as file:
            pickle.dump(dictionary, file)

    return dictionary


def prepare_harry_book(input_directory: str,
                       output_directory: str,
                       format_mode: int,
                       remove_new_lines: bool = True,
                       to_lower=True,
                       remove_hyphens=True):
    """
This function read each file in input_directory, removes from the files things like chapter numbers and page numbers
using regular expressions' and saves the preprocessed text in new file in output_directory. It changes ??? and ??? double
quotation marks to " quotation mark, ??? ` and ?? single quotation mark to ' quotation mark as well as ?? to normal coma(,).
Two or more dots, next to each other or separated only by space, and ??? at the end of the word are changed to three dots
after a space. (maybe.. --> maybe ... , maybe??? --> maybe ..., maybe. . .  -->   maybe ...). Because of the fact that, if
there is quotation mark after a dot, word_tokenize from nltk sometimes consider dots as part of the word and somtimes
not this function changes all dots followed by quotation mark (single or double) to spaces and dot.
So ala." will be change to ala . " and cat. ' will be chage to cat . '
This function also removes all underscores (_), backslashes (\) and flashes (/) from the text.

input_directory:
    The directory of files to be preprocessed. Each file in that directory must have .txt extension. If at least one
    file in passed directory has different extension function stops work (return). If passed directory do not exist
    function print this information and stop work.
output_directory:
    The directory in which preprocessed text will be stored. If directory does not exist it will be created. The
    preprocessed text is saved in the file of the same name plus _prepared ending. So if in input directory, there is
    file named "deathly_hallows.txt", then in output directory file "deathly_hallows_prepared.txt" will appear. (If such
    files exists it will be truncated, otherwise file will be created)
format_mode:
    The format of a book. In different books pages signs are in different places, chapter names in different places, to
    use this function we need to specify the format of the file. Supported formats: 1,2,3.
remove_new_lines:
    If true, all new line signs will be removed form the file.
    Default: True
to_lower:
    If true, all Capital letters in file will be change to corresponding lower case letters.
    Default: True
remove_hyphens:
    If true each of the signs: "???", "???", and "-" will be removed from the file.
    """

    patterns = []
    patterns.append((re.compile(r'["\???\???]'), '"'))
    patterns.append((re.compile(r"[\'\???\`\??]"), "'"))
    patterns.append((re.compile(r"[\,\??]"), ','))
    patterns.append((re.compile(r'((\. ?){2,})|???'), ' ... '))

    if format_mode == 1:
        patterns.append((re.compile(
            r'\nPage [0-9]+ of [0-9]+\nGet free e-books and video tutorials at www\.passuneb\.com\n'
        ), ' '))
        patterns.append((re.compile(r'\n?CHAPTER .*\n*\n'), '\n'))
    elif format_mode == 2:
        patterns.append((re.compile(
            r'\nC H A P T E R .+\naTHEaPAGEaSIGNa [0-9]+ aTHEaPAGEaSIGNa\n[A-Z ]+\n'
        ), ' '))
        patterns.append((re.compile(
            r'\n[A-Z \.\'\"\,\-]*\naTHEaPAGEaSIGNa [0-9]+ aTHEaPAGEaSIGNa\n'),
                         ' '))
    elif format_mode == 3:
        patterns.append((re.compile(
            r"\naNUMBERaSIXaSIGNa [0-9]+ aNUMBERaSIXaSIGNa\nC H A P T E R [A-Z -]+\n[A-Z ,.???'-]+\n"
        ), ' '))
        patterns.append((re.compile(
            r"\n[A-Z \'\-\n\.???]+\naNUMBERaSIXaSIGNa[\n ][0-9]+( aNUMBERaSIXaSIGNa)?\n"
        ), ' '))
        patterns.append(
            (re.compile(r"\nCHAPTER [A-Z -]+\naNUMBERaSIXaSIGNa [0-9]+\n"),
             ' '))
        patterns.append(
            (re.compile(r"\nCHAPTER [A-Z -]+\n[0-9]+ aNUMBERaSIXaSIGNa\n"),
             ' '))
        patterns.append((re.compile(
            r"Get free e-books and video tutorials at www\.passuneb\.com"),
                         ' '))
    patterns.append((re.compile(r'-\n'), ''))

    if remove_new_lines:
        patterns.append((re.compile(r'\n'), ' '))
        patterns.append((re.compile(r'  '), ' '))

    patterns.append((re.compile(r'\. ?"'), ' . "'))
    patterns.append((re.compile(r"\. ?'"), " . '"))

    if remove_hyphens:
        patterns.append((re.compile(r'[??????\-/\\\_]'), ' '))
    else:
        patterns.append((re.compile(r'[/\\\_]'), ' '))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for input_file in listdir(input_directory):
        if input_file[-4:] != '.txt':
            print(
                "The directory should contain only the txt files! Can't read file: ",
                input_file)
            return

        with open(input_directory + '/' + input_file, 'rt',
                  encoding='utf-8') as file:
            text = file.read()
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        if to_lower:
            text = text.lower()

        with open(output_directory + '/' + input_file[:-4] + '_prepared.txt',
                  'wt',
                  encoding='utf-8') as file:
            file.write(text)


def words_frequencies(input_directory: str):
    """
Read all properly named files in given directory, count words and returns pandas Series representing frequencies of
words.

Parameters:
input_directory:
    Path to the directory with files to read and count words. In the passed directory there must be 7 files named
    'harry_potter_n_prepared.txt', where n belongs to {1,2,3,4,5,6,7}. If at least one file is missing function prints
    information on the screen and returns None. Function ignores all other files in directory.
    Note: This function assumes that files are prepared.
    (Things like page signs, chapters names and some strange characters were removed from the text.)
return:
    Tuple of three pandas Series.
    First Series:
        Count of each word in all files. Indices are words and values are positive integers. Exemplatory output:
        Index,     words_count
        fence      2
        cat        6
        wizard     25
        It means that in word 'fence' appears twice in all files. (In union of files, not in each file
        separately.)
    Second series:
        Count of unique words witch appears given number of times. Exemplary output:
        Index,         words_count
        frequency
        1                 8
        2                 16
        3                 4
        4                 15
        It means that there are 8 unique words, witch appears only once in all files (In union of files, not in each
        file separately.), there are 16 unique words, witch appears exactly two times in all files, and so on...
        Sum of words_count column is number of unique words in all files.
    Third series:
        How many words (not unique) appears given number of times in all files. Exemplary output:
        Index,         how_many
        frequency
        1                 8
        2                 32
        3                 12
        4                 60
        It means that in union of all files there are 12 words (not unique), such that each of them appears three times
        in all files. (In union of files, not in each file separately.) For example each of words 'hollow','narrow',
        blue' and 'yellow' appears exactly three times in all files.
    """
    words = []
    for i in range(1, 8):
        with open(input_directory + '/harry_potter_' + str(i) +
                  '_prepared.txt',
                  'rt',
                  encoding='utf-8') as file:
            words += word_tokenize(file.read())
    words1 = pd.Series(words, dtype='string',
                       name='frequency').value_counts(ascending=True)

    words2 = words1.value_counts(sort=False).sort_index()
    words2.rename('words_count', inplace=True)
    words2.index.rename('frequency', inplace=True)

    words3 = pd.DataFrame({
        'words_count': words2,
        'temporary': words2.index
    },
                          index=words2.index)
    words3['how_many'] = words3['words_count'] * words3['temporary']
    words3 = words3['how_many']

    return words1, words2, words3


def count_distance(word1: str, word2: str, corpus: Corpus or dict,
                   embedding: Embedding) -> float or None:
    """
This function counts distance between two words in particular embedding. If any of two words is not in corpus, then this
information is printed and function returns None. Distance between two words is square of difference of two dense
vectors which represents those words.
Parameters:
    word1:
        First word. Example: 'cat'.
    word2:
        Second word. Example: 'wizard'.
    corpus:
        Any object that supports converting word (str) to token (int) using __getitem__ method. (It should be possible
        to transform word 'cat' to a token using corpus['cat'].) Function uses this object to transform word to a token.
    embedding:
        Embedding object that will be used to transform token into a dense vector.
Return:
     Square of difference of two dense vectors representing given words.

    """
    if word1 not in corpus:
        print(f"Sorry, can't find {word1} in corpus.")
        return None
    if word2 not in corpus:
        print(f"Sorry, can't find {word2} in corpus.")
        return None

    device = next(embedding.parameters()).device
    with torch.no_grad():
        vectors = embedding.to_dense(
            torch.tensor([corpus[word1], corpus[word2]],
                         dtype=torch.long).to(device))
        vec = vectors[0] - vectors[1]
        distance = (vec @ vec).item()
    return distance


def plot_mse(results: Dict[str, int or float]):
    """
This function read consecutive mses from passed dictionary, plots it using matplotlib.pyplot and then show graph on the
screen.

Arguments:
        results: dictionary with consecutive mses. Consecutive mses should be named mse_initial, mse_after_epoch_1,
        mse_after_epoch_2, mse_after_epoch_3, and so on... All other items from dictionary are ignored. Exemplary
        dictionary: {'dense_layer_size': 256,  'embedding_sizes': [128, 512],  'mse_initial': 64.29942370265994,
         'mse_after_epoch_0': 52.60520227320573,  'mse_after_epoch_1: 52.30913876929527',
          'mse_after_epoch_2': 52.20851488967295,  'mse_after_epoch_3': 52.14135190074598}
        """
    values = [results['mse_initial']]
    i = 0
    while 'mse_after_epoch_' + str(i) in results:
        values.append(results['mse_after_epoch_' + str(i)])
        i += 1
    x = list(range(-1, i, 1))
    plt.figure(figsize=(30, 30))
    matplotlib.rcParams.update({'font.size': 35})
    plt.grid(visible=True, which='both')
    plt.plot(x, values)
    plt.show()


# Author: Jedrzej Chmiel
