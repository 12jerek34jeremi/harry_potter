#author: Jedrzej Chmiel
import pickle


class Corpus:
    """
This class can be used to change words to tokens and tokens to words. After creating an object of this class named
'the_corpus' you can check what word has id 15432 using the_corpus[15432]. You can also check what is the id of word
'cat' using the_corpus['cat'].
    """

    def __init__(self, dictionary_filepath: str):
        """
    Creates object of class Corpus.
    Parameters:
        dictionary_filepath:
            The file path to the dictionary[str, int] saved using pickle module. The dictionary should contain id's of
            all necessary words. Exemplatory dictionary:
            {"cat": 0, "wizzard": 1, "grass": 2}
            Dictionary should not contain any id gaps! (If max id is 5 and min id is 0 one, then one word should be
            assigned to each integer number belonging to <0, 5>). The lowest id should be 0. This function checks if
            dictionary saved in passed file is correct. If there is an id gap in dictionary, function prints this
            information on the screen and raises an exception.
    This constructor creates a list of str. In this list under each index is word which id is this index. This list is
    used then to quickly check what word has given id.
        """
        try:
            with open(dictionary_filepath, 'rb') as file:
                self.dictionary = pickle.load(file)
        except Exception as e:
            print(
                "There was an error while trying to read dictionary from file: ",
                dictionary_filepath)
            print(e)
            return

        self.__length = len(self.dictionary)
        words = [None for _ in range(self.__length)]
        for word, word_id in self.dictionary.items():
            words[word_id] = word

        if None in words:
            print("Dictionary saved in file: ", dictionary_filepath,
                  " has a id gap.")
            print("There is no word assigned to id: ", words.index(None))
            raise Exception("Id gap in dictionary.")
            return

        self.__words = words

    def __getitem__(self, index: str or int):
        """
    If type of index is str function returns the id (int) assigned to index.
    If type of index is int function returns the word (str) assigned to index.
    If index is of any other type function raises an TypeError.
    Parameters:
        index:
            word (str) (to get id) or id (int) (to get word)
    Return:
        word (str) or id (int)
        """
        if isinstance(index, int):
            return self.__words[index]
        elif isinstance(index, str):
            return self.dictionary[index]
        else:
            raise TypeError(
                "Unsupported index type: " + str(type(index)) +
                " (in __getitem__ function of Corpus class object)")

    def __len__(self):
        "Returns the number of words in dictionary. (max_id+1)"
        return self.__length


#author: Jedrzej Chmiel
