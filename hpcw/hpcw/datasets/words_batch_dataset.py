import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from typing import Tuple


class WordsBatchDataset(Dataset):
    """DataSet for training word batch. One dataset is one book.
    To train over all 7 books you need to create 7 datasets.  Uses word_tokenize from nltk.tokenize to split file
    into words. This function creates tensor of sequential tokens at sequential index. So if the file starts with "you are
     the wizzard, harry.", and following words have following ids: {'you':124, 'are':412, 'the':26, 'wizzard':25,
      ',':432, 'harry':622, '.':11324}, then this tensor will be like [124, 412, 26, 25, 432, 622, 11324, ...]
     """

    def __init__(self,
                 book_filapath: str,
                 dictionary: dict,
                 sequence_length: int,
                 transform: callable = None,
                 target_transform: callable = None):
        """
        Creates dataset from one file.
    Parameters:
        book_filapath:
            file path to file from which to read, should be .txt file with UTF-8 encoding.
        dictionary:
            dictionary of id's of each word. Like {'cat':0, 'wizzard':1, ''hermione': 2, ...}
        sequence_length:
            how many words before and after are used to predict middle word.
        transform:
            function to be applied on each input in __getitem__ method.
        target_transform:
            function to be applied on each target in __getitem__ method.
        """
        super().__init__()
        self.__transform = transform
        self.__target_transform = target_transform
        self.__sequence_length = sequence_length

        try:
            with open(book_filapath, 'rt', encoding='UTF-8') as file:
                words = word_tokenize(file.read())
        except Exception as e:
            print("There was an error while trying to read words from file: ",
                  book_filapath)
            print(e)
            return
        self.tokens = torch.tensor([dictionary[word] for word in words],
                                   dtype=torch.long)
        self.__length = len(self.tokens) - (2 * sequence_length)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
    Parameters:
        index:
            The index of a word. Which is gonna to be predicted.
        Return:
            A tupple. First element of a tuple is tensor of shape (2*s_l,), where s_l is sequence_length. First part of
            tensor are words before word which is to be predicted and second part of tensor are words after word which is
            to be predicted. The second element of tuple is token of predicted word.

    Suppose sequence_length is 3, and the file passed to constructor, of object of this classed called 'obj', starts with:
    "hogward is the best school for wizzards.". Suppose those are ids of words used in this sentance:
    {'hogward': 123, 'is': 34, 'the': 13645, 'best': 7452, 'school': 15123, 'for': 541, 'wizzards': 231}.

    Calling obj[0] should return (tensor([123, 34, 13645, 231, 541, 15123], tensor(7452))
    You can't have 'Hogward' as middle word becouse there are no words before. The actual length of the dataset is
    nr of words in file (len of list produced by word_tokenize) - 2 * sequence_length.
        """
        index = index + self.__sequence_length
        X = torch.cat(
            (self.tokens[index - self.__sequence_length:index],
             torch.flip(
                 self.tokens[index + 1:index + self.__sequence_length + 1],
                 (0, ))),
            dim=0)
        y = self.tokens[index]
        if self.__transform is not None:
            X = self.__transform(X)
        if self.__target_transform is not None:
            y = self.__target_transform(y)
        return X, y

    def __len__(self) -> int:
        """Returns length of this dataset.The actual length of the dataset is
    nr of words in file (len of list produced by word_tokenize) - 2 * sequence_length.
        """
        return self.__length
