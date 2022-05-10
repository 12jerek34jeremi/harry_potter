#author: Jedrzej Chmiel
import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Dict

class Embedding(nn.Module):
    """
This class can be used to transform a token (like 1513) to a dense vector (see to_dense method) and to transform a dense
vector to an item (see get_word_propabilities method).
    """

    def __init__(self, corpus_size: int, embedding_size: int,
                 dropout_factor: float, sizes = [512, 1024, 2048]):
        """
    Creates an Embedding class object.
    Parameters:
        corpus_size: the size of corpus, how many words there are in dictionary
        embedding_size: the lenght of a dense vector which will represent a word
        dropout_factor: the dropout factor used in each layer of encoding network.
        """

        super().__init__()
        self.__embedding = nn.Embedding(corpus_size, embedding_size)

        self.__encoding = nn.ModuleList()
        for input_dim, output_dim in zip([embedding_size]+sizes[:-1], sizes):
            self.__encoding.extend([nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout(dropout_factor)])
        self.__encoding.append(nn.Linear(sizes[-1], corpus_size))

        self.corpus_size = corpus_size
        self.embedding_size = embedding_size
        self.dropout_factor = dropout_factor
        self.sizes = sizes

    def to_dense(self, tokens: torch.Tensor):
        """
    Transform tokens to dense vecor.
    Parameters:
        tokens:
            The tensor of shape (N,) where N is number of tokens (a batch size).
    Return:
        The tensor of shape (N, e_s), where e_s is embedding size (length of vector representing on word) and N is batch
        size (length of tokens)
        """
        return self.__embedding(tokens)

    def words_probabilities(self, dense_embedding: torch.Tensor):
        """
    Used to transform dense vector (embedding) to tokens. Returns the tensor representing the propability that given
    vector represents given word.
    Parameters:
        dense_embedding:
            A tensor of shape (N, e_s), where here e_s is embedding size (length of vector representing on word) and N is
            batch size (number of words).
        Return:
            A tensor of shape (N, c_s) where c_s is corpus size (number of all available words) and N is batch size
            (number of passed words). If dense_embedding is of shape (1, e_s) and returened tensor lookes like this:
            [[0.01, 0.02, 0.93, 0.1, 0.003, ..., 0.01]], then it means that for 93% passed dense_vector represent word
            of id 2 (because 0.93 is at position [0,2]).
        """
        return f.Softmax(self.__encoding(dense_embedding), dim=-1)

    def forward(self, dense_embedding: torch.Tensor):
        """
    Used to transform dense vector (embedding) to log propabilities of each token. Returns the tensor representing the
    log probability that given vector represents given word.
    Parameters:
        dense_embedding:
            A tensor of shape (N, e_s), where here e_s is embedding size (length of vector representing on word) and N
            is batch size (number of words).
        Return:
            A tensor of shape (N, c_s) where c_s is corpus size (number of all available words) and N is batch size
            (number of passed words). If dense_embedding is of shape (1,5) and returned tensor looks like this:
            [[-2.5, -145.0, -1.0, -2.0, -0.1, -0.2]], then it means that log probability of the fact that passed vector
            represent word of if 1 is -145.0.
        """
        return f.log_softmax(self.__encoding(dense_embedding), dim=-1)

    def save(
        self,
        filepath: str,
    ) -> bool:
        """
    Saves object in file described by filepath. The directory in which file is gonna to be must exist before calling
    this function. If file doesn't exist it will be created, otherwise it will be truncated. If an problem was
    encounter while trying to save this model in a given, file function returns False. Otherwise, function returns True.
    In file there is also saved the embedding used by this object.
    Parameters:
        filepath:
            The path of the file in which this model will be saved.
    Return:
        True in case of success, False in case of failure.
        """
        parameters_dict = self.info()
        parameters_dict["state_dict"] = self.state_dict()
        try:
            torch.save(parameters_dict, filepath)
            return True
        except Exception as e:
            print(
                f"Sorry, an exception occurred while trying to save model to file {filepath}"
            )
            return False

    def info(self) -> Dict[str, int or float]:
        parameters_dict = {
            "corpus_size": self.corpus_size,
            "embedding_size": self.embedding_size,
            "dropout_factor": self.dropout_factor,
            'sizes': self.sizes
        }
        return parameters_dict

    @staticmethod
    def load(filepath: str) -> 'Embedding':
        """
    An static function to. Loads an embedding model from file and returns it. If any problems occur while trying to read
    object from a file, function returns None.
    Parameters:
        filepath:
            The filepath to object in which file is saved. It should have been created by save method of this class.
    Return:
        The Embedding class object in case of success. None otherwise.
        """
        try:
            parameters_dict = torch.load(filepath)
        except Exception as e:
            print(
                f"Sorry, an exception occurred while trying to save model to file {filepath}"
            )
            return None

        if 'sizes' in parameters_dict:
            sizes = parameters_dict['sizes']
        else:
            sizes = [512,1024,2048]

        embedding = Embedding(parameters_dict['corpus_size'],
                              parameters_dict['embedding_size'],
                              parameters_dict['dropout_factor'], sizes=sizes)
        embedding.load_state_dict(parameters_dict['state_dict'])
        return embedding


#author: Jedrzej Chmiel
