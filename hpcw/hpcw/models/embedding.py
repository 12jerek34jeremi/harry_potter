# author: Jedrzej Chmiel
import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Dict


class Embedding(nn.Module):
    """
This class can be used to transform a token (token is a real number, like 1513) to a dense vector (see to_dense method)
and to transform a dense vector back to the token(see get_word_propabilities method). This class uses torch.nn.Embedding
to transform token to dense vector and fully connected layer to transform a dense vector back to the token. This fully
connected layer is called underneath the 'encoding'.
    """

    def __init__(self,
                 corpus_size: int,
                 embedding_size: int,
                 dropout_factor: float,
                 sizes=[512, 1024, 2048]):
        """
Creates an Embedding class object.
Parameters:
    corpus_size:
        The size of corpus, how many words there are in dictionary.
    embedding_size:
        The lenght of a dense vector which will represent a word
        dropout_factor: the dropout factor used in each hidden layer of encoding network.
    sizes:
        Sizes is list of lengths of consecutive hidden layers in encoding network. Encoding network is fully connected
        network used for tansforming a dense vector back to the token.Each hiden layer has a ReLU activation and dropout
        applied. Dropout factor is specified by 'dropout_factor' argument. Length of first (input) layer of encoding
        network is embedding_size and length of last (output) encoding layer is corpus_size.
        """

        super().__init__()
        self.__embedding = nn.Embedding(corpus_size, embedding_size)

        self.__encoding = nn.ModuleList()
        for input_dim, output_dim in zip([embedding_size] + sizes[:-1], sizes):
            self.__encoding.extend([
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_factor)
            ])
        self.__encoding.append(nn.Linear(sizes[-1], corpus_size))

        self.corpus_size = corpus_size
        self.embedding_size = embedding_size
        self.dropout_factor = dropout_factor
        self.sizes = sizes
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def get_embedding(self) -> torch.nn.Embedding:
        """
    This function just returns a reference to embedding of this object. In first version of Embedding class embedding
    was a private class member (was named __embedding). That first version was used to train few models. PyTorch is using
    variables names to create state dictionary. So it wasn't possible to modify change name for '__embedding' to
    'embedding', it wouldn't be possible to load previous versions of Embedding class. Because of the need to access
    embedding of this class objects such a method was necessary.

        """
        return self.__embedding

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
        result = dense_embedding
        for l in self.__encoding:
            result = l(result)

        return f.softmax(result, dim=-1)

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
        result = dense_embedding
        for l in self.__encoding:
            result = l(result)

        return self.log_softmax(result)

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
            The path of the file from which this model will be loaded. If file exists, it will be truncated. If file does
            not exist, it will be created. Directory in which file is must exist. If filepath is /data/dir1/embedding.pth
            and in 'data' directory there is no 'dir1' directory, then this will lead to an error.
    Return:
        The Embedding class object in case of success. None otherwise.
        """
        try:
            parameters_dict = torch.load(filepath)
        except Exception as e:
            print(
                f"Sorry, an exception occurred while trying to load model from file {filepath}"
            )
            return

        if 'sizes' in parameters_dict:
            sizes = parameters_dict['sizes']
        else:
            sizes = [512, 1024, 2048]

        embedding = Embedding(parameters_dict['corpus_size'],
                              parameters_dict['embedding_size'],
                              parameters_dict['dropout_factor'],
                              sizes=sizes)
        embedding.load_state_dict(parameters_dict['state_dict'])
        return embedding

    def save_embedding(self, filepath: str):
        """
    This function saves in file only embedding of this object using torch.save function.
    Parameters:
        filepath:
            The path of the file in which this embedding of this model will be saved. If file exists, it will be
            truncated. If file does not exist, it will be created. Directory in which file is must exist. If filepath is
            /data/dir1/embedding.pth and in 'data' directory there is no 'dir1' directory, then this will lead to an
            error.
    Return:
        This function does not return anything. (It always returns None)
        """
        torch.save(
            {
                "corpus_size": self.corpus_size,
                "embedding_size": self.embedding_size,
                "embedding_embedding_state_dict":
                    self.__embedding.state_dict()
            }, filepath)

    def load_embedding(self, filepath: str) -> bool:
        """
    This function loads an embedding from file. Current embedding of this object is discarded and replaced with the one
    loaded from the file. File from which embedding is loaded should be created by save_embedding method of this class.
    If any problems occur while trying to read object from a file function prints this information on screen and
    returns False.
    Parameters:
        filepath:
            The path of the file from which this model will be loaded. If file exists, it will be truncated. If file does
            not exist, it will be created. Directory in which file is must exist. If filepath is /data/dir1/embedding.pth
            and in 'data' directory there is no 'dir1' directory, then this will lead to an error. Passed file should be
            created using save_embedding method of this class.
    Return:
        True class object in case of success. False otherwise.
        """
        parameters_dict = torch.load(filepath)

        if parameters_dict[
            'corpus_size'] != self.corpus_size or parameters_dict[
            'embedding_size'] != self.embedding_size:
            print(
                f"Expected corpus_size: {self.corpus_size} and embedding size: {self.embedding_size}. \n"
                f"Got corpus_size: {parameters_dict['corpus_size']},"
                f" embedding_size: {parameters_dict['embedding_size']}")
            return False

        self.__embedding.load_state_dict(
            parameters_dict['embedding_embedding_state_dict'])
        return True

    def save_encoding(self, filepath: str):
        """
    This function saves in file only encoding of this object using torch.save function.
    Parameters:
        filepath:
            The path of the file in which this embedding of this model will be saved. If file exists, it will be
            truncated. If file does not exist, it will be created. Directory in which file is must exist. If filepath is
            /data/dir1/embedding.pth and in 'data' directory there is no 'dir1' directory, then this will lead to an
            error.
    Return:
        This function does not return anything. (It always returns None)
     """
        torch.save(
            {
                "corpus_size": self.corpus_size,
                "embedding_size": self.embedding_size,
                "sizes": self.sizes,
                "encoding_state_dict": self.__encoding.state_dict()
            }, filepath)

    def load_encoding(self, filepath: str) -> bool:
        """
    This function loads an embedding from file. Current embedding of this object is discarded and replaced with the one
    loaded from the file. File from which embedding is loaded should be created by save_embedding method of this class.
    If any problems occur while trying to read object from a file function prints this information on screen and
    returns False.
    Parameters:
        filepath:
            The path of the file from which this model will be loaded. If file exists, it will be truncated. If file does
            not exist, it will be created. Directory in which file is must exist. If filepath is /data/dir1/embedding.pth
            and in 'data' directory there is no 'dir1' directory, then this will lead to an error. Passed file should be
            created using save_embedding method of this class.
    Return:
        True class object in case of success. False otherwise.
        """
        parameters_dict = torch.load(filepath)

        if parameters_dict[
            'corpus_size'] != self.corpus_size or parameters_dict[
            'embedding_size'] != self.embedding_size or parameters_dict[
            'sizes'] != self.sizes:
            print(
                f"Expected corpus_size: {self.corpus_size}, embedding size: {self.embedding_size}"
                f" and encoding sizes: {self.sizes}. \n"
                f"Got corpus_size: {parameters_dict['corpus_size']},"
                f" embedding_size: {parameters_dict['embedding_size']}, encoding sizes: {parameters_dict['sizes']}"
            )
            return False

        self.__encoding.load_state_dict(parameters_dict['encoding_state_dict'])
        return True

# author: Jedrzej Chmiel
