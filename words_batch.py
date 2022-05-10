# Author: Jedrzej Chmiel

import torch
import torch.nn as nn
from embedding import Embedding
from typing import Dict

class WordsBatch(nn.Module): #second version
    """A nural netwrok of given archicture:
    Assume embedding_size is 128, corpus_size is 21371, hidden_state_size is 256 and dropout_factor is 0.1 and
    sequence_length is 3.

                    Fully connected layer of size 128.                           -
                                    |                                               |
                    Fully concatenation layer of length 1024, ReLU activation       |
                                 and 0.1 dropout.                                   ----- the tail
                                       |                                            |
                                    tail_input (512,)                            -
                                       |
                                       | (concatenation)
                    hidden_state  -----|-----  hidden_state
                        (256,)                 (256,)
                          |                     |
    LSTM------LSTM------LSTM                   LSTM------LSTM------LSTM
      |         |         |                     |         |         |
    LSTM------LSTM------LSTM                   LSTM------LSTM------LSTM
      |         |         |                     |         |         |
    dense     dense     dense                  dense     dense     dense
    vector    vector    vector                 vector    vector    vector
    (128,)    (128,)    (128,)                 (128,)    (128,)    (128,)         <----shape of tensor
      |         |         |                       |         |         |
    token      token      token                token      token      token        <--- each token is one-item tensor
      |         |          |                      |         |          |
    Hogward     is         the                 school      for       wizzards

    (We want to predict the word 'best' in sentance: Hogward is the best school for wizzards)

    """

    def __init__(self, embedding: Embedding, hidden_state_size: int,
                 dropout_factor: float, sequence_length: int, dense_layer_size: int = 1024):
        """
    Parameters:
        embedding:
            The object of Embedding class. It will be used to convert tokens for dense vectors.
        hidden_state_size:
            The size of hidden_state in both layers of LSTM.
        dropout_factor:
            Dropout factor in LSTM layers and in tail in fully connected layers.
        sequence_length:
            How many words before and after will be used to predict the middle word. If sequence_length is 3 then
            input to this model should be three words before and three words after.
        """
        super().__init__()
        self.embedding = embedding
        embedding_size = embedding.embedding_size
        self.lstm_before = nn.LSTM(embedding_size,
                                   hidden_state_size,
                                   2,
                                   dropout=dropout_factor,
                                   batch_first=True)
        self.lstm_after = nn.LSTM(embedding_size,
                                  hidden_state_size,
                                  2,
                                  dropout=dropout_factor,
                                  batch_first=True)
        self.tail = nn.Sequential(nn.Linear(hidden_state_size * 2, dense_layer_size),
                                  nn.ReLU(), nn.Dropout(dropout_factor),
                                  nn.Linear(dense_layer_size, embedding_size))
        self.sequence_length = sequence_length
        self.hidden_state_size = hidden_state_size
        self.dropout_factor = dropout_factor
        self.dense_layer_size = dense_layer_size

    def forward(self, input):
        """
    Returns a dense vector (embedding) representing the predicted middle word.
    Parameters:
        input:
            The tensor of shape (N, 2*s_l), where N is batch size and s_l is sequence_length. First part of second axis
             are words before word which is to be predicted and second part of second axis are words after word which is
             to be predicted.
    Return:
        The tensor of shape (N, e_s), where N is batch size and e_s is embedding_size.

    Suppose sequence_length is 3, batch size is 1, embedding_size is 128 and we want to predict the word "best" in
    sentance: "hogward is the best school for wizzards". Suppose those are ids of words used in this sentance:
    {'hogward': 123, 'is': 34, 'the': 13645, 'best': 7452, 'school': 15123, 'for': 541, 'wizzards': 231}.
    To predict the middle word in that sentance we should pass to this function the following tensor:
    [[123, 34, 13645, 231, 541, 15123]]. And we should get tensor of shape (1, 128) being the dense vector representing
    the word 'best'.
        """
        batch_size = input.shape[0]
        input = self.embedding.to_dense(input)
        _, (hiddens_before,
            _) = self.lstm_before(input[:, :self.sequence_length, :])
        _, (hiddens_after,
            _) = self.lstm_after(input[:, self.sequence_length:, :])
        return self.tail(
            torch.stack([
                torch.cat((hiddens_before[1, i], hiddens_after[1, i]), dim=0)
                for i in range(batch_size)
            ]))

    def save(
        self,
        filepath: str,
    ) -> bool:
        """
    Saves object in file described by filepath. The directory in which file is gonna tobe must exist before calling
    this function. If file doesn't exist it will be created, otherwise it will be truncated. If an problem was
    encounter while trying to save this model in a given, file function returns False. Otherwise, function returns True.
    Parameters:
        filepath:
            The path of the file in which this model will be saved.
    Return:
        True in case of success, False in case of failure.
        """
        parameters_dict = self.info()
        parameters_dict['words_batch_state_dict'] = self.state_dict()
        parameters_dict['embedding_state_dict'] = self.embedding.state_dict()
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
            "sequence_length": self.sequence_length,
            "hidden_state_size": self.hidden_state_size,
            "words_batch_dropout_factor": self.dropout_factor,
            "corpus_size": self.embedding.corpus_size,
            "embedding_size": self.embedding.embedding_size,
            "embedding_dropout_factor": self.embedding.dropout_factor,
            "dense_layer_size": self.dense_layer_size,
            'embedding_sizes': self.embedding.sizes
        }
        return parameters_dict

    @staticmethod
    def load(filepath: str) -> 'WordsBatch':
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

        if 'dense_layer_size' in parameters_dict:
            dense_layer_size = parameters_dict['dense_layer_size']
        else:
            dense_layer_size = 1024
        if 'embedding_sizes' in parameters_dict:
            embedding_sizes = parameters_dict['embedding_sizes']
        else:
            embedding_sizes = [512,1024,2048]

        embedding = Embedding(parameters_dict['corpus_size'],
                              parameters_dict['embedding_size'],
                              parameters_dict['embedding_dropout_factor'],
                              sizes=embedding_sizes)
        embedding.load_state_dict(parameters_dict['embedding_state_dict'])

        words_batch = WordsBatch(embedding,
                                 parameters_dict['hidden_state_size'],
                                 parameters_dict['words_batch_dropout_factor'],
                                 parameters_dict['sequence_length'],
                                 dense_layer_size = dense_layer_size)
        words_batch.load_state_dict(parameters_dict['words_batch_state_dict'])
        return words_batch


# Author: Jedrzej Chmiel
