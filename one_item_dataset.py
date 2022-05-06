#author: Jedrzej Chmiel
import torch
from torch.utils.data import Dataset


class OneItemDataset(Dataset):
    """This class is used to train encoding and embedding without words_batch. One item of this dataset is just a one
    item tensor (shape (,) ). This item is an token representing a word."""

    def __init__(self, dictionary_length, transform=None):
        super().__init__()
        self.__transform = transform
        self.__length = dictionary_length

    def __getitem__(self, index: int) -> torch.Tensor:
        """
    Parameters:
        index:
            An token of w word.
    Return:
        A one item tensor with this token :).
        :return:
        """
        x = torch.tensor(index, dtype=torch.long)
        if self.__transform is not None:
            x = self.__transform(x)
        return x

    def __len__(self):
        return self.__length


#author: Jedrzej Chmiel
