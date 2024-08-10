import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DigitsData(Dataset):
    def __init__(self, data):
        self.digits = pd.read_csv(data, header=None)

        vocab = "abcdefghijklmnopqrstuvwxyz "
        self.vocab = dict(zip(list(vocab), range(len(vocab))))
        self.vocab["<start>"] = 27
        self.vocab["<end>"] = 28

    def __len__(self):
        return len(self.digits)

    def __getitem__(self, idx):
        digit = self.digits.iloc[idx, 0]
        digit_ohe = self._get_ohe_label(digit)
        return digit_ohe, digit

    def _get_ohe_label(self, label):
        tokens = [27]  # <start>
        tokens += [self.vocab[char] for char in label]
        extra = 15 - len(tokens)
        tokens += [28] * extra  # <end>

        num_classes = len(self.vocab)
        y = F.one_hot(torch.tensor(tokens), num_classes)

        return y


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    digits = "digits.csv"
    data = DigitsData(digits)
    dataloader = DataLoader(data, batch_size=10, shuffle=True)
    print(next(iter(dataloader))[0].shape)
