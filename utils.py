import torch
import numpy as np

from model import LSTM

vocab = "abcdefghijklmnopqrstuvwxyz "
vocab = dict(zip(list(vocab), range(len(vocab))))
vocab["<start>"] = 27
vocab["<end>"] = 28
VOCAB_OPP = {vocab[key]: key for key in vocab}


def _digit_in_words(digit):
    singles = [
        "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten"
    ]
    ones = [
        "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"
    ]
    tens = [
        "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety"
    ]

    assert digit >= 0 and digit <= 99

    if digit <= 10:
        return singles[digit]
    elif digit < 20:
        return ones[digit - 11]
    else:
        d = digit // 10
        r = digit % 10
        if not r:
            return tens[d - 2]
        else:
            return tens[d - 2] + " " + singles[r]


def generate_data(path_to_csv):
    corpus = ""

    for i in range(100):
        corpus += "\n".join([_digit_in_words(i)] * 100) + "\n"

    with open(path_to_csv, "w") as f:
        f.write(corpus)


def generate_random_number():
    lstm = LSTM(29, 50)

    try:
        ckpt = 20
        checkpoint = torch.load(f"checkpoints/itr_{ckpt}.pt")
        lstm.load_state_dict(checkpoint['lstm_state_dict'])
    except FileNotFoundError:
        print(f"No trained model found for checkpoint {ckpt}. Train your model first.")  # noqa: E501

    chars = ""
    inp = torch.zeros((1, 1, 29))
    inp[0, 0, 27] = 1  # <start>
    out = None

    while out != 28:
        with torch.no_grad():
            logits = lstm.forward(inp)

        probs = torch.exp(logits).detach().numpy().reshape(-1, 29)
        # use random samepling to generate random numbers 😅♻️
        out = np.random.choice(len(probs[-1]), p=probs[-1])

        chars += VOCAB_OPP[out]

        x = torch.zeros((1, 1, 29))
        x[0, 0, out] = 1
        inp = torch.cat((inp, x), dim=0)

    return chars[:-5]
