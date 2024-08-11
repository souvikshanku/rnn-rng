from dataloader import DigitsData
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import LSTM
from utils import generate_data


def train(dataloader, num_epochs=3):
    lstm = LSTM(input_size=29, hidden_size=50)
    optimizer = optim.Adam([*lstm.parameters()], lr=0.003)

    ckpt = 0
    for epoch in range(num_epochs):
        print(f" ---------------- Epoch: {ckpt + epoch + 1} ----------------")

        for digits, _ in dataloader:
            # dim: seq length, batch size, vocab size
            x = digits.transpose(0, 1).float()

            out = lstm.forward(x[:-1])
            loss = - torch.sum(out * x[1:]) / out.shape[1]
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save({
            "iter": ckpt + epoch + 1,
            "lstm_state_dict": lstm.state_dict(),
        }, f"checkpoints/itr_{ckpt + epoch + 1}.pt")
        print("Checkpint saved...")

    return lstm


if __name__ == "__main__":
    path_to_csv = "digits.csv"
    print("Generating data to train model on...")
    generate_data(path_to_csv)
    print(f"{path_to_csv} created...")

    data = DigitsData(path_to_csv)
    dataloader = DataLoader(data, batch_size=128, shuffle=True)

    print("Training model...")
    train(dataloader, num_epochs=20)
    print("Training model completed.")
