from dataset import DigitsData
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LSTM


def train(dataloader, num_epochs=3):
    # rnn = RNN()
    lstm = LSTM(29, 50)
    ckpt = 0

    optimizer = optim.Adam([*lstm.parameters()], lr=0.003)

    for epoch in range(num_epochs):
        print(f" ---------------- Epoch: {ckpt + epoch + 1} ----------------")

        for digits, _ in dataloader:
            x = digits.transpose(0, 1).float()

            # print(x.shape)
            # print("--------------------------")

            out = lstm.forward(x[:-1])
            # print(out.shape)

            loss = - torch.sum(out * x[1:]) / out.shape[1]

            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            torch.save({
                "iter": ckpt + epoch + 1,
                "lstm_state_dict": lstm.state_dict(),
            }, f"checkpoints/itr_{ckpt + epoch + 1}.pt")
            print("Checkpint saved...")

    return lstm


if __name__ == "__main__":
    digits = "digits.csv"
    data = DigitsData(digits)
    dataloader = DataLoader(data, batch_size=256, shuffle=True)
    train(dataloader, num_epochs=40)
