import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 50
        self.vocab_size = 29
        self.num_layers = 1

        self.W_ih = nn.Parameter(torch.randn(
            (self.num_layers, self.hidden_size, self.vocab_size)
        ))
        self.b_ih = nn.Parameter(torch.zeros(
            (self.num_layers, self.hidden_size)
        ))
        self.W_hh = nn.Parameter(torch.randn(
            (self.num_layers, self.hidden_size, self.hidden_size)
        ))
        self.b_hh = nn.Parameter(torch.zeros(
            (self.num_layers, self.hidden_size)
        ))

        self.W_ho = nn.Parameter(torch.randn(
            (self.vocab_size, self.hidden_size)
        ))
        self.b_ho = nn.Parameter(torch.zeros(
            (self.vocab_size)
        ))

    def forward(self, x):
        self.seq_len = x.shape[0]
        batch_size = x.shape[1]
        output = []
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h_t_minus_1 = h_0.clone()
        h_t = h_0.clone()

        for t in range(self.seq_len):
            for layer in range(self.num_layers):
                h_t[layer] = torch.tanh(
                    x[t] @ self.W_ih[layer].T
                    + self.b_ih[layer]
                    + h_t_minus_1[layer] @ self.W_hh[layer].T
                    + self.b_hh[layer]
                )

            output.append(h_t[-1].clone())
            h_t_minus_1 = h_t.clone()

        output = torch.stack(output)
        output = output.view((self.seq_len, batch_size, self.hidden_size))
        output = output @ self.W_ho.T + self.b_ho

        return F.log_softmax(output, dim=-1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.xh = nn.Linear(input_size, hidden_size * 4)
        self.hh = nn.Linear(hidden_size, hidden_size * 4)

        self.out_fc = nn.Linear(hidden_size, self.input_size)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inp):
        # inp shape: seq length, batch size, vocab size
        output = []

        seq_length = inp.shape[0]
        batch_size = inp.shape[1]

        c_t_minus_1 = torch.zeros(batch_size, self.hidden_size)
        h_t_minus_1 = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_length):
            x_t = inp[t]

            gates = self.xh(x_t) + self.hh(h_t_minus_1)
            input_gate, forget_gate, gate_gate, output_gate = gates.chunk(4, 1)

            i_t = torch.sigmoid(input_gate)
            f_t = torch.sigmoid(forget_gate)
            o_t = torch.sigmoid(output_gate)
            g_t = torch.tanh(gate_gate)

            c_t = (f_t * c_t_minus_1) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            c_t_minus_1 = c_t
            h_t_minus_1 = h_t

            out = self.out_fc(h_t)
            output.append(out)

        output = torch.stack(output)

        return F.log_softmax(output, dim=-1)
