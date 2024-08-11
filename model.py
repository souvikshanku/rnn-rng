import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        output = []

        # inp shape: seq length, batch size, vocab size
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
