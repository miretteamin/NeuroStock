from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self, input_size=1, num_layers=2,hidden_size=64, output_size=64, num_steps = 15):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.batch_norm1 = nn.BatchNorm1d(num_steps)

    def forward(self, x):
        # h0 = torch.zeros(2, x.size(0), 100).to(device) # num_layers * num_directions, batch_size, hidden_size
        # c0 = torch.zeros(2, x.size(0), 100).to(device)
        x = self.batch_norm1(x)
        out, _ = self.lstm1(x)
        # out, _ = self.lstm2(out)
        out = F.gelu(self.fc1(out[:, -1, :]))
        return out