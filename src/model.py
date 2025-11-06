import torch.nn as nn

class ASLMLP(nn.Module):
    def __init__(self, input_size=63, hidden_size=256, output_size=29):
        super(ASLMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)