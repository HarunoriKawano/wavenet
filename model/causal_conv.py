from torch import nn


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=2, stride=1, padding=1, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        return hidden_states[:, :, :-1]


class DilatedCausalConv1d(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()

        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels * 2,
                                    kernel_size=2, stride=1,
                                    dilation=dilation,
                                    padding=dilation,
                                    bias=False)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states[:, :, :-self.dilation]

        return hidden_states
