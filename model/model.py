from torch import nn

from model.causal_conv import CausalConv
from model.residual_network import ResidualNetwork


class OutLayer(nn.Module):
    def __init__(self, skip_channels, out_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.out_conv = nn.Conv1d(skip_channels, out_size, kernel_size=1)

    def forward(self, hidden_states):
        hidden_states = self.relu(hidden_states)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.out_conv(hidden_states)

        return hidden_states


class WaveNet(nn.Module):
    def __init__(self, input_channels: int, res_channels: int, skip_channels: int, out_size: int, num_layers: int):
        super().__init__()
        self.causal_conv = CausalConv(input_channels, res_channels)
        self.residual_network = ResidualNetwork(res_channels, skip_channels, num_layers)
        self.out_layer = OutLayer(skip_channels, out_size)

    def forward(self, inputs, ref):
        hidden_states = self.causal_conv(inputs)
        hidden_states = self.residual_network(hidden_states, ref)
        out = self.out_layer(hidden_states)

        return out
