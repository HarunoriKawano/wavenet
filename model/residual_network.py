import torch
from torch import nn


from model.causal_conv import DilatedCausalConv1d

class ResidualBlock(nn.Module):
    def __init__(self, res_channels: int, skip_channels: int, dilation: int):
        super().__init__()

        self.dilated_conv = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.conv_res = nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = nn.Conv1d(res_channels, skip_channels, 1)
        self.conv_ref = nn.Conv1d(res_channels, res_channels * 2, 1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, ref):
        clone = hidden_states.clone()
        hidden_states = self.dilated_conv(hidden_states)
        ref = self.conv_ref(ref)

        # PixelCNN gate
        d = hidden_states.size(1) // 2
        gated_tanh = self.tanh(hidden_states[:, d:] + ref[:, d:])
        gated_sigmoid = self.sigmoid(hidden_states[:, :d] + ref[:, :d])
        gated = gated_tanh * gated_sigmoid

        # Residual network
        hidden_states = self.conv_res(gated)

        # Skip connection
        skip_connection = self.conv_skip(gated)

        return hidden_states + clone, skip_connection


class ResidualLayer(nn.Module):
    def __init__(self, res_channels: int, skip_channels: int):
        super().__init__()
        dilations = [2**n for n in range(10)]
        self.blocks = nn.ModuleList([ResidualBlock(res_channels, skip_channels, dilation) for dilation in dilations])

    def forward(self, hidden_states, ref):
        skip_connections = []
        for block in self.blocks:
            hidden_states, skip_connection = block(hidden_states, ref)
            skip_connections.append(skip_connection)
        skip_connection = torch.stack(skip_connections, dim=-1).sum(dim=-1)

        return hidden_states, skip_connection

class ResidualNetwork(nn.Module):
    def __init__(self, res_channels: int, skip_channels: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([ResidualLayer(res_channels, skip_channels) for _ in range(num_layers)])

    def forward(self, hidden_states, ref):
        skip_connections = []
        for layer in self.layers:
            hidden_states, skip_connection = layer(hidden_states, ref)
            skip_connections.append(skip_connection)
        skip_connection = torch.stack(skip_connections, dim=-1).sum(-1)

        return skip_connection
