import torch

from model import WaveNet


class WaveNetFramework:
    def __init__(self, wavenet: WaveNet, out_size: int):
        self.wavenet = wavenet
        self.out_size = out_size

    def training(self, label, ref):
        label = torch.cat([torch.zeros_like(label[:, :, :1]), label[:, :, :-1]], dim=2)
        predicted = self.wavenet(label, ref)

        return predicted

    def inference(self, ref):
        predicted = torch.zeros(ref.size(0), 1, 1, device=ref.device)
        for i in range(ref.size(2)):
            out = self.wavenet(predicted, ref[:, :, :i+1])[:, :, -1]
            out = self.i_mu_law_companding(out.argmax(dim=1, keepdims=True))
            predicted[:, :, -1] = out
            predicted = torch.cat([predicted, torch.zeros_like(predicted[:, :, :1])], dim=-1)

        return predicted[:, :, :-1]

    def i_mu_law_companding(self, predicted):
        y = 2 * (predicted.float() / self.out_size) - 1

        x = torch.sign(y) * (1/self.out_size) * ((1 + self.out_size)**torch.abs(y) - 1)

        return x






