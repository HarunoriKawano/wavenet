import torch

from model import WaveNetFramework, WaveNet

if __name__ == '__main__':
    model = WaveNet(1, 32, 128, 256, 5)
    framework = WaveNetFramework(model, 16000)
    labels = torch.rand(4, 1, 16000)  # waveform
    mel_spec = torch.rand(4, 32, 16000)  # feature

    out = framework.training(labels, mel_spec)
    print(out.shape)

    mel_spec = torch.rand(4, 32, 5)  # feature
    out = framework.inference(mel_spec)
    print(out.shape)
