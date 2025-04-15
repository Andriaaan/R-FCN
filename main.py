import torch

from model.rfcn import RFCN

if __name__ == "__main__":
    model = RFCN()
    images = torch.randn(2, 3, 224, 224)
    proposals = model(images)
    print(proposals[0].shape)  # ~[1000, 4]e)