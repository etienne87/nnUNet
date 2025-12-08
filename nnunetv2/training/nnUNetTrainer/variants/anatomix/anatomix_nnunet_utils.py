import torch
import torch.nn as nn
from anatomix.model import Unet
from monai.networks.blocks import UnetOutBlock



def decoder_channels(model):
    """Returns list of output channels at each decoder stage"""
    channels = []
    for idx in model.decoder_idx:
        # Get the conv layer right after the upsample
        # The upsample is at idx, conv is at idx+1
        conv_layer = model.model[idx + 1]
        channels.append(conv_layer.out_channels)
    return channels



class AnatomixNNUNetWrapper(nn.Module):
    """Anatomix nnUNet wrapper with deep supervision support"""
    def __init__(self, n_classes):
        super(AnatomixNNUNetWrapper, self).__init__()
        ckpt_path = "/home/eperot/registration/anatomix/model-weights/anatomix.pth"
        self.model = Unet(3, 1, 16, 4, ngf=16)
        self.model.load_state_dict(torch.load(ckpt_path))
        # add option to freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        self.in_channels = decoder_channels(self.model)
        self.fin_layers =  nn.ModuleList([UnetOutBlock(3, in_ch, n_classes + 1, False) for in_ch in self.in_channels])

    def forward(self, x):
        if self.training:
            _, outs = self.model(x, layers=[i+3 for i in self.model.decoder_idx])
            levels = []
            for i, out in enumerate(outs):
                levels += [self.fin_layers[i](out)]
            return levels
        else:
            return self.fin_layers[-1](self.model(x))



if __name__ == "__main__":
    x = torch.randn((1, 1, 128, 128, 128)).cuda()
    model = AnatomixNNUNetWrapper(n_classes=3).cuda()
    model.train()
    with torch.no_grad():
        out = model(x)
        print(type(out))
    model.eval()
    with torch.no_grad():
        out = model(x)
        print(out.shape)
