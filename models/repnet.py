'''
Author: Huang-Junchen huangjc_mail@163.com
Date: 2023-06-13 17:42:02
LastEditors: Huang-Junchen huangjc_mail@163.com
LastEditTime: 2023-06-14 17:37:42
FilePath: /RepNet/models/repnet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch import nn
import torchvision.models as models
from torchinfo import summary

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np

class RepNet(nn.Module):
    def __init__(self, num_frames: int = 64, temperature: float = 13.544) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.temperature = temperature

        self.encoder = Encoder(num_frames)
        self.tsm = TSM(temperature)

    def forward(self, x):
        x = self.encoder(x)
        x = self.tsm(x)
        return x




class Encoder(nn.Module):
    def __init__(self, num_frames: int = 64) -> None:
        super().__init__()

        self.num_frames = num_frames
        # Use ResNet-50 toas base CNN to extract 2D convolutional neural network
        # input size are 112x112x3
        resnet50 = models.resnet50(pretrained=True, progress=True)

        # use the output of conv4_block3 layer to have a larger spatial 2D feature map 7x7x1024
        new_layer3 = nn.Sequential(resnet50.layer3[0],
                                  resnet50.layer3[1],
                                  resnet50.layer3[2])
        
        self.conv = nn.Sequential(*list(resnet50.children())[:-4],
                                  new_layer3
                                  )

        # Temporal context (3D convolution)
        self.temporal_conv = nn.Sequential(
                                    nn.Conv3d(in_channels=1024, out_channels=512,
                                       kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                       padding=(3, 1, 1), dilation=(3, 1, 1)),
                                    nn.BatchNorm3d(512)
                                )
        self.relu = nn.ReLU()

        # Dimensionality reduction (Global 2D Max-pooling)
        self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
                    
    def forward(self, x):
        batch_size, _, c, h, w = x.shape
        # Convolutional feature extractor
        x = x.view(-1, c, h, w)
        x = self.conv(x)

        # Temporal context
        x = x.view(batch_size, self.num_frames, x.shape[1], x.shape[2], x.shape[3])
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = self.relu(x)

        # Dimensionality reduction
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.transpose(0, 1)

        return x
    
    
class TSM(nn.Module):
    '''TSM == Temporal Self-similarity Matrix'''
    def __init__(self, temperature) -> None:
        super().__init__()
        self.temperature = temperature

        self.relu = nn.ReLU()

    def get_sims(self, embs, temperature):
        '''embs = per Frame Embeddings (Encoders output)'''
        """Calculates self-similarity between batch of sequence of embeddings."""
        batch_size = embs.size(0)
        seq_len = embs.size(1)
        embs = embs.view(batch_size, seq_len, -1)

        def _get_sims(embs):
            """Calculates self-similarity between sequence of embeddings."""
            dist = torch.cdist(embs, embs, p=2)
            sims = -torch.pow(dist, 2)
            return sims

        sims = torch.stack([_get_sims(embs[i]) for i in range(batch_size)])
        sims /= temperature
        sims = torch.softmax(sims, dim=-1)
        sims = sims.unsqueeze(-1)
        return sims
    
    def forward(self, x):
        x = self.get_sims(x, self.temperature)
        return x
    
    

        


    
    
if __name__ == "__main__":
    model = RepNet()
    summary(model, input_size=(1, 64, 3, 112, 112))
    x = torch.rand(1, 64, 3, 112, 112).to('cuda')
    output = model(x)
    print(output.shape)
    # print(output)