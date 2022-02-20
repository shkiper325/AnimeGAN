import torch
import torch.nn as nn

from parameters import *

ngf = 64
nc = 3
ndf = 64

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(PRIOR_DIM, ngf * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # (nc) x 64 x 64
        )
        
        if USE_CUDA:
            self.cuda()
            
    def forward(self, x):
        x = torch.reshape(x, (-1, PRIOR_DIM, 1, 1))
        
        x = self.main(x)
        
        return x

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        
        self.main = nn.Sequential(
            # (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        if USE_CUDA:
            self.cuda()
        
    def forward(self, x):
        x = self.main(x)
        
        return x.view((-1, 1))
    
    def clip(self):
        self.main[0].weight.data.clamp_(min=-CLIP, max=CLIP)
        self.main[2].weight.data.clamp_(min=-CLIP, max=CLIP)
        self.main[4].weight.data.clamp_(min=-CLIP, max=CLIP)
        self.main[6].weight.data.clamp_(min=-CLIP, max=CLIP)
        self.main[8].weight.data.clamp_(min=-CLIP, max=CLIP)