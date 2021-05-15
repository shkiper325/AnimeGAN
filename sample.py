import os

import numpy as np
import cv2

import torch
import torch.nn as nn

MODEL_PATH = 'checkpoint'
SAMPLE_COUNT = 50
PRIOR_DIM = 100
OUT_DIR = 'samples'

ngf = 64
nc = 3

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
if USE_CUDA:
    print('Using CUDA')
else:
    print('Using CPU')


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(PRIOR_DIM, ngf * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            #nn.BatchNorm2d(ngf * 8),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            #nn.BatchNorm2d(ngf * 4),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            #nn.BatchNorm2d(ngf * 2),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            #nn.BatchNorm2d(ngf),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
        if USE_CUDA:
            self.cuda()
            
    def forward(self, x):
        x = torch.reshape(x, (-1, PRIOR_DIM, 1, 1))
        
        x = self.main(x)
        
        return x

def prior():
    return np.random.multivariate_normal(np.zeros(PRIOR_DIM), np.identity(PRIOR_DIM))


def main():
    gen = G()
    model = torch.load(MODEL_PATH)
    gen.load_state_dict(model['gen'])

    for sample_num in range(SAMPLE_COUNT):
        gen.eval()        
        image = (gen(FloatTensor(np.expand_dims(prior(), 0))).detach().cpu().numpy()[0] + 1) / 2
        image = (np.swapaxes(image, 0, 2) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, str(sample_num) + '.jpg'), image)

if __name__ == '__main__':
    main()