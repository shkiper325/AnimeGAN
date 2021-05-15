import numpy as np
import imageio
import cv2
import n_sphere
import os

import torch
import torch.nn as nn

PRIOR_DIM = 100
MODEL_PATH = 'checkpoint'
GIF_PATH = 'out.gif'
IMG_COUNT = 100
HIDDEN_SIZE = 100
PI = 3.1415

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
    print('Loading model')

    gen = G()
    model = torch.load(MODEL_PATH)
    gen.load_state_dict(model['gen'])

    gen.eval()

    gif_frames = []

    for i in range(IMG_COUNT):
        print('Processing image number ', i)

        phi = None
        if i <= IMG_COUNT / 2:
            phi = PI * i * 2 / IMG_COUNT
        else:
            phi = PI - (PI * i * 2 / IMG_COUNT - PI)
        phi_n_minus_1 = PI * i / IMG_COUNT

        N = np.array([phi for i in range(HIDDEN_SIZE)]) + 1e-10
        N[0] = 10 #radius
        N[-1] *= phi_n_minus_1 #last angle is in [0, 2*PI)

        N = n_sphere.convert_rectangular(N)

        N = np.expand_dims(N, 0)
        N = FloatTensor(N)
        image = (gen(N).detach().cpu().numpy()[0] + 1) / 2
        image = (np.swapaxes(image, 0, 2) * 255).astype(np.uint8)

        print(image.shape)

        #image = np.clip(image + 1, 0, 2-0.001) / 2 * 256.0
        #image = image.astype(np.uint8)
        #image = np.moveaxis(image, 0, 2)

        gif_frames.append(image)

        # cv2.imwrite(os.path.join('gif_source', str(i) + '.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) # For debug

    imageio.mimwrite(GIF_PATH, gif_frames)

    print('Done!')

if __name__ == '__main__':
    main()