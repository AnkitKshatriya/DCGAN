import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ngpu, num_channels, num_features):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (num_channels) x 64 x 64
            # params: in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(num_channels, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features) x 32 x 32
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*2) x 16 x 16
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*4) x 8 x 8
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*8) x 4 x 4
            nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, data):
        if self.train():
            return self.main(data)

        out = self.main[0](data)
        out = self.main[1](out)

        mx1 = nn.MaxPool1d(out, 2, 2)
        flt1 = mx1.flatten()

        out = self.main[2](out)
        out = self.main[3](out)
        out = self.main[4](out)

        mx2 = nn.MaxPool1d(out, 2, 2)
        flt2 = mx2.flatten()

        out = self.main[5](out)
        out = self.main[6](out)
        out = self.main[7](out)

        mx3 = nn.MaxPool1d(out, 2, 2)
        flt3 = mx3.flatten()

        flt25 = torch.cat(flt1, flt2)
        final = torch.cat(flt25, flt3)

        return final
