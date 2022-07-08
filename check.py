from model import Generator
from torchsummary import summary

a = Generator(256, 3, 8, 5, lr_mlp=0.01, n_res=1)
summary(a, (3, 256, 256))