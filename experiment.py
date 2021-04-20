from generator import Generator
import random
import torch
from dcgan import DCGAN_Model

if __name__ == "__main__":
    model = DCGAN_Model('celeb', 64, 5, batch_size=1024)
    # model.plot_training_data()
    model.init_gen_disc()
    model.train()

