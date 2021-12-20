import cv2
import hydra
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tqdm

from dataloader import get_dataloaders
import pycocotools.mask as maskUtils
from train_fasterrcnn import train_fasterrcnn
from train_maskrcnn import train_maskrcnn

@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    if cfg.maskrcnn:
        train_maskrcnn(cfg)
    else:
        train_fasterrcnn(cfg)

if __name__ == '__main__':
    main()
