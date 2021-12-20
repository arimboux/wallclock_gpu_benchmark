import hydra
import time

from datetime import timedelta
from train_fasterrcnn import train_fasterrcnn
from train_maskrcnn import train_maskrcnn


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    t1 = time.time()
    if cfg.maskrcnn:
        train_maskrcnn(cfg)
    else:
        train_fasterrcnn(cfg)

    total_time = time.time() - t1
    print(str(timedelta(seconds=total_time)))


if __name__ == '__main__':
    main()
