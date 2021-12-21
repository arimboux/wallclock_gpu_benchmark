import math
import torch
import torchvision
import tqdm
import hydra
import time

from datetime import timedelta

from dataloader import get_dataloaders


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    t1 = time.time()

    if cfg.maskrcnn:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    train_loader, val_loader = get_dataloaders(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.)

    channels = torch.channels_last if cfg.channel_last else torch.preserve_format
    model = model.to(device=torch.device('cuda'), memory_format=channels)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    model.train()
    for x, target in tqdm.tqdm(train_loader):
        x = x.to(device=torch.device('cuda'), memory_format=channels)
        target = [{k: v.to(device=torch.device('cuda')) for k, v in t.items()}
                  for t in target]

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            #  Forward input and target
            loss_dict = model(x, target)

        # Reduce the loss (objectness, rpn_box_reg, classifier, box_reg)
        batch_loss = sum(loss for loss in loss_dict.values()
                         if not torch.isnan(loss) and loss != float("Inf"))

        # Check for non-defined or exploding losses
        if (
            (not math.isnan(batch_loss.item()))
            and (not math.isinf(batch_loss.item()))
        ):
            # Backprop
            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    for x, target in tqdm.tqdm(val_loader):

        target = [{k: v.to(device=torch.device('cuda')) for k, v in t.items()}
                  for t in target]

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            #  Forward input and target
            loss_dict = model(x, target)

    total_time = time.time() - t1

    print(str(timedelta(seconds=total_time)))


if __name__ == '__main__':
    main()
