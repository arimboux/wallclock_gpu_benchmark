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

def get_targets(annot, shapes):

    out = []
    for _annot, _shape in zip(annot, shapes):
        boxes = []
        masks = []
        for a in _annot:
            x1, y1, w, h = a['bbox']
            x2, y2 = x1 + w, y1 + h

            ratio_w = 600 / _shape[2]
            ratio_h = 600 / _shape[1]
            x1, x2 = x1 * ratio_w, x2 * ratio_w
            y1, y2 = y1 * ratio_h, y2 * ratio_h

            rles = maskUtils.frPyObjects(a['segmentation'], _shape[1], _shape[2])
            if not isinstance(rles, list):
                rles = [rles]

            rle = maskUtils.merge(rles)
            m = maskUtils.decode(rle)

            m = cv2.resize(m, (600, 600))

            if x1 != x2 and y1 != y2:
                boxes.append([x1, y1, x2, y2])
                masks.append(m)

            # plt.imshow(m)
            # plt.show()

        masks = np.array(masks)
        boxes = torch.tensor(boxes)
        labels = torch.tensor([a['category_id'] for a in _annot])
        masks = torch.tensor(masks)

        out.append(dict(boxes=boxes, labels=labels, masks=masks))

    return out


def train_maskrcnn(cfg):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn()

    train_loader, val_loader = get_dataloaders(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.)

    channels = torch.channels_last if cfg.channel_last else torch.preserve_format
    model = model.to(device=torch.device('cuda'), memory_format=channels)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    model.train()
    for x, annot in tqdm.tqdm(train_loader):

        # img = tensor_to_image(x[0], (0, 0, 0), (1, 1, 1))
        # img = np.ascontiguousarray(img)

        # img = cv2.resize(img, (600, 600))

        shapes = [img.shape for img in x]

        target = get_targets(annot, shapes)

        # ex0 = target[0]
        # for box in ex0['boxes']:
        #     x1, y1, x2, y2 = box
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        #     img = cv2.rectangle(img, (x1, y1), (x2, y2), (1.0, 0, 0), 2)

        # plt.imshow(img)
        # plt.show()

        x = [torchvision.transforms.functional.resize(i, (600, 600)) for i in x]
        x = torch.stack(x)
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
    for x, target in val_loader:
        if torch.cuda.is_available():
            x = [img.cuda(non_blocking=True) for img in x]
            target = [{k: v.cuda(non_blocking=True) for k, v in t.items()}
                      for t in target]

        #  Forward input and target
        out = model(x)
