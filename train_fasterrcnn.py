import math
import torch
import torchvision
import tqdm

from dataloader import get_dataloaders

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

            boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(boxes)
        labels = torch.tensor([a['category_id'] for a in _annot])
        masks = torch.tensor(masks)

        out.append(dict(boxes=boxes, labels=labels))

    return out

def train_fasterrcnn(cfg):

    print('Training fasterrcnn')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    train_loader, val_loader = get_dataloaders(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.)

    channels = torch.channels_last if cfg.channel_last else torch.preserve_format
    model = model.to(device=torch.device('cuda'), memory_format=channels)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    model.train()
    for x, annot in tqdm.tqdm(train_loader):

        shapes = [img.shape for img in x]

        target = get_targets(annot, shapes)

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
    for x, annot in tqdm.tqdm(val_loader):
        shapes = [img.shape for img in x]

        target = get_targets(annot, shapes)

        x = [torchvision.transforms.functional.resize(i, (600, 600)) for i in x]
        x = torch.stack(x)
        x = x.to(device=torch.device('cuda'), memory_format=channels)

        target = [{k: v.to(device=torch.device('cuda')) for k, v in t.items()}
                  for t in target]

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            #  Forward input and target
            loss_dict = model(x, target)
