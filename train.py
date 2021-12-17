import hydra
import math
import torch
import torchvision
import tqdm

from dataloader import get_dataloaders


def get_targets(annot):

    out = []
    for _annot in annot:
        boxes = []
        for a in _annot:
            x1, y1, w, h = a['bbox']
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(boxes)
        labels = torch.tensor([a['category_id'] for a in _annot])

        out.append(dict(boxes=boxes, labels=labels))

    return out


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    train_loader, val_loader = get_dataloaders(cfg)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.)
    
    if torch.cuda.is_available():
        model = model.cuda()

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    model.train()
    for x, annot in tqdm.tqdm(train_loader):
        
        target = get_targets(annot)

        if torch.cuda.is_available():
            x = [img.cuda(non_blocking=True) for img in x]
            target = [{k: v.cuda(non_blocking=True) for k, v in t.items()}
                    for t in target]

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            #  Forward input and target
            loss_dict = model(x, target)
        
        # Reduce the loss (objectness, rpn_box_reg, classifier, box_reg)
        batch_loss = sum(loss for loss in loss_dict.values()
                        if not torch.isnan(loss) and loss != float("Inf"))

        # Check for non-defined or exploding losses
        if (not math.isnan(batch_loss.item())) and (not math.isinf(batch_loss.item())):
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
        print(out)


if __name__ == '__main__':
    main()
