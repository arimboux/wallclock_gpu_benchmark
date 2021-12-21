import numpy as np
import os
import pycocotools.mask as maskUtils
import torch
import torchvision

from PIL import Image
from torchvision.datasets import VisionDataset

from utils import _coco_remove_images_without_annotations
from transforms import Resize, to_tensor


class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        annFile,
        transform=None,
        target_transform=None,
        transforms=None,
        return_mask=False,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.return_mask = return_mask

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        # return Image.open(os.path.join(self.root, path)).convert("RGB")
        return torchvision.io.read_image(os.path.join(self.root, path), torchvision.io.image.ImageReadMode.RGB) / 255.

    def _load_target(self, id, image_shape, resize_target=600):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))

        target = dict()
        boxes = []
        masks = []
        labels = []

        for ann in anns:
            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h

            if x1 != x2 and y1 != y2:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])
                if self.return_mask:
                    rles = maskUtils.frPyObjects(ann['segmentation'], image_shape[1], image_shape[2])
                    if not isinstance(rles, list):
                        rles = [rles]

                    rle = maskUtils.merge(rles)
                    m = maskUtils.decode(rle)
                    masks.append(m)

        target['labels'] = np.array(labels)
        target['boxes'] = np.array(boxes)
        if self.return_mask:
            target['masks'] = np.array(masks)

        return target

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id, image.shape)

        image, target = to_tensor(image, target)
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)


# collate_fn needs for batch
def collate_fn(batch):
    list_batch = list(zip(*batch))
    list_batch[0] = torch.stack(list_batch[0])

    return tuple(list_batch)


def get_dataloaders(cfg):

    train = CocoDetection(
        '/data/datasets/coco/images/train2017',
        '/data/datasets/coco/annotations/instances_train2017.json',
        transforms=Resize((600, 600))
    )

    val = CocoDetection(
        '/data/datasets/coco/images/val2017',
        '/data/datasets/coco/annotations/instances_val2017.json',
        transforms=Resize((600, 600))
    )

    train = _coco_remove_images_without_annotations(train)
    val = _coco_remove_images_without_annotations(val)

    if cfg.sample_size is not None:
        train = [train[i] for i in range(cfg.sample_size)]
        val = [val[i] for i in range(int(cfg.sample_size * 0.1))]

    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=collate_fn
    )

    val_dataloader = torch.utils.data.DataLoader(
        val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader
