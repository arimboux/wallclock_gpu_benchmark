import torch
import torchvision

from torchvision.datasets import CocoDetection
import torchvision.transforms as T

def _coco_remove_images_without_annotations(dataset, cat_list=None): 
    def _has_only_empty_bbox(anno): 
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno) 

    def _count_visible_keypoints(anno): 
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno) 

    min_keypoints_per_image = 10 

    def _has_valid_annotation(anno): 
        # if it's empty, there is no annotation 
        if len(anno) == 0: 
            return False 
        # if all boxes have close to zero area, there is no annotation 
        if _has_only_empty_bbox(anno): 
            return False 
        # keypoints task have a slight different critera for considering 
        # if an annotation is valid 
        if "keypoints" not in anno[0]: 
            return True 
        # for keypoint detection tasks, only consider valid images those 
        # containing at least min_keypoints_per_image 
        if _count_visible_keypoints(anno) >= min_keypoints_per_image: 
            return True 
        return False 

    assert isinstance(dataset, torchvision.datasets.CocoDetection) 
    ids = [] 
    for ds_idx, img_id in enumerate(dataset.ids): 
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None) 
        anno = dataset.coco.loadAnns(ann_ids) 
        if cat_list: 
            anno = [obj for obj in anno if obj["category_id"] in cat_list] 
        if _has_valid_annotation(anno): 
            ids.append(ds_idx) 

    dataset = torch.utils.data.Subset(dataset, ids) 
    return dataset 

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders():

    train = CocoDetection('/data/datasets/coco/images/train2017', 
                                    '/data/datasets/coco/annotations/instances_train2017.json',
                                    transform=T.ToTensor())

    val = CocoDetection('/data/datasets/coco/images/val2017', 
                                '/data/datasets/coco/annotations/instances_val2017.json',
                                transform=T.ToTensor())

    train = _coco_remove_images_without_annotations(train)
    val = _coco_remove_images_without_annotations(val)

    train_batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(train,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    val_dataloader = torch.utils.data.DataLoader(val,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)


    return train_dataloader, val_dataloader
