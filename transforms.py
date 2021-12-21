import numpy as np
import torch
import torchvision.transforms.transforms as transforms


class Resize(transforms.Resize):
    def __init__(self, size, interpolation=2):

        super().__init__(size, interpolation)

    def __call__(self, image, target):

        if isinstance(self.size, int):
            if image.shape[1] < image.shape[2]:
                actual_size = (int((self.size * image.shape[2] / image.shape[1])), self.size)
            else:
                actual_size = (self.size, int((self.size * image.shape[1] / image.shape[2])))
        else:
            actual_size = self.size

        # Resize bboxes
        target['boxes'][:, [-4, -2]] *= actual_size[1] / image.shape[1]
        target['boxes'][:, [-3, -1]] *= actual_size[0] / image.shape[2]

        if 'masks' in target:
            print('not_implemented')

        # Resize image
        image = super(Resize, self).__call__(image)

        return image, target


class ToTensor():
    def __init__(self):
        pass

    def __call__(self, image, target):

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(np.array(image))

        target = {k: torch.tensor(v, dtype=torch.float32) for k, v in target.items()}

        return image, target


def to_tensor(image, target):

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(np.array(image))

    target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
    target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
    if 'masks' in target:
        target['masks'] = torch.tensor(target['masks'], dtype=torch.float32)

    return image, target
