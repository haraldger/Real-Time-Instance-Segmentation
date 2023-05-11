import argparse
from math import ceil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from model.yolact import Yolact
from multibox_loss import MultiboxLoss
from torchvision.transforms.functional import to_tensor
import numpy as np
from pycocotools import mask as maskUtils
import torch
from pycocotools import mask as coco_mask
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import os

class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        width, height = image.size

        masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
        masks = np.vstack(masks)
        masks = masks.reshape(-1, height, width)
        # Masks should always have 100 as first dimension, either pad or drop
        if masks.shape[0] < 100:
            masks = np.pad(masks, ((0, 100 - masks.shape[0]), (0, 0), (0, 0)), mode='constant')
        elif masks.shape[0] > 100:
            masks = masks[:100, :, :]
        masks = torch.tensor(masks, dtype=torch.uint8)

        return image, target, masks

    def __len__(self) -> int:
        return len(self.ids)

def transform_targets(targets, batch_size, k, mask_size):
    gt_labels = torch.zeros((batch_size, k), dtype=torch.long) # TODO different dimension than expected in multibox loss?
    gt_locations = torch.zeros((batch_size, k, 4), dtype=torch.float32)
    # gt_masks = torch.zeros((batch_size, k, mask_size, mask_size), dtype=torch.float32)
    num_objects = torch.zeros((batch_size, 1), dtype=torch.long)

    for batch_idx, target in enumerate(targets):
        num_objects[batch_idx, 0] = len(target)
        for idx, annotation in enumerate(target):
            if idx >= k:
                break

            # Labels
            gt_labels[batch_idx, idx] = annotation["category_id"] 

            # Locations (bbox)
            bbox = annotation["bbox"]  # [x, y, width, height]
            gt_locations[batch_idx, idx, :] = torch.tensor([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

            # Masks
            # rle = coco_mask.frPyObjects(annotation["segmentation"], bbox[3], bbox[2])
            # mask = coco_mask.decode(rle)
            # print('type mask', type(mask))
            # mask_resized = torch.tensor(mask, dtype=torch.float32)#.unsqueeze
            # print('shape mask', mask_resized.shape)
            # mask_resized = torch.nn.functional.interpolate(mask_resized.unsqueeze(0), size=(mask_size, mask_size), mode='bilinear', align_corners=False)
            # mask_resized = (mask_resized > 0.5).float().squeeze(0)
            # gt_masks[batch_idx, idx, :, :] = mask_resized

    return gt_labels, gt_locations, num_objects



def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    masks = [item[2] for item in batch]

    images = [to_tensor(img) for img in images]

    # Pad images to the same size
    max_height = max([img.size(1) for img in images])
    max_width = max([img.size(2) for img in images])
    padded_images = []
    for img in images:
        pad_height = max_height - img.size(1)
        pad_width = max_width - img.size(2)
        padded_img = torch.nn.functional.pad(img, (0, pad_width, 0, pad_height))
        padded_images.append(padded_img)
    images = torch.stack(padded_images)
    masks = torch.stack(masks)

    return images, targets, masks



def train(num_epochs=100, batch_size=4, k=100, mask_size=138, lr=0.001, momentum=0.9, weight_decay=0.0005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_dir = "data/coco_holds_resized"
    train_annotations = "data/coco_holds_resized/annotations.json"
    val_data_dir = "data/coco_holds_resized"
    val_annotations = "data/coco_holds_resized/annotations.json"

    # data_transforms = transforms.Compose([
        # transforms.Resize((550, 550)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    train_dataset = CocoDetection(root=train_data_dir, annFile=train_annotations)
    val_dataset = CocoDetection(root=val_data_dir, annFile=val_annotations)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate)
    

    yolact = Yolact().to(device)
    criterion = MultiboxLoss().to(device)
    optimizer = optim.SGD(yolact.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    print("Starting Training")
    losses = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        yolact.train()
        running_loss = 0.0
        for i, (images, targets, gt_masks) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            gt_masks = gt_masks.to(device)
            gt_labels, gt_locations, num_objects = transform_targets(targets, batch_size=batch_size, k=k, mask_size=mask_size)
            
            optimizer.zero_grad()

            bboxes, classes, masks, columns_to_keep = yolact(images)

            loss = criterion(classes, bboxes, masks, gt_labels, gt_locations, gt_masks, num_objects)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Training loss: {running_loss / len(train_loader):.4f}")
        losses.append(running_loss)

        if epoch % 10 == 9:
            yolact.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for i, (images, targets, gt_masks) in enumerate(val_loader):
                    images = images.to(device)
                    gt_labels, gt_locations, num_objects = transform_targets(targets, 2, 100, 138)

                    bboxes, classes, masks, columns_to_keep = yolact(images)
                    loss = criterion(classes, bboxes, masks, gt_labels, gt_locations, gt_masks, num_objects)

                    validation_loss += loss.item()

            print(f"Validation loss: {validation_loss / len(val_loader):.4f}")

        if epoch % 10 == 9:
            torch.save(yolact.state_dict(), f"model/yolact_{epoch + 1}.pth")
                    

    print("Finished Training")


def main():
    parser = argparse.ArgumentParser(description='Train Yolact')
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-k', '--k', type=int, default=100, help='Number of objects to detect')
    parser.add_argument('-ms', '--mask_size', type=int, default=138, help='Size of mask')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005, help='Weight decay')
    args = parser.parse_args()

    train(num_epochs=args.num_epochs, batch_size=args.batch_size, 
          k=args.k, mask_size=args.mask_size, 
          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)


if __name__ == "__main__":
    main()
