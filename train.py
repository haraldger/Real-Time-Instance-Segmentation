import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from model.yolact import Yolact
from multibox_loss import MultiboxLoss
from torchvision.transforms.functional import to_tensor
import numpy as np
from pycocotools import mask as maskUtils
import torch
from pycocotools import mask as coco_mask

def transform_targets(targets, batch_size, k, mask_size):
    gt_labels = torch.zeros((batch_size, k), dtype=torch.long) # TODO different dimension than expected in multibox loss?
    gt_locations = torch.zeros((batch_size, k, 4), dtype=torch.float32)
    gt_masks = torch.zeros((batch_size, k, mask_size, mask_size), dtype=torch.float32)
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
            # mask_resized = torch.tensor(mask, dtype=torch.float32).unsqueeze
            # mask_resized = torch.nn.functional.interpolate(mask_resized.unsqueeze(0), size=(mask_size, mask_size), mode='bilinear', align_corners=False)
            # mask_resized = (mask_resized > 0.5).float().squeeze(0)
            # gt_masks[batch_idx, idx, :, :] = mask_resized

    return gt_labels, gt_locations, gt_masks, num_objects



def custom_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

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

    return images, targets



def train():
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

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=custom_collate)
    

    yolact = Yolact().to(device)
    criterion = MultiboxLoss().to(device)
    optimizer = optim.SGD(yolact.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    num_epochs = 2
    print("Starting Training")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        yolact.train()
        running_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            gt_labels, gt_locations, gt_masks, num_objects = transform_targets(targets, 2, 100, 138)

            optimizer.zero_grad()

            bboxes, classes, masks, columns_to_keep = yolact(images)

            loss = criterion(classes, bboxes, masks, gt_labels, gt_locations, gt_masks, num_objects)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    print("Finished Training")

if __name__ == "__main__":
    train()
