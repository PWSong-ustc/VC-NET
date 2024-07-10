import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa  # 导入iaa
from torchvision.transforms import functional as F
import torch
class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # check files
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

class DriveDataset1(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset1, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE_1", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split(".")[0] + "_manual1.png")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
        #                  for i in img_names]
        # # check files
        # for i in self.roi_mask:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        # roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

def mask_to_onehot(mask, ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask,-1)
    for colour in range (3):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map

def augment_seg(img_aug,img,seg):
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapsOnImage(seg, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug, segmap_aug

class DriveDataset_mutil_class(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset_mutil_class, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "EC_multi_class_3", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".jpg")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.roi_mask = [os.path.join(data_root, "1st_manual", i.split(".")[0] + "_manual1.png")
                       for i in img_names]
        for i in range(len(self.roi_mask)):
            self.roi_mask[i] = self.roi_mask[i].replace(".jpg", ".png")

        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

        # self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
        #                for i in img_names]
        # # check files
        # for i in self.manual:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

        # roi_names = [i for i in os.listdir(os.path.join(data_root, "label")) if i.endswith(".png")]

        # for i in range(len(self.roi_mask)):
        #     self.roi_mask[i] = self.roi_mask[i].replace(".jpg", ".png")
        # check files
        # for i in self.roi_mask:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        # manual = Image.open(self.manual[idx]).convert('L')
        # manual = np.array(manual) / 255
        mask = Image.open(self.roi_mask[idx]).convert('L')

        # img=np.array(img)
        mask = np.array(mask)

        mask[mask == 255] = 1
        #mask[mask == 127] = 2
        mask[mask == 76] = 2

        # roi_mask = 255 - np.array(roi_mask)
        # mask = np.clip(roi_mask, a_min=0, a_max=255)
        # mask =  mask/255.0

        # unique, count = np11.unique(mask, return_counts=True)
        # a = dict(zip(unique, count))
        # print(a)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)
        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        if self.flag == "training":
            img, mask = self.transforms(img, mask)

            img = np.array(img)
            mask = np.array(mask)

            img = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask)

            img, mask = augment_seg(self.img_aug, img, mask)
            # mask=np.array(mask)
            mask = torch.from_numpy(mask.astype(np.float32))
            img = np.ascontiguousarray(img)
            img = F.to_tensor(img)
            img = F.normalize(img, mean=mean, std=std)
            # mask=F.normalize(mask, mean=mean, std=std)

        if self.flag == "test":
            # img, mask = self.transforms(img, mask)
            mask = np.array(mask)

            mask = torch.from_numpy(mask.astype(np.float32))
            img = F.to_tensor(img)
            img = F.normalize(img, mean=mean, std=std)
            # mask = F.normalize(mask, mean=mean, std=std)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=1)
        # a=batched_imgs
        # b=batched_targets
        return batched_imgs, batched_targets
        # return images, targets

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

