import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from src import UNet

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def create_visual_anno(anno):
    """"""
    #assert np.max(anno) <= 4, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 255, 255],  # cornsilk
        2: [255, 0, 0],  # cornflowerblue
        3: [50, 183, 250],  # mediumAquamarine[50, 183, 250]
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno



def main():
    classes = 2  # exclude background

    ori_path='/'
    weights_path = os.path.join(ori_path,'best_model.pth')
    img_path = "/15.jpg"
    save_path=os.path.join(ori_path,'15.png')

    #roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    assert os.path.exists( weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    #assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #print("using {} device.".format(device))

    strGPUs = [str(x) for x in [1]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(strGPUs)
    device = torch.device("cuda")

    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    model = torch.nn.DataParallel(model).to(device)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    #model.to(device)

    # load roi mask
    # roi_img = Image.open(roi_mask_path).convert('L')
    # roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    #data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
    img = F.to_tensor(original_img)
    img = F.normalize(img, mean=mean, std=std)

    #img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        xx, output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        #out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = output['out'].argmax(1).squeeze(0)
        #prediction = torch.argmax(torch.softmax(output['out'], dim=1), dim=1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)

        unique, count = np.unique(prediction, return_counts=True)
        a = dict(zip(unique, count))
        print(a)

        roi1 = create_visual_anno(prediction)
        pil_image = Image.fromarray(roi1)
        pil_image.save(save_path)


if __name__ == '__main__':
    main()
