import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.transforms.functional import normalize
from runner.src.models.isnet import ISNetDIS
import cv2

import warnings
warnings.simplefilter("ignore")

class BackgroundRemoval():
    def __init__(self, pmodel='saved_models/IS-Net/isnet-general-use.pth'):
        self.model_path = pmodel
        self.input_size = [1024, 1024]
        self.net = self.load_model()

    def load_model(self):
        net = ISNetDIS()
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(self.model_path))
            net = net.cuda()
        else:
            net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        net.eval()
        return net

    def pre_process(self, image):
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]
        im_shp = image.shape[0:2]
        im_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), self.input_size, mode="bilinear").type(torch.uint8)
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        if torch.cuda.is_available():
            image = image.cuda()

        return image, im_shp

    def post_process(self, tensor, shape):
        result = torch.squeeze(F.upsample(tensor[0][0], shape, mode='bilinear'), 0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        final_result = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)[:, :, 0]
        return final_result

    def image_prediction(self, image: np.ndarray) -> np.ndarray:
        tensor, shape = self.pre_process(image)
        result = self.net(tensor)
        rarray = self.post_process(result, shape)
        return rarray

if __name__ == '__main__':
    processor = BackgroundRemoval()
    image = '/media/banglv/4TbDATA/backup_banglv1/django_removal/data/ex1.jpg'
    arr = cv2.imread(image)
    result = processor.image_prediction(arr)
    plt.imshow(result)
    plt.show()