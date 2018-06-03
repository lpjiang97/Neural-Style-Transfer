import scipy
import torch
from PIL import Image
import torchvision.transforms as transforms


IMSIZE = 256


def create_loader(imsize):
    return transforms.Compose([
        transforms.Resize((imsize, imsize)), transforms.ToTensor()])


def image_loader(image_name, device):
    image = Image.open(image_name)
    # batch size 1
    loader = create_loader(IMSIZE)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def image_saver(input, path):
    image = input.data.clone().cpu()
    image = image.view(3, IMSIZE, IMSIZE)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    scipy.misc.imsave(path, image)

