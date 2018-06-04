import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
from util import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256
loader = transforms.Compose([
             transforms.Scale(imsize),
             transforms.ToTensor()
         ])


def gram(x):
    batch, channel, width, height = x.size()
    # flatten features
    features = x.view(batch * channel, width * height)
    # gram matrix
    G = torch.mm(features, features.t())
    # normalize
    return G.div(batch * channel * width * height)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram(target_feature).detach()

    def forward(self, input):
        G = gram(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class LossNet():
    
    def __init__(self, content, style, style_weight=1000000, content_weight=1):
        # content image
        self.content = content
        # style image
        self.style = style
        # where to insert loss
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        # weight to update style and content, should be huge on style, little on content
        self.style_weight = style_weight
        self.content_weight = content_weight
        # pre-trained net
        self.vgg = models.vgg19(pretrained=True).features.to(device).eval()
        # transformation net
        self.trans_net = nn.Sequential(
            nn.ReflectionPad2d(40),
            nn.Conv2d(3, 32, 9, stride=1, padding=4),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(32, 3, 9, stride=1, padding=4),
        ).to(device)
        # optimizer
        self.optimizer = optim.Adam(self.trans_net.parameters(), lr=1e-3)
        # gram matrix loss
        self.style_loss = StyleLoss()
        
    
    # def build_model(self):
    #     content_losses = []
    #     style_losses = []

    #     content = self.content.clone()
    #     style = self.style.clone()
    #     
    #     i = 0  # increment every time we see a conv
    #     for layer in self.vgg.children():
    #         if isinstance(layer, nn.Conv2d):
    #             i += 1
    #             name = 'conv_{}'.format(i)
    #         elif isinstance(layer, nn.ReLU):
    #             name = 'relu_{}'.format(i)
    #             layer = nn.ReLU(inplace=False)
    #         elif isinstance(layer, nn.MaxPool2d):
    #             name = 'pool_{}'.format(i)
    #         elif isinstance(layer, nn.BatchNorm2d):
    #             name = 'bn_{}'.format(i)
    #         else:
    #             raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

    #         self.model.add_module(name, layer)

    #         if name in self.content_layer:
    #             # add content loss:
    #             target = self.model(content).detach()
    #             content_loss = ContentLoss(target)
    #             self.model.add_module("content_loss_{}".format(i), content_loss)
    #             content_losses.append(content_loss)

    #         if name in self.style_layer:
    #             # add style loss:
    #             target_feature = self.model(style).detach()
    #             style_loss = StyleLoss(target_feature)
    #             self.model.add_module("style_loss_{}".format(i), style_loss)
    #             style_losses.append(style_loss)

    #     # trimming
    #     for i in range(len(self.model) - 1, -1, -1):
    #         if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
    #             break

    #     self.model = self.model[:(i + 1)]
    #     return style_losses, content_losses
    
    def train(self, content):
        self.optimizer.zero_grad()

        content = content.clone()
        style = self.style.clone()
        x = self.transformation_network.forward(content)

        content_loss = 0
        style_loss = 0

        i = 1
        not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
        for layer in list(self.vgg.features):
            layer = not_inplace(layer).to(device)

            x, content, style = layer.forward(x), layer.forward(content), layer.forward(style)

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)

                if name in self.content_layers:
                    content_loss += self.loss(x * self.content_weight, content.detach() * self.content_weight)
                if name in self.style_layers:
                    x_grad, style_grad = self.style_loss.forward(x), self.style_loss.forward(style)
                    style_loss += self.loss(x_grad * self.style_weight, style_grad.detach() * self.style_weight)

            if isinstance(layer, nn.ReLU):
                i += 1

        total_loss = content_loss + style_loss
        total_loss.backward()


def save_images(input, paths):
    N = input.size()[0]
    images = input.data.clone().cpu()
    for n in range(N):
        image = images[n]
        image = image.view(3, imsize, imsize)
        image = unloader(image)
        scipy.misc.imsave(paths[n], image)
 

def main():
    N = 4
    num_epochs = 3

    style_img = image_loader("images/starry.jpg", device)
    content_img = image_loader("images/max.jpg", device)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    coco = datasets.ImageFolder(root='data/contents', transform=loader)
    content_loader = torch.utils.data.DataLoader(coco, batch_size=N, shuffle=True, **kwargs)

    lossnet = LossNet(content_img, style_img)

    for epoch in range(num_epochs):
        for i, content_batch in enumerate(content_loader):
          iteration = epoch * i + i
          content_loss, style_loss, x = lossnet.train(content_batch, style_batch)

          if i % 10 == 0:
              print("Iteration: %d" % (iteration))
              print("Content loss: %f" % (content_loss.data[0]))
              print("Style loss: %f" % (style_loss.data[0]))

          if i % 500 == 0:
              path = "outputs/%d_" % (iteration)
              paths = [path + str(n) + ".png" for n in range(N)]
              save_images(x, paths)

              path = "outputs/content_%d_" % (iteration)
              paths = [path + str(n) + ".png" for n in range(N)]
              save_images(content_batch, paths)
              lossnet.save()


if __name__ == "__main__":
    main()
