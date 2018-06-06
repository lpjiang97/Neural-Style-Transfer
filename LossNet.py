import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
from util import *
from sys import argv
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        # final output to be saved
        self.x = self.content.clone()
        # where to insert loss
        self.content_layer = ['conv_4']
        self.style_layer = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        # weight to update style and content, should be huge on style, little on content
        self.style_weight = style_weight
        self.content_weight = content_weight
        # pre-trained net
        self.vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.optimizer = optim.LBFGS([self.x.requires_grad_()])
        # start with norm layer 
        self.model = nn.Sequential(Normalization(
            torch.tensor([0.485, 0.456, 0.406]).to(device), 
            torch.tensor([0.229, 0.224, 0.225]).to(device)))
    
    def build_model(self):
        content_losses = []
        style_losses = []

        content = self.content.clone()
        style = self.style.clone()
        
        i = 0  # increment every time we see a conv
        for layer in self.vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in self.content_layer:
                # add content loss:
                target = self.model(content).detach()
                content_loss = ContentLoss(target)
                self.model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layer:
                # add style loss:
                target_feature = self.model(style).detach()
                style_loss = StyleLoss(target_feature)
                self.model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # trimming
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break

        self.model = self.model[:(i + 1)]
        return style_losses, content_losses
    
    def train(self, epochs):
        print('Start model...')
        style_losses, content_losses = self.build_model()

        run = [0]
        while run[0] <= epochs:
            def closure():
                # correct the values of updated input image
                self.x.data.clamp_(0, 1)

                self.optimizer.zero_grad()
                self.model(self.x)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= self.style_weight
                content_score *= self.content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            self.optimizer.step(closure)

        # a last correction...
        self.x.data.clamp_(0, 1)


def train_all():
   
    content_images = os.listdir("image/")
    style_images = os.listdir("style/")

    for content_image in content_images:
        for style_image in style_images:
            output_name = content_image[:-4] + "-" + style_image[:-4]
            style_img = image_loader("style/" + style_image, device) 
            content_img = image_loader("image/" + content_image, device) 
            # loss net
            lossnet = LossNet(content_img, style_img)
            lossnet.train(300)
            image_saver(lossnet.x, "output/" + output_name + ".jpg")
            print("one image done!")
    

def main():
    train_all()


if __name__ == "__main__":
    main()
