import torch
import torch.nn as nn
from torchinfo import summary

# custom VGG13 model, more: https://arxiv.org/pdf/1608.01041.pdf


class myModel(nn.Module):
    def __init__(self, num_classes, mode="train") -> None:
        super(myModel, self).__init__()
        self.mode = mode
        self.convBlock = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fullyConnected = nn.Sequential(
            nn.Linear(14 * 14 * 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, x):
        x = self.convBlock(x)
        # print(x.shape)
        x = x.view(-1, 14 * 14 * 256)
        x = self.fullyConnected(x)
        if self.mode == "train":
            return x
        elif self.mode == "pred":
            return self.softmax(x)

        """
        NOTE:   this is probably not the most efficient way to have softmax in the
                last layer when you call predict()
                + more simple solution: apply torch.nn.functional.softmax(dim=1)
                to the last layer in the function predict() itself.
                Ex:
                import torch.nn.functional as F
                ...
                output = model(input)
                output = F.softmax(dim=1)(output)
                (I haven't test this one yet, it may not work but you get the idea)
        """
