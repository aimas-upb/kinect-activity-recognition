import torch.nn as nn
import torch
import torchvision.models as models


class RegressionModel(nn.Module):

    def __init__(self, no_of_joints, joint_size):
        super(RegressionModel, self).__init__()
        self.no_of_joints = no_of_joints
        self.joint_size = joint_size

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.conv0 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.fc = nn.Linear(288 * self.joint_size, self.no_of_joints * self.joint_size)

    def forward(self, input):
        (batch_size, in_channel, height, width) = input.size()
        output = self.resnet(input)

        output = self.conv0(output)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)

        output = output.view((batch_size, -1))
        output = self.fc(output)
        output = output.view((batch_size, self.no_of_joints, self.joint_size))
        return output

