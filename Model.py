import torch

class LightHead(torch.nn.Module):
    def __init__(self, in_, backbone, mode="S", out_size="Thin"):
        """

        :param in_: output features of the backbone you try to light head
        :param backbone:
        :param mode:
        :param out_size:
        """
        super(LightHead, self).__init__()
        assert "S" in mode or "L" in mode, "Please specity the correct Light head mode"
        assert "Thin" in out_size or "Large" in out_size, "Please specify the model out size"
        self.backbone = backbone
        if mode == "L":
            self.out_mode = 256
        else:
            self.out_mode = 64


        if out_size =="Thin":
            self.c_out = 10
        else:
            self.c_out = 10 * 7 * 7

        self.conv1 = torch.nn.Conv2d(in_channels=in_, out_channels=self.out_mode, kernel_size=(15, 1), stride=1, padding=(7, 0), bias=True)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.out_mode, out_channels=self.c_out, kernel_size=(1, 15),  stride=1, padding=(0, 7), bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=in_, out_channels=self.out_mode, kernel_size=(15, 1), stride=1, padding=(7, 0), bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=self.out_mode, out_channels=self.c_out, kernel_size=(1, 15), stride=1, padding=(0, 7), bias=True)

    def forward(self, input):
        x_backbone = self.backbone(input)
        x = self.conv1(x_backbone)
        x = self.relu(x)
        x = self.conv2(x)
        x_relu_2 = self.relu(x)

        x = self.conv3(x_backbone)
        x = self.relu(x)
        x = self.conv4(x)
        x_relu_4 = self.relu(x)

        return x_relu_2 + x_relu_4