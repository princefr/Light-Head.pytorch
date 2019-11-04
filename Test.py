from .Model import LightHead
import torchvision




backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone = LightHead(1280, backbone, mode="S", out_size="Thin")