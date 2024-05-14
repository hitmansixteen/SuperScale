from ultralytics import YOLO
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import json
from flask_cors import CORS
import cv2
import os
import torch
import numpy as np
import random
from torch.cuda import amp
import cv2
import os
import torch
from torch import nn, Tensor, optim
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
import random
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted
import time
from torchvision.transforms import functional as F_vision
import base64


class EsrganGenerator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, channels = 64, growth_channels = 32, num_rrdb = 23, upscale = 4):
        super(EsrganGenerator, self).__init__()
        self.upscale = upscale

        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        feature_extractor_network = []
        for _ in range(num_rrdb):
            feature_extractor_network.append(ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*feature_extractor_network)

        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        self.upsampling1 = nn.Sequential(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)), nn.LeakyReLU(0.2, True))
        self.upsampling2 = nn.Sequential(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)), nn.LeakyReLU(0.2, True))

        self.conv3 = nn.Sequential(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)), nn.LeakyReLU(0.2, True))

        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.2
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _forward_impl(self, x):
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.upsampling2(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.conv3(x)
        x = self.conv4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
class ResidualDenseBlock(nn.Module):

    def __init__(self, channels, growth_channels):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))

        x = torch.mul(out5, 0.2)
        x = torch.add(x, identity)

        return x
    
class ResidualResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth_channels):
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x):
        identity = x

        x = self.rdb1(x)
        x = self.rdb2(x)
        x = self.rdb3(x)

        x = torch.mul(x, 0.2)
        x = torch.add(x, identity)

        return x
    

def build_model(device: torch.device):
    g_model = EsrganGenerator(in_channels=3, out_channels=3, channels=64, growth_channels=32, num_rrdb=23)
    
    g_model = g_model.to(device)

    return g_model

#python3 object_detector.py
#http://localhost:8080

app = Flask(__name__)

@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    cr_image_base64 = encode_image_to_base64(boxes)
    return Response(json.dumps({"image_data": cr_image_base64}), mimetype='application/json')

def image_to_tensor(image, range_norm, half):
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    if half:
        tensor = tensor.half()

    return tensor

def detect_objects_on_image(buf):
    image = np.array(buf).astype(np.float32) / 255.0
    device = 'cuda'
    device = torch.device("cuda", 0)
    tensor = image_to_tensor(image, False, False).unsqueeze_(0)
    input_tensor = tensor.to(device, non_blocking=True)
    g_model = build_model(device)
    g_model = g_model.to(device)
    checkpoint = torch.load('model_epoch_120.pth')
    model_state_dict = checkpoint
    g_model.load_state_dict(model_state_dict)
    scaler = amp.GradScaler()
    g_model.eval()
    with torch.no_grad():
        sr_tensor = g_model(input_tensor)
    image = sr_tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    cr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.png', cr_image)
    return cr_image

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    base64_encoded = base64.b64encode(buffer)
    return base64_encoded.decode()

if __name__ == "__main__":
    app.run()