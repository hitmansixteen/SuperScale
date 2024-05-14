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
import imgaug.augmenters as iaa
import glob
from tqdm import tqdm
import tarfile



def save_model_to_tar(model, epoch, file_path):

    torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')


    with tarfile.open(file_path, 'w') as tar:
        tar.add(f'model_epoch_{epoch}.pth', arcname=f'model_epoch_{epoch}.pth')


    os.remove(f'model_epoch_{epoch}.pth')




def augment_images():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Crop(percent=(0, 0.1)),
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
        iaa.Multiply((0.8, 1.2)),
        iaa.Affine(
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)

    folder_path = "DIV2k_train_HR"

    image_files = glob.glob(os.path.join(folder_path, "*.png"))


    for image_file in image_files:

        img = cv2.imread(image_file)
        
        original_file_name, ext = os.path.splitext(os.path.basename(image_file))
        if 'augmented' not in original_file_name:
            augmented_img = seq(image=img)
            augmented_save_path = os.path.join(folder_path, f"{original_file_name}_augmented{ext}")
            cv2.imwrite(augmented_save_path, augmented_img)

def random_crop_torch(gt_images ,lr_images, gt_patch_size, upscale_factor):

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    lr_patch_size = gt_patch_size // upscale_factor

    lr_top = random.randint(0, lr_image_height - lr_patch_size)
    lr_left = random.randint(0, lr_image_width - lr_patch_size)

    if input_type == "Tensor":
        lr_images = [lr_image[
                     :,
                     :,
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size] for lr_image in lr_images]
    else:
        lr_images = [lr_image[
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size,
                     ...] for lr_image in lr_images]

    gt_top, gt_left = int(lr_top * upscale_factor), int(lr_left * upscale_factor)

    if input_type == "Tensor":
        gt_images = [v[
                     :,
                     :,
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size] for v in gt_images]
    else:
        gt_images = [v[
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size,
                     ...] for v in gt_images]

    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images

def random_rotate_torch(gt_images, lr_images, upscale_factor, angles, gt_center = None, lr_center = None, rotate_scale_factor = 1.0):
    
    angle = random.choice(angles)

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    if lr_center is None:
        lr_center = [lr_image_width // 2, lr_image_height // 2]

    lr_matrix = cv2.getRotationMatrix2D(lr_center, angle, rotate_scale_factor)

    if input_type == "Tensor":
        lr_images = [F_vision.rotate(lr_image, angle, center=lr_center) for lr_image in lr_images]
    else:
        lr_images = [cv2.warpAffine(lr_image, lr_matrix, (lr_image_width, lr_image_height)) for lr_image in lr_images]

    gt_image_width = int(lr_image_width * upscale_factor)
    gt_image_height = int(lr_image_height * upscale_factor)

    if gt_center is None:
        gt_center = [gt_image_width // 2, gt_image_height // 2]

    gt_matrix = cv2.getRotationMatrix2D(gt_center, angle, rotate_scale_factor)

    if input_type == "Tensor":
        gt_images = [F_vision.rotate(gt_image, angle, center=gt_center) for gt_image in gt_images]
    else:
        gt_images = [cv2.warpAffine(gt_image, gt_matrix, (gt_image_width, gt_image_height)) for gt_image in gt_images]

    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images

def random_horizontally_flip_torch(gt_images, lr_images, p = 0.5):
    
    flip_prob = random.random()

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            lr_images = [F_vision.hflip(lr_image) for lr_image in lr_images]
            gt_images = [F_vision.hflip(gt_image) for gt_image in gt_images]
        else:
            lr_images = [cv2.flip(lr_image, 1) for lr_image in lr_images]
            gt_images = [cv2.flip(gt_image, 1) for gt_image in gt_images]

   
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images

def random_vertically_flip_torch(gt_images, lr_images, p = 0.5):

    flip_prob = random.random()

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            lr_images = [F_vision.vflip(lr_image) for lr_image in lr_images]
            gt_images = [F_vision.vflip(gt_image) for gt_image in gt_images]
        else:
            lr_images = [cv2.flip(lr_image, 0) for lr_image in lr_images]
            gt_images = [cv2.flip(gt_image, 0) for gt_image in gt_images]

    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images

def image_to_tensor(image, range_norm, half):
    
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    if half:
        tensor = tensor.half()

    return tensor

def load_dataset( device):
    augment_images()
    degenerated_train_datasets = BaseImageDataset('DIV2K_train_HR', None, 4)

    degenerated_train_dataloader = DataLoader(degenerated_train_datasets, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, persistent_workers=False)
    
    train_data_prefetcher = CUDAPrefetcher(degenerated_train_dataloader, device)

    return train_data_prefetcher

class BaseImageDataset(Dataset):

    def __init__(self, gt_images_dir, lr_images_dir = None, upscale_factor = 4):

        super(BaseImageDataset, self).__init__()
        
        image_file_names = natsorted(os.listdir(gt_images_dir))
        self.lr_image_file_names = None
        self.gt_image_file_names = [os.path.join(gt_images_dir, image_file_name) for image_file_name in image_file_names]

        self.upscale_factor = upscale_factor

    def __getitem__(self, batch_index):
        gt_image = cv2.imread(self.gt_image_file_names[batch_index]).astype(np.float32)
        gt_image = cv2.resize(gt_image, (1500, 1500))
        lr_image = cv2.resize(gt_image, (1500 // 4, 1500 // 4)) / 255.
        gt_image = gt_image / 255.
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        gt_tensor = image_to_tensor(gt_image, False, False)
        lr_tensor = image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor, "lr": lr_tensor}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)



#ESRGAN model classes

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
    
class EsrganDiscriminator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, channels = 64):
        super(EsrganDiscriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, int(2 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(2 * channels), int(2 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(2 * channels), int(4 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(4 * channels), int(4 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(4 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(8 * channels), int(8 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(8 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(8 * channels), int(8 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(8 * channels) * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, out_channels)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out
    
class ContentLoss(nn.Module):
    def __init__(self, model_weights_path = "", feature_nodes = None, feature_normalize_mean = None, feature_normalize_std = None):
        super(ContentLoss, self).__init__()

        if model_weights_path == "":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif model_weights_path is not None and os.path.exists(model_weights_path):
            checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
            if "state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError("Model weight file not found")

        self.feature_extractor = create_feature_extractor(model, feature_nodes)

        self.feature_extractor_nodes = feature_nodes

        self.normalize = transforms.Normalize(feature_normalize_mean, feature_normalize_std)

        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False
        self.feature_extractor.eval()

    def forward(self, sr_tensor, gt_tensor):
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
        device = sr_tensor.device

        losses = []
        
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        for i in range(len(self.feature_extractor_nodes)):
            losses.append(F_torch.l1_loss(sr_feature[self.feature_extractor_nodes[i]],
                                          gt_feature[self.feature_extractor_nodes[i]]))

        losses = torch.Tensor([losses]).to(device)

        return losses
    
class CUDAPrefetcher:

    def __init__(self, dataloader, device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)





#Training code
def build_model(device: torch.device):
    g_model = EsrganGenerator(in_channels=3, out_channels=3, channels=64, growth_channels=32, num_rrdb=23)
    d_model = EsrganDiscriminator(in_channels=3, out_channels=1, channels=64)

    g_model = g_model.to(device)
    d_model = d_model.to(device)

    return g_model, d_model

def define_loss(device):
    pixel_criterion = nn.L1Loss()

    feature_criterion = ContentLoss("", ["features.34"], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    adversarial_criterion = nn.BCEWithLogitsLoss()
    
    pixel_criterion = pixel_criterion.to(device)
    feature_criterion = feature_criterion.to(device)
    adversarial_criterion = adversarial_criterion.to(device)

    return pixel_criterion, feature_criterion, adversarial_criterion

def define_scheduler(g_optimizer, d_optimizer):
   
    g_scheduler = lr_scheduler.MultiStepLR(g_optimizer, [ 16, 32, 64, 104 ], 0.5)
    d_scheduler = lr_scheduler.MultiStepLR(d_optimizer, [ 16, 32, 64, 104 ], 0.5)


    return g_scheduler, d_scheduler

def define_optimizer(g_model, d_model):
    g_optimizer = optim.Adam(g_model.parameters(), 0.0005, [0.9, 0.999], 0.0001, 0.0)
    d_optimizer = optim.Adam(d_model.parameters(), 0.000005, [0.9, 0.999], 0.0001, 0.0)

    return g_optimizer, d_optimizer

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train(g_model, d_model, train_data_prefetcher, pixel_criterion, content_criterion, adversarial_criterion, g_optimizer, d_optimizer, epoch, scaler, device):
    
    batches = len(train_data_prefetcher)

    g_model.train()
    d_model.train()

    pixel_weight = torch.Tensor([0.01]).to(device)
    content_weight = torch.Tensor([1.0]).to(device)
    adversarial_weight = torch.Tensor([0.005]).to(device)

    batch_index = 0
    train_data_prefetcher.reset()
    end = time.time()
    batch_data = train_data_prefetcher.next()

    batch_size = batch_data["gt"].shape[0]

    real_label = torch.full([batch_size, 1], 1.0, dtype=torch.float, device=device)
    fake_label = torch.full([batch_size, 1], 0.0, dtype=torch.float, device=device)

    while batch_data is not None:
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

       
        gt, lr = random_crop_torch(gt, lr, 128, 4)
        gt, lr = random_rotate_torch(gt, lr, 4, [0, 90, 180, 270])
        gt, lr = random_vertically_flip_torch(gt, lr)
        gt, lr = random_horizontally_flip_torch(gt, lr)

       
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        g_model.zero_grad(set_to_none=True)

        with amp.autocast():
            sr = g_model(lr)
    
            new_tensor = gt
            new_tensor.detach()
            gt_output = d_model(new_tensor)
            sr_output = d_model(sr)
            pixel_loss = pixel_criterion(sr, gt)
            content_loss = content_criterion(sr, gt)
            d_loss_gt = adversarial_criterion(gt_output - torch.mean(sr_output), fake_label) * 0.5
            d_loss_sr = adversarial_criterion(sr_output - torch.mean(gt_output), real_label) * 0.5
            adversarial_loss = d_loss_gt + d_loss_sr
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_loss))
            content_loss = torch.sum(torch.mul(content_weight, content_loss))
            adversarial_loss = torch.sum(torch.mul(adversarial_weight, adversarial_loss))
            g_loss = pixel_loss + content_loss + adversarial_loss

        scaler.scale(g_loss).backward()

        scaler.step(g_optimizer)
        scaler.update()

        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        d_model.zero_grad(set_to_none=True)

        with amp.autocast():
            gt_output = d_model(gt)
            sr_output = d_model(sr.detach().clone())
            d_loss_gt = adversarial_criterion(gt_output - torch.mean(sr_output), real_label) * 0.5
        scaler.scale(d_loss_gt).backward(retain_graph=True)

        with amp.autocast():
            sr_output = d_model(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output - torch.mean(gt_output), fake_label) * 0.5
        scaler.scale(d_loss_sr).backward()

        d_loss = d_loss_gt + d_loss_sr

        scaler.step(d_optimizer)
        scaler.update()

        end = time.time()

        batch_data = train_data_prefetcher.next()

        batch_index += 1
        file_path = "d_loss_data.txt"
        try:
            with open(file_path, 'a') as f:
                f.write(str(d_loss) + '\n')
        except FileNotFoundError:
            with open(file_path, 'w') as f:
                f.write(str(d_loss) + '\n')
        file_path = "g_loss_data.txt"
        try:
            with open(file_path, 'a') as f:
                f.write(str(g_loss) + '\n')
        except FileNotFoundError:
            with open(file_path, 'w') as f:
                f.write(str(g_loss) + '\n')

        print(f"Batch no: {batch_index} G loss: {g_loss} D Loss: {d_loss}")
           





#Image post processing
def tensor_to_image(tensor, range_norm, half):

    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

scaler = amp.GradScaler()

start_epoch = 0

device = torch.device("cuda", 0)


train_data_prefetcher = load_dataset(device)
g_model, d_model = build_model(device)
pixel_criterion, feature_criterion, adversarial_criterion = define_loss(device)
g_optimizer, d_optimizer = define_optimizer(g_model, d_model)
g_scheduler, d_scheduler = define_scheduler(g_optimizer, d_optimizer)

g_model = g_model.to(device)


d_model = d_model.to(device)



for epoch in tqdm(range(start_epoch, 250)):
    train(g_model, d_model, train_data_prefetcher, pixel_criterion, feature_criterion, adversarial_criterion, g_optimizer, d_optimizer, epoch, scaler, device)

    g_scheduler.step()
    d_scheduler.step()
    
    file_path = f"weights/model_epoch_{epoch + 1}.tar"
    save_model_to_tar(g_model, epoch + 1, file_path)
    print("\n")