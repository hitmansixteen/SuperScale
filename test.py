import cv2
import os
import torch
from torch import nn, Tensor, optim
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from collections import OrderedDict
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
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from PIL import Image

def calculate_metrics(image1, image2):
    # Convert images to numpy arrays
    grayscale_image1 = image1.convert("L")
    grayscale_image2 = image2.convert("L")
    image1 = np.array(image1)
    image2 = np.array(image2)
    grayscale_image1 = np.array(grayscale_image1)
    grayscale_image2 = np.array(grayscale_image2)

    # Ensure images are in the range [0, 1]
    image1 = image1.astype(np.float64) / 255.0
    image2 = image2.astype(np.float64) / 255.0

    grayscale_image1 = grayscale_image1.astype(np.float64)
    grayscale_image2 = grayscale_image2.astype(np.float64)
    grayscale_image1.squeeze()
    grayscale_image2.squeeze()
    data_range = grayscale_image1.max() - grayscale_image1.min()
    # Calculate SSIM
    ssim_score, _ = ssim(grayscale_image1, grayscale_image2, data_range=data_range, full=True)

    # Calculate MSE
    mse_value = mean_squared_error(image1 * 255, image2 * 255)

    # Calculate PSNR
    psnr_value = psnr(image1, image2)

    return mse_value, ssim_score, psnr_value

def calculate_metrics_for_folders(input_folder, comparison_folder, output_folder):
    mse_list = []
    ssim_list = []
    psnr_list = []

    for input_image_name in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, input_image_name)
        comparison_image_path = os.path.join(comparison_folder, input_image_name)
        
        input_image = Image.open(input_image_path)
        comparison_image = Image.open(comparison_image_path)

        mse, ssim_score, psnr_value = calculate_metrics(input_image, comparison_image)

        # Store metrics in files
        mse_file_path = os.path.join(output_folder, "mse.txt")
        ssim_file_path = os.path.join(output_folder, "ssim.txt")
        psnr_file_path = os.path.join(output_folder, "psnr.txt")

        with open(mse_file_path, 'a') as f:
            f.write(f"{mse}\n")

        with open(ssim_file_path, 'a') as f:
            f.write(f"{ssim_score}\n")

        with open(psnr_file_path, 'a') as f:
            f.write(f"{psnr_value}\n")

        # Append metrics to lists
        mse_list.append(mse)
        ssim_list.append(ssim_score)
        psnr_list.append(psnr_value)

    # Calculate average metrics
    avg_mse = np.mean(mse_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr = np.mean(psnr_list)

    return avg_mse, avg_ssim, avg_psnr



def image_to_tensor(image, range_norm, half):
    
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    if half:
        tensor = tensor.half()

    return tensor

def load_dataset( device):
   
    paired_test_datasets = PairedImageDataset('testdata', 'testlrdata')
    
    paired_test_dataloader = DataLoader(paired_test_datasets,
                                batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, persistent_workers=False)

    paired_test_data_prefetcher = CUDAPrefetcher(paired_test_dataloader, device)
    return paired_test_data_prefetcher



#Training Image Dataset
class PairedImageDataset(Dataset):

    def __init__(self, paired_gt_images_dir, paired_lr_images_dir):
        
        super(PairedImageDataset, self).__init__()
        if not os.path.exists(paired_lr_images_dir):
            raise FileNotFoundError(f"Registered low-resolution image address does not exist: {paired_lr_images_dir}")
        if not os.path.exists(paired_gt_images_dir):
            raise FileNotFoundError(f"Registered high-resolution image address does not exist: {paired_gt_images_dir}")

        image_files = natsorted(os.listdir(paired_lr_images_dir))
        self.paired_gt_image_file_names = [os.path.join(paired_gt_images_dir, x) for x in image_files]
        self.paired_lr_image_file_names = [os.path.join(paired_lr_images_dir, x) for x in image_files]
        print(self.paired_gt_image_file_names)
        print(self.paired_lr_image_file_names)

    def __getitem__(self, batch_index):
        
        gt_image = cv2.imread(self.paired_gt_image_file_names[batch_index]).astype(np.float32) / 255.
        lr_image = cv2.imread(self.paired_lr_image_file_names[batch_index]).astype(np.float32) / 255.

        
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)


        gt_tensor = image_to_tensor(gt_image, False, False)
        lr_tensor = image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor, "lr": lr_tensor, "image_name": self.paired_lr_image_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.paired_lr_image_file_names)

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


def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



#Training code
def build_model(device: torch.device):
    g_model = EsrganGenerator(in_channels=3, out_channels=3, channels=64, growth_channels=32, num_rrdb=23)
    
    g_model = g_model.to(device)

    return g_model


def test(g_model, test_data_prefetcher, device):
    save_image = False
    save_image_dir = ""

    if 'testfolder':
        save_image = True
        save_image_dir = os.path.join('testfolder', 'div2k')
        make_directory(save_image_dir)

    batches = len(test_data_prefetcher)
    if batches > 100:
        print_freq = 100
    else:
        print_freq = batches

    g_model.eval()

    with torch.no_grad():
        batch_index = 0

        test_data_prefetcher.reset()
        batch_data = test_data_prefetcher.next()

        while batch_data is not None:
            gt = batch_data["gt"].to(device, non_blocking=True)
            lr = batch_data["lr"].to(device, non_blocking=True)

            sr = g_model(lr)

            if batch_data["image_name"] == "":
                raise ValueError("The image_name is None, please check the dataset.")
            if save_image:
                image_name = os.path.basename(batch_data["image_name"][0])
                sr_image = tensor_to_image(sr, False, False)
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_image_dir, image_name), sr_image)

            batch_data = test_data_prefetcher.next()

            gt_image = tensor_to_image(gt, False, False)
            sr_image = tensor_to_image(sr, False, False)

            ssim = 0
            psnr = 0
            #fid = calculate_fid(gt_image, sr_image)
            #kid = calculate_kid(gt_image, sr_image)
            fid = 0
            kid = 0
            batch_index += 1

        
            print(f"Batch no: {batch_index} psnr: {psnr} ssim: {ssim} fid: {fid} ssim: {kid}")
            


#Image post processing
def tensor_to_image(tensor, range_norm, half):

    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image

def load_model_from_tar(tar_file_path):
    # Open the tar file
    with tarfile.open(tar_file_path, 'r') as tar:
        pth_file = tar.extractfile(tar.getmembers()[0]) 

        return torch.load(pth_file)
    

def generate_graphs(epoch, mse_values, ssim_values, psnr_values):
    plt.figure(figsize=(10, 7))
    plt.plot(mse_values, label='MSE', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Average MSE per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'graphs/mses/epoch_mse.png')

    # Generate graph for SSIM
    plt.figure(figsize=(10, 7))
    plt.plot(ssim_values, label='SSIM', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Average SSIM per Epoch')
    plt.ylim(0, 1)  # Set y-axis range for SSIM
    plt.legend()
    plt.grid(True)
    plt.savefig(f'graphs/ssims/epoch_ssim.png')

    # Generate graph for PSNR
    plt.figure(figsize=(10, 7))
    plt.plot(psnr_values, label='PSNR', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Average PSNR per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'graphs/psnrs/epoch_psnr.png')

# Example usage:
tar_file_path = 'model.tar'

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

scaler = amp.GradScaler()

start_epoch = 0
best_psnr = 0.0
best_ssim = 0.0

device = torch.device("cuda", 0)

avg_mse = []
avg_ssim = []
avg_psnr = []

paired_test_data_prefetcher = load_dataset(device)
g_model = build_model(device)

g_model = g_model.to(device)

for epoch in range(241, 251):
    g_model.load_state_dict(load_model_from_tar('weights/model_epoch_' + str(epoch) + '.tar'))
    test(g_model, paired_test_data_prefetcher, device)
    avgm, avgs, avgp = calculate_metrics_for_folders('testfolder/div2k', 'testdata', 'metrics')
    avg_mse.append(avgm)
    avg_ssim.append(avgs)
    avg_psnr.append(avgp)
    generate_graphs(epoch, avg_mse, avg_ssim, avg_psnr)


