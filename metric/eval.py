import argparse
# import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import lpips
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--path_real', default='E:/DataSet/KAIST/testB', type=str)
    parser.add_argument('-p2', '--path_fake', type=str)
    parser.add_argument('-s', '--size', type=int, default=256)
    parser.add_argument('-o', '--output_name', type=str)

    args = parser.parse_args()
    real_names = list(glob.glob('{}/*.png'.format(args.path_real)))
    if len(real_names) == 0:
        real_names = list(glob.glob('{}/*.jpg'.format(args.path_real)))
    print(real_names[0])

    fake_names = list(glob.glob('{}/*.png'.format(args.path_fake)))
    if len(fake_names) == 0:
        fake_names = list(glob.glob('{}/*.jpg'.format(args.path_fake)))
    print(fake_names[0])
    print(len(real_names), len(fake_names))
    assert len(real_names) == len(fake_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_names.sort()
    fake_names.sort()

    lpips_model = lpips.LPIPS(net="alex")
    psnr_model = PeakSignalNoiseRatio(data_range=255.0)
    ssim_model = StructuralSimilarityIndexMeasure(data_range=255.0)
    mse_model = nn.MSELoss()

    avg_psnr = []
    avg_ssim = []
    avg_lpips = []
    avg_mse = []
    idx = 0

    # inception_model = torchvision.models.inception_v3(pretrained=True)
    # transform = transforms.Compose([transforms.Resize(299),transforms.CenterCrop(299),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    fid = calculate_fid_given_paths([args.path_real, args.path_fake], batch_size=50, device=device, dims=2048)

    for rname, fname in tqdm(zip(real_names, fake_names)):
        real_img = Image.open(rname).convert('RGB')  # 转换为RGB模式，0-255
        real_img = np.array(real_img.resize((args.size, args.size)))
        fake_img = Image.open(fname).convert('RGB')  # 转换为RGB模式，0-255
        fake_img = np.array(fake_img.resize((args.size, args.size)))

        real_tensor = torch.tensor(real_img).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
        fake_tensor = torch.tensor(fake_img).permute(2, 0, 1).unsqueeze(0).float()

        ssim = ssim_model(real_tensor, fake_tensor)

        psnr = psnr_model(real_tensor, fake_tensor)

        mse = mse_model(real_tensor, fake_tensor)

        real_lpips = real_tensor / 255
        fake_lpips = fake_tensor / 255
        # lpips模型输入是否需要数据在[-1,1]之间？
        lpips = lpips_model(real_lpips, fake_lpips).item()

        avg_psnr.append(psnr)
        avg_ssim.append(ssim)
        avg_lpips.append(lpips)
        avg_mse.append(mse)

    print(idx)

    # log
    with open(args.output_name, 'a+') as f:
        f.write(args.path_fake + '\n')
        f.write('# Validation # PSNR: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_psnr), np.std(avg_psnr)) + '\n')
        f.write('# Validation # SSIM: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_ssim), np.std(avg_ssim)) + '\n')
        f.write('# Validation # LPIPS: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_lpips), np.std(avg_lpips)) + '\n')
        f.write('# Validation # MSE: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_mse), np.std(avg_mse)) + '\n')
        f.write('# Validation # FID: mean: {:.4e}, std: {:.4e}'.format(np.mean(fid) , np.std(fid))+ '\n')

        print('# Validation # PSNR: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_psnr), np.std(avg_psnr)))
        print('# Validation # SSIM: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_ssim), np.std(avg_ssim)))
        print('# Validation # LPIPS: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_lpips), np.std(avg_lpips)))
        print('# Validation # MSE: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_mse), np.std(avg_mse)))
        print('# Validation # FID: mean: {:.4e}, std: {:.4e}'.format(np.mean(fid), np.std(fid)))
