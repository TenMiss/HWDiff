import numpy as np
import cv2
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import utils
from natsort import natsorted
from torch.nn.parallel import DataParallel, DistributedDataParallel
from glob import glob
from copy import deepcopy

import sys
sys.path.append("/home/zhengchaobing/lixiaopan/mycode")
from basicsr.models.archs.DFT_arch import DFT
from basicsr.models.archs.deblurnet_arch import DeblurNet_v2
from basicsr.models.archs import define_network
from skimage import img_as_ubyte
import os
from metrics import calculate_psnr, calculate_ssim

os.environ["CUDA_VISIBLE_DEVICES"]='1'

def overlapping_grid_indices(x_cond):
    _, c, h, w = x_cond.shape

    h_list = [h]

    w_list = [w]

    return h_list, w_list

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps_overlapping(input_, rough_preds, model_restoration, device, betas, seq, seq_next, eta):

    x = torch.randn(input_.size(), device=device)
    n = input_.size(0)
    x0_preds = []
    xs = [x]

    for k, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * k).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())
        xt = xs[-1].to('cuda')

        et = model_restoration(torch.cat([data_transform(input_), xt], dim=1), data_transform(rough_preds), t)

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t.to('cpu'))

        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def model_to_device(net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(torch.device('cuda'))
        net = DataParallel(net)
        return net



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Deraining using C2F-DFT')

    parser.add_argument('--input_dir', default='/data1/lxp/dataset/HIDE/input', type=str, help='Directory of validation images')

    parser.add_argument('--result_dir', default='./results_16/', type=str, help='Directory for results')
    parser.add_argument('--weights', default='/data1/lxp/cc/experiments/Deblurring_HFG_fine_2/models/net_g_300000.pth', type=str, help='Path to weights')
    # parser.add_argument('--weights', default='/home/zhengchaobing/lixiaopan/mycode/experiments/Deblurring_HFG_fine/models/net_g_40000.pth', type=str, help='Path to weights')
    args = parser.parse_args()

    ####### Load yaml #######
    yaml_file = '/home/zhengchaobing/lixiaopan/HFG-main/Options/Deblurring_HFG_fine.yml'
    load_path_rough = '/data1/lxp/pretained_models/net_g_400000_res33.pth'

    
    
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    opt = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
    device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')


    # define rough network
    # load roungh modle 
    rough_s = opt['network_g_rough'].pop('type')
    model_rough_g = DeblurNet_v2(**opt['network_g_rough'])
    checkpoint_rough = torch.load(load_path_rough)

    model_rough_g.load_state_dict(checkpoint_rough['params'])
    model_rough_g.cuda()
    model_rough_g = nn.DataParallel(model_rough_g)
    model_rough_g.eval()
    

    s = opt['network_g'].pop('type')
    model_restoration = DFT(**opt['network_g'])

    checkpoint = torch.load(args.weights)
    
    model_restoration.load_state_dict(checkpoint['params_ema'])
    print("===>Testing using weights: ", args.weights)

    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)

    model_restoration.eval()

    datasets = ['GoPro']
    betas = get_beta_schedule(
        beta_schedule=opt['diffusion'].get('beta_schedule'),
        beta_start=opt['diffusion'].get('beta_start'),
        beta_end=opt['diffusion'].get('beta_end'),
        num_diffusion_timesteps=opt['diffusion'].get('num_diffusion_timesteps')
    )
    betas = torch.from_numpy(betas).float().to(device)

    skip = opt['diffusion'].get('num_diffusion_timesteps') // opt['diffusion'].get('sampling_timesteps')
    seq = range(0, opt['diffusion'].get('num_diffusion_timesteps'), skip)
    eta = 0.
    seq_next = [-1] + list(seq[:-1])


    for dataset in datasets:
        result_dir = os.path.join(args.result_dir, dataset)
        os.makedirs(result_dir, exist_ok=True)

        # inp_dir = os.path.join(args.input_dir, dataset, 'input')
        inp_dir = os.path.join(args.input_dir)
        files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
        # print("===>222:", files)
        with torch.no_grad():
            for file_ in tqdm(files):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                img = np.float32(utils.load_img(file_)) / 255.
                img = torch.from_numpy(img).permute(2, 0, 1)
                input_ = img.unsqueeze(0).cuda()

                rough_preds = model_rough_g(input_)
              
                xs, _ = generalized_steps_overlapping(input_, rough_preds, model_restoration, device, betas, seq, seq_next, eta)
                restored = inverse_data_transform(xs[-1])

                restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),
                               img_as_ubyte(restored))






