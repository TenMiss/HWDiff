import torch
import torch.nn as nn
from basicsr.models.archs.network_module import *
# from models.network_module import *
# from models import dcn_module

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# ---------------------------------------
#       Base Model for network
# ---------------------------------------
class BaseModel(nn.Module):

    def __init__(self,):
        super(BaseModel, self).__init__()

    @staticmethod
    def load_config_from_json(file):
        pass

    def dump_config_to_json(self, file):
        pass

    def load_ckpt(self, model_path, force_load = False):
        state_dict = torch.load(model_path, map_location = torch.device('cpu'))

        if force_load:
            state_dict_temp = self.state_dict()
            key_temp = set(list(state_dict_temp.keys()))

            for n, p in state_dict.items():
                # temp code
                # if 'res_layer' in n:
                #     n = n.replace('res_layers.', 'res_block_')

                key_temp.remove(n)

                if n in state_dict_temp.keys():
                    if state_dict_temp[n].shape != p.data.shape:
                        print('%s size mismatch, pass!' % n)
                        continue
                    state_dict_temp[n].copy_(p.data)
                else:
                    print('%s not exist, pass!' % n)
            state_dict = state_dict_temp

            if len(key_temp) != 0:
                for k in key_temp:
                    print("param %s not found in state dict!" % k)

        self.load_state_dict(state_dict)
        print("Load checkpoint {} successfully!".format(model_path))


# ----------------------------------------
#                DeblurNet
# ----------------------------------------
class DeblurNet_v2(BaseModel):

    def __init__(self, in_channel, out_channel, activ, norm, pad_type, deblur_res_num, deblur_res_num2, final_activ, upsample_layer, shuffle_mode, ngf):
        super(DeblurNet_v2, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.activ = activ
        self.norm = norm
        self.pad_type = pad_type
        self.deblur_res_num = deblur_res_num
        self.deblur_res_num2 = deblur_res_num2
        self.final_activ = final_activ
        self.upsample_layer = upsample_layer
        self.shuffle_mode = shuffle_mode
        self.ngf = ngf

        self.build_layers()

    def build_upsample_layer(self, in_channel, out_channel, upsample_level = None):
        if self.upsample_layer == 'pixelshuffle':
            return PixelShuffleAlign(upscale_factor = 2, mode = self.shuffle_mode)
        elif self.upsample_layer == 'bilinear':
            return nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
                                 nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = 1))

    def build_layers(self):
        
        self.dwt = DWT()
        self.idwt = IWT()
        
        self.fusion_conv = Conv2dLayer(self.in_channel * 4 * 4,
                                       self.ngf * 4 * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                       activation = self.activ, norm = self.norm)

        self.downsample_conv = Conv2dLayer(self.in_channel  * 4,
                                         self.in_channel  * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)
        
        # deblur resblocks
        for i in range(self.deblur_res_num):
            in_channels = self.ngf * 4 * 4
            block = ResBlock(dim = in_channels,
                             kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type,
                             activation = self.activ, norm = self.norm)
            setattr(self, 'deblur_res_block_%d' % i, block)
        
        # upsample layer after deblur resblocks
        self.upsample_conv = Conv2dLayer(self.ngf * 4,
                                         self.ngf * 4, 3, stride = 1, padding = 1, pad_type = self.pad_type,
                                         activation = self.activ, norm = self.norm)

        self.deblur_layer = Conv2dLayer(self.ngf, 3, 3,
                                        stride = 1, padding = 1, pad_type = self.pad_type,
                                        activation = 'none', norm = 'none')

        # deblur resblock2
        for i in range(self.deblur_res_num2):
            in_channels = self.ngf
            block = ResBlock(dim = in_channels,
                             kernel_size = 3, stride = 1, padding = 1, pad_type = self.pad_type,
                             activation = self.activ, norm = self.norm)
            setattr(self, 'deblur_res_block2_%d' % i, block)
        
        if self.final_activ == 'tanh':
            self.final_activ = nn.Tanh()
        
    def forward(self, lq):
        sl = self.dwt(lq)
        sl = self.downsample_conv(sl)
        sl = self.dwt(sl)
        sl = self.fusion_conv(sl)

        deblur_sl = sl

        for i in range(self.deblur_res_num):
            resblock = getattr(self, 'deblur_res_block_%d' % i)
            deblur_sl = resblock(deblur_sl)

        sl = sl + deblur_sl

        sl = self.idwt(sl)
        sl = self.upsample_conv(sl)
        sl = self.idwt(sl)

        deblur_sl = sl
        for i in range(self.deblur_res_num2):
            resblock = getattr(self, 'deblur_res_block2_%d' % i)
            deblur_sl = resblock(deblur_sl)
        sl = deblur_sl + sl

        deblur_sl = self.deblur_layer(sl)
        deblur_out = lq + deblur_sl
        # if self.final_activ == 'tanh':
        #     deblur_out = self.final_activ(deblur_out)
        return deblur_out
    

if __name__ == "__main__":

    # import argparse
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--in_channel', type = int, default = 3, help = '')
    # parser.add_argument('--out_channel', type = int, default = 3, help = '')
    # parser.add_argument('--ngf', type = int, default = 64, help = '')
    # parser.add_argument('--ngf2', type = int, default = 8, help = '')
    # parser.add_argument('--activ', type = str, default = 'lrelu', help = '')
    # parser.add_argument('--norm', type = str, default = 'none', help = '')
    # parser.add_argument('--pad', type = str, default = 'zero', help = '')

    # parser.add_argument('--deblur_res_num', type = int, default = 8, help = '')
    # parser.add_argument('--deblur_res_num2', type = int, default = 4, help = '')
    # parser.add_argument('--denoise_res_num', type = int, default = 8, help = '')
    # parser.add_argument('--denoise_res_num2', type = int, default = 4, help = '')
    # parser.add_argument('--groups', type = int, default = 8, help = '')
    
    # parser.add_argument('--final_activ', type = str, default = 'none', help = '')
    # parser.add_argument('--pad_type', type = str, default = 'zero', help = '')
    # parser.add_argument('--upsample_layer', type = str, default = 'pixelshuffle', help = '')
    # parser.add_argument('--shuffle_mode', type = str, default = 'caffe', help = '')
    
    # opt = parser.parse_args()
    
    # a = torch.randn(1, 3, 512, 512).cuda()

    net = DeblurNet_v2(
        in_channel = 3, 
        out_channel = 3, 
        activ= 'lrelu', 
        norm= 'none', 
        ngf=64, 
        deblur_res_num= 3, 
        deblur_res_num2= 3, 
        final_activ= 'none', 
        pad_type= 'zero', 
        upsample_layer= 'pixelshuffle', 
        shuffle_mode= 'caffe')
    # out = net(a, a)

    #net = DenoiseNet_v2(opt).cuda()
    #out = net(a, a, a)

    # print(out.shape)
    #save_state_dict = net.state_dict()
    #save_path = 'GNet-epoch-1.pkl'
    #torch.save(save_state_dict, save_path
    
    # inp = torch.Tensor(1,3, 256,256)

    # inp_shape = (3, 256, 256)

    # from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    # print(macs, params)

    inp = torch.Tensor(1,3, 256,256)

    from thop import profile


    macs, params = profile(net, inputs=(inp, ))
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("---|---|---")
    print("%s | %.2f | %.2f" % ('xxxx', params / (1000 ** 2), macs / (1000 ** 3)))
