import torch
import torch.nn as nn
import numpy as np
# from basiclayers import *
from dehazemodels.basiclayers import *
from functools import reduce
from basicsr.archs.arch_util import DCNv2Pack

class SKN(nn.Module):
    '''
    Selective Kernel Module
    input:
        feature_1 : [b, c, h, w]
        feature_2 : [b, c, h, w]
    output:
        feature_att : [b, c, h, w]
    '''
    def __init__(self, M, n_feature, down_ratio):
        super(SKN, self).__init__() 
        self.n_feature = n_feature
        self.down_ratio = down_ratio
        self.M = M

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1=nn.Sequential(nn.Conv2d(self.n_feature,self.n_feature//self.down_ratio,1,bias=False),
                            nn.LeakyReLU(inplace=True))
        self.fc2=nn.Conv2d(n_feature//self.down_ratio,self.n_feature*self.M,1,1,bias=False)

        # self.fc3=nn.Sequential(nn.Conv2d(self.n_feature,self.n_feature//self.down_ratio,1,bias=False),
        #                     nn.LeakyReLU(inplace=True))
        # self.fc4=nn.Conv2d(n_feature//self.down_ratio,self.n_feature,1,1,bias=False)

        self.softmax=nn.Softmax(dim=1)

    def forward(self, input):
        assert self.M == len(input)
        batch_size=input[0].size(0)
        U=reduce(lambda x,y:x+y,input)
        s = self.global_pool(U)
        z=self.fc1(s)
        a_b=self.fc2(z)
        a_b=a_b.reshape(batch_size,self.M,self.n_feature,-1)
        a_b=self.softmax(a_b)
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b
        a_b=list(map(lambda x:x.reshape(batch_size,self.n_feature,1,1),a_b))
        V=list(map(lambda x,y:x*y,input,a_b))
        V=reduce(lambda x,y:x+y,V)
        return V

class Feature_Extraction(nn.Module):
    def __init__(self, in_feat, num_feat):
        super(Feature_Extraction, self).__init__()
        self.in_feat = in_feat
        self.num_feat = num_feat
        self.pixelunshuffle = PixelUnshuffle(downscale_factor=2)
        self.head_conv = nn.Sequential(nn.Conv2d(in_feat*4, num_feat, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=True))
        self.body = make_layer(ResidualBlockNoBN, 2, num_feat=self.num_feat)

    def forward(self, x):
        x = self.pixelunshuffle(x)
        h = self.head_conv(x)
        global_residual = h
        out = global_residual + self.body(h)
        return out

class Feature_Reconstruction(nn.Module):
    def __init__(self, out_feat, num_feat):
        super(Feature_Reconstruction, self).__init__()
        self.out_feat = out_feat
        self.num_feat = num_feat
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        self.end_conv = nn.Sequential(nn.Conv2d(self.num_feat//4, self.out_feat, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=True))
        self.body = make_layer(ResidualBlockNoBN, 4, num_feat=self.num_feat//4)

    def forward(self, x):
        h = self.pixelshuffle(x)
        out = self.end_conv(self.body(h))
        return out

class Parallax_Interaction_Module(nn.Module):
    def __init__(self, num_feat, num_window=4):
        super(Parallax_Interaction_Module, self).__init__()
        # long-range interact
        self.num_window = num_window
        self.embedding_left = nn.Conv2d(num_feat, num_feat//2, 1, 1, 0)
        self.embedding_right = nn.Conv2d(num_feat, num_feat//2, 1, 1, 0)
        self.embedding_left_V = nn.Conv2d(num_feat, num_feat//2, 1, 1, 0)
        self.embedding_right_V = nn.Conv2d(num_feat, num_feat//2, 1, 1, 0)
        self.outconv_left = nn.Conv2d(num_feat//2, num_feat, 1, 1, 0)
        self.outconv_right = nn.Conv2d(num_feat//2, num_feat, 1, 1, 0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

        # short-range interact
        self.head_conv = nn.Sequential(nn.Conv2d(num_feat*2, num_feat, 1, 1, 0), nn.ReLU())
        self.body = make_layer(ResidualBlockNoBN_1x9, num_feat=num_feat, is_dynamic=True, num_basic_block=3)

        # fuse
        self.sknet_1 = SKN(3, num_feat, down_ratio=4)
        self.sknet_2 = SKN(3, num_feat, down_ratio=4)
    
    def forward(self, left_feat, right_feat):
        # left_feat   [B, 2C, H, W]
        # right_feat  [B, 2C, H, W]
        # Light parallax attention module (long-range interact)
        Q = self.embedding_left(left_feat)
        K = self.embedding_right(right_feat)
        V_left = self.embedding_left_V(left_feat)
        V_right = self.embedding_right_V(right_feat) 
        B, C, H, W = Q.shape
        Q = Q.permute(0, 2, 3, 1).contiguous().view(B*H, W, -1)                             # B * H * W * C
        K = K.permute(0, 2, 1, 3).contiguous().view(B*H, -1, W)                             # B * H * C * W

        if self.num_window == 1:       
            score = torch.bmm(Q, K)                                                         # (B * H) * W * W
            score_T = score.permute(0,2,1)                                                  # (B * H) * W * W                                                   
            M_right_to_left = self.softmax(score)
            M_left_to_right = self.softmax(score_T)

            V_right = V_right.permute(0,2,3,1).contiguous().view(-1, W, C)                                        # (B*H) * W * C
            left_feat_warped = self.relu(self.outconv_left(torch.bmm(M_right_to_left, V_right).contiguous().view(B, H, W, C).permute(0,3,1,2))) # B * 2C * H * W 
            V_left = V_left.permute(0,2,3,1).contiguous().view(-1, W, C)                                          # (B*H) * W * C
            right_feat_warped = self.relu(self.outconv_left(torch.bmm(M_left_to_right, V_left).contiguous().view(B, H, W, C).permute(0,3,1,2))) # B * 2C * H * W
        else:
            self.window_size = W // self.num_window
            Q = Q.view(-1, self.window_size, C)                                                                                                 # (B * H * W // win) * Win * C
            K = K.view(-1, C, self.num_window, self.window_size).permute(0,2,1,3).contiguous().view(-1, C, self.window_size)                    # (B * H * W // win) * C * Win
            score = torch.bmm(Q, K)                                                                                                             # (B * H * W // win) * Win * Win
            score_T = score.permute(0,2,1)                                                                                                      # (B * H * W // win) * Win * Win
            M_right_to_left = self.softmax(score)
            M_left_to_right = self.softmax(score_T)

            V_right = V_right.view(B, C, H, self.num_window, self.window_size).permute(0, 2, 3, 4, 1).contiguous().view(-1, self.window_size, C)                      # (B * H * W // win) * Win * C 
            left_feat_warped = self.relu(self.outconv_left(torch.bmm(M_right_to_left, V_right).contiguous().view(B, H, self.num_window, self.window_size, C).permute(0, 4, 1, 2, 3).contiguous().view(B, C, H, W))) # B * 2C * H * W 
            V_left = V_left.view(B, C, H, self.num_window, self.window_size).permute(0, 2, 3, 4, 1).contiguous().view(-1, self.window_size, C)                        # (B * H * W // win) * Win * C
            right_feat_warped = self.relu(self.outconv_left(torch.bmm(M_right_to_left, V_right).contiguous().view(B, H, self.num_window, self.window_size, C).permute(0, 4, 1, 2, 3).contiguous().view(B, C, H, W)))# B * 2C * H * W
        # short interact
        # mixed_feat [B, 2C, H, W]
        mixed_feat = self.body(self.head_conv(torch.cat([left_feat, right_feat], dim=1)))

        # output
        # left_interacted_feat   [B, 2C, H, W]
        # right_interacted_feat  [B, 2C, H, W]
        left_interacted_feat = self.sknet_1([left_feat, left_feat_warped, mixed_feat])
        right_interacted_feat = self.sknet_2([right_feat, right_feat_warped, mixed_feat])

        return left_interacted_feat, right_interacted_feat

class Motion_Alignment_Module(nn.Module):
    def __init__(self, num_feat):
        super(Motion_Alignment_Module, self).__init__()
        self.num_feat = num_feat
        self.offset_generator = nn.Sequential(nn.Conv2d(num_feat*2, num_feat, 1, 1, 0), nn.ReLU())
        self.dcn = DCNv2Pack(num_feat, num_feat, 3, 1, 1)
        self.sknet = SKN(2, num_feat, 4) 
        self.body = make_layer(ResidualBlockNoBN, num_feat=num_feat, is_dynamic=True, num_basic_block=3)
    def forward(self, cur_feat, pre_feat):
        offset = self.offset_generator(torch.cat([cur_feat, pre_feat], dim=1))
        warped_feat = self.dcn(pre_feat, offset)
        fused_feat = self.sknet([warped_feat, cur_feat])
        aligned_feat = self.body(fused_feat)
        return aligned_feat

class PMCB(nn.Module):
    def __init__(self, num_feat):
        super(PMCB, self).__init__()
        self.num_feat = num_feat
        self.PIM = Parallax_Interaction_Module(num_feat=self.num_feat)
        self.MAM_left = Motion_Alignment_Module(num_feat=self.num_feat)
        self.MAM_right = Motion_Alignment_Module(num_feat=self.num_feat)
    
    def forward(self, left_feat, right_feat, pre_left_feat_pim, pre_right_feat_pim):
        left_feat_pim, right_feat_pim = self.PIM(left_feat, right_feat)
        left_feat_mam = self.MAM_left(left_feat_pim, pre_left_feat_pim)
        right_feat_mam = self.MAM_right(right_feat_pim, pre_right_feat_pim)
        return left_feat_mam, right_feat_mam, left_feat_pim, right_feat_pim

class PMCN(nn.Module):
    def __init__(self, num_feat=64):
        super().__init__()
        self.num_feat = num_feat
        
        self.feature_extraction = Feature_Extraction(in_feat=3, num_feat=self.num_feat)
        self.basicblock_1 = PMCB(num_feat=num_feat)
        self.basicblock_2 = PMCB(num_feat=num_feat)
        self.basicblock_3 = PMCB(num_feat=num_feat)
        self.basicblock_4 = PMCB(num_feat=num_feat)
        self.feature_reconstrution = Feature_Reconstruction(out_feat=3, num_feat=self.num_feat)

        self.downsample = nn.AvgPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)

        
        
    def forward(self, 
                left_foggy_img, 
                right_foggy_img, 
                pre_left_feat_pim, 
                pre_right_feat_pim,
                pre_left_feat_pim_down1,
                pre_right_feat_pim_down1,
                pre_left_feat_pim_down2,
                pre_right_feat_pim_down2,
                pre_left_feat_pim_up1,
                pre_right_feat_pim_up1
                ):
        # left [B, 3, H, W]
        # right [B, 3, H, W]

        # feature extraction
        
        left_feat = self.feature_extraction(left_foggy_img)                # left_feat [B, C, H//2, W//2]
        right_feat = self.feature_extraction(right_foggy_img)              # right_feat [B, C, H//2, W//2]

        # BasicBlock1
        l, r, left_feat_pim, right_feat_pim = self.basicblock_1(left_feat, right_feat, pre_left_feat_pim, pre_right_feat_pim)              # l [B, C, H//2, W//2]  r [B, C, H//2, W//2]
        left_feat_down1 = self.downsample(l)                         # left_feat_down1 [B, C, H//4, W//4]
        right_feat_down1 = self.downsample(r)                        # right_feat_down1 [B, C, H//4, W//4]

        # BasicBlock2
        l, r, left_feat_pim_down1, right_feat_pim_down1 = self.basicblock_2(left_feat_down1, right_feat_down1, pre_left_feat_pim_down1, pre_right_feat_pim_down1)  # l [B, C, H//4, W//4]  r [B, C, H//4, W//4]
        left_feat_down2 = self.downsample(l)                         # left_feat_down2 [B, C, H//8, W//8]
        right_feat_down2 = self.downsample(r)                        # right_feat_down2 [B, C, H//8, W//8]

        # BasicBlock3
        l, r, left_feat_pim_down2, right_feat_pim_down2 = self.basicblock_3(left_feat_down2, right_feat_down2, pre_left_feat_pim_down2, pre_right_feat_pim_down2)  # l [B, C, H//8, W//8]  r [B, C, H//8, W//8]
        left_feat_up1 = self.upsample(l + left_feat_down2)           # left_feat_up2 [B, C, H//4, W//4]
        right_feat_up1 = self.upsample(r + right_feat_down2)         # right_feat_up2 [B, C, H//4, W//4]

        # BasicBlock4
        l, r, left_feat_pim_up1, right_feat_pim_up1 = self.basicblock_4(left_feat_up1, right_feat_up1, pre_left_feat_pim_up1, pre_right_feat_pim_up1)  # l [B, C, H//4, W//4]  r [B, C, H//4, W//4]
        left_feat_ = self.upsample(l + left_feat_up1)           # left_feat_up2 [B, C, H//2, W//2]
        right_feat_ = self.upsample(r + right_feat_up1)         # right_feat_up2 [B, C, H//2, W//2]

        fuse_left_feat = left_feat + left_feat_
        fuse_right_feat = right_feat + right_feat_

        # feature reconstruction
        left_dehaze_img =self.feature_reconstrution(fuse_left_feat)
        right_dehaze_img =self.feature_reconstrution(fuse_right_feat)

        return {'dehazed_left': left_dehaze_img,
                'dehazed_right': right_dehaze_img,
                'left_feat_pim': left_feat_pim,
                'right_feat_pim': right_feat_pim,
                'left_feat_pim_down1': left_feat_pim_down1,
                'right_feat_pim_down1': right_feat_pim_down1,
                'left_feat_pim_down2': left_feat_pim_down2,
                'right_feat_pim_down2': right_feat_pim_down2,
                'left_feat_pim_up1': left_feat_pim_up1,
                'right_feat_pim_up1': right_feat_pim_up1}


if __name__ == '__main__':
    import time
    num_feat = 64
    patch_size = (512, 1024)

    net = PMCN(num_feat).cuda()
    a = torch.randn(1, 3, patch_size[0], patch_size[1]).cuda()
    b = torch.randn(1, num_feat, patch_size[0]//2, patch_size[1]//2).cuda()
    c = torch.randn(1, num_feat, patch_size[0]//4, patch_size[1]//4).cuda()
    d = torch.randn(1, num_feat, patch_size[0]//8, patch_size[1]//8).cuda()
    times_diff = []
    with torch.no_grad():
        for i in range(200):
            start_time = time.time()
            e = net(a,a,b,b,c,c,d,d,c,c)
            end_time = time.time()
            print(e['dehazed_left'].shape)
            time_diff = end_time - start_time
            times_diff.append(time_diff)
            print(time_diff * 1000)
        time_diff_avg = np.mean(np.array(times_diff))
        print('Total Time : %.6f ms' % (time_diff_avg*1000))
    # print(e['dehazed_left'].shape)
    # print(e['dehazed_right'].shape)