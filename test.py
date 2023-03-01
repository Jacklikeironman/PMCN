import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader import Trainval_Dataset
import imageio 
from dehazemodels.PMCN import PMCN
from metrics import *
from utils import *

class Model:
    def __init__(self, fog_intensity=0.01):
        self.info_top = {}

        # Hyper-parameters:
        self.fog_intensity = fog_intensity
        self.model_name = 'PMCN' 
        self.frame_num = 7

        self.info_top['model name'] = self.model_name
        self.info_top['fog intensity'] = self.fog_intensity
        self.info_top['Input frame num'] = self.frame_num

        print('\n\n********** Model info **********')
        print_info([self.info_top])
        print('********** *********** **********')

    def checkpoint(self, model, epoch, optimizer, backup_dir):
        checkpoint_name = 'epoch_%.4d_.pth' % (epoch)
        state = {'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, backup_dir + '/' + checkpoint_name)
        print("Checkpoint saved to {}".format(checkpoint_name)) 

    
    def test(self, need_output=False):

        ckpt = 'checkpoints/model_{}.pth'.format(str(self.fog_intensity))

        log_dir = os.path.join('./log_{}'.format(str(self.fog_intensity)), self.model_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        f = open(os.path.join(log_dir, 'testlog.txt'),'a')

        result_dir = os.path.join('./result_{}'.format(str(self.fog_intensity)), self.model_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        print('\n')
        print('===> Loading datasets')
        val_set = Trainval_Dataset(patch_size=0, fog_intensity=self.fog_intensity, mode='val')
        validation_data_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
        print('===> Done\n')

        print('===> Building model ')
        model = PMCN(64)
        calculate_variables(model, print_vars=False)
        model = model.cuda()
        l2_loss_fn = nn.MSELoss(reduction='mean').cuda()
        SSIM_loss_fn = SSIM(window_size = 11).cuda()
        print('===> Done\n')

        print('===> Initialize and prepare...')
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state'])
        print('===> Done\n')

        # info:
        self.info_evaluate = {}
        self.info_evaluate['checkpoint'] = ckpt
        print('\n\n********** evaluate **********')
        print_info([self.info_evaluate])
        print('********** ******** **********')

        with torch.no_grad():
            model.eval()
            total_left_psnr_before = 0.0
            total_left_psnr_after = 0.0
            left_psnr_gain = 0.0
            total_left_ssim_before = 0.0
            total_left_ssim_after = 0.0
            left_ssim_gain = 0.0

            total_right_psnr_before = 0.0
            total_right_psnr_after = 0.0
            right_psnr_gain = 0.0
            total_right_ssim_before = 0.0
            total_right_ssim_after = 0.0
            right_ssim_gain = 0.0

            for n_count, batch in enumerate(validation_data_loader):
                torch.cuda.empty_cache()
                label_left_patches, label_right_patches, fog_left_patches, fog_right_patches = batch[0], batch[1], batch[2], batch[3]
                for i in range(self.frame_num):
                    left_label = label_left_patches[:, i, :, :, :].cuda()
                    right_label = label_right_patches[:, i, :, :, :].cuda()
                    left_input = fog_left_patches[:, i, :, :, :].cuda()
                    right_input = fog_right_patches[:, i, :, :, :].cuda()
                    B, _, H, W = left_label.shape
                    if i == 0:
                        pre_left_feat_pim = torch.zeros([B, 64, H // 2, W // 2], requires_grad=False).cuda()
                        pre_right_feat_pim = torch.zeros([B, 64, H // 2, W // 2], requires_grad=False).cuda()
                        pre_left_feat_pim_down1 = torch.zeros([B, 64, H // 4, W // 4], requires_grad=False).cuda()
                        pre_right_feat_pim_down1 = torch.zeros([B, 64, H // 4, W // 4], requires_grad=False).cuda()
                        pre_left_feat_pim_down2 = torch.zeros([B, 64, H // 8, W // 8], requires_grad=False).cuda()
                        pre_right_feat_pim_down2 = torch.zeros([B, 64, H // 8, W // 8], requires_grad=False).cuda()
                        pre_left_feat_pim_up1 = torch.zeros([B, 64, H // 4, W // 4], requires_grad=False).cuda()
                        pre_right_feat_pim_up1 = torch.zeros([B, 64, H // 4, W // 4], requires_grad=False).cuda()

                    batch_output = model(left_input, 
                                        right_input, 
                                        pre_left_feat_pim, 
                                        pre_right_feat_pim, 
                                        pre_left_feat_pim_down1, 
                                        pre_right_feat_pim_down1, 
                                        pre_left_feat_pim_down2, 
                                        pre_right_feat_pim_down2, 
                                        pre_left_feat_pim_up1, 
                                        pre_right_feat_pim_up1)

                    left_mse_loss = l2_loss_fn(batch_output['dehazed_left'], left_label)
                    left_mse_loss_before = l2_loss_fn(left_input, left_label)
                    left_ssim = SSIM_loss_fn(batch_output['dehazed_left'], left_label)
                    left_ssim_before = SSIM_loss_fn(left_input, left_label)

                    right_mse_loss = l2_loss_fn(batch_output['dehazed_right'], right_label)
                    right_mse_loss_before = l2_loss_fn(right_input, right_label)
                    right_ssim = SSIM_loss_fn(batch_output['dehazed_right'], right_label)
                    right_ssim_before = SSIM_loss_fn(right_input, right_label)

                    left_psnr_before = np.multiply(10.0, np.log(1.0 * 1.0 / left_mse_loss_before.cpu()) / np.log(10.0))
                    left_psnr = np.multiply(10.0, np.log(1.0 * 1.0 / left_mse_loss.cpu()) / np.log(10.0))
                    left_psnr_gain += (left_psnr - left_psnr_before)
                    left_ssim_gain += (left_ssim.cpu() - left_ssim_before.cpu())
                    total_left_psnr_before += left_psnr_before
                    total_left_psnr_after += left_psnr
                    total_left_ssim_before += left_ssim_before
                    total_left_ssim_after += left_ssim

                    right_psnr_before = np.multiply(10.0, np.log(1.0 * 1.0 / right_mse_loss_before.cpu()) / np.log(10.0))
                    right_psnr = np.multiply(10.0, np.log(1.0 * 1.0 / right_mse_loss.cpu()) / np.log(10.0))
                    right_psnr_gain += (right_psnr - right_psnr_before)
                    right_ssim_gain += (right_ssim.cpu() - right_ssim_before.cpu())
                    total_right_psnr_before += right_psnr_before
                    total_right_psnr_after += right_psnr
                    total_right_ssim_before += right_ssim_before
                    total_right_ssim_after += right_ssim

                    pre_left_feat_pim = batch_output['left_feat_pim'].detach()
                    pre_right_feat_pim = batch_output['right_feat_pim'].detach()
                    pre_left_feat_pim_down1 = batch_output['left_feat_pim_down1'].detach()
                    pre_right_feat_pim_down1 = batch_output['right_feat_pim_down1'].detach()
                    pre_left_feat_pim_down2 = batch_output['left_feat_pim_down2'].detach()
                    pre_right_feat_pim_down2 = batch_output['right_feat_pim_down2'].detach()
                    pre_left_feat_pim_up1 = batch_output['left_feat_pim_up1'].detach()
                    pre_right_feat_pim_up1 = batch_output['right_feat_pim_up1'].detach()

                    log = 'Frame%4d/%4d   left_PSNR_before: %.4f  left_PSNR_after: %.4f  left_PSNR_gain: %.4f left_SSIM_before: %.4f left_SSIM_after: %.4f left_SSIM_gain: %.4f right_PSNR_before: %.4f  right_PSNR_after: %.4f  right_PSNR_gain: %.4f right_SSIM_before: %.4f right_SSIM_after: %.4f right_SSIM_gain: %.4f' % ((n_count * self.frame_num + i + 1), len(validation_data_loader) * self.frame_num, left_psnr_before, left_psnr, left_psnr-left_psnr_before, left_ssim_before.cpu(), left_ssim.cpu(), left_ssim.cpu()-left_ssim_before.cpu(), right_psnr_before, right_psnr, right_psnr-right_psnr_before, right_ssim_before.cpu(), right_ssim.cpu(), right_ssim.cpu()-right_ssim_before.cpu())
                    print(log)
                    f.write(log + '\n')

                    if need_output:
                        left_label_dir = os.path.join(result_dir, 'left_label')
                        if not os.path.exists(left_label_dir):
                            os.makedirs(left_label_dir)

                        left_input_dir = os.path.join(result_dir, 'left_input')
                        if not os.path.exists(left_input_dir):
                            os.makedirs(left_input_dir)

                        left_output_dir = os.path.join(result_dir, 'left_output')
                        if not os.path.exists(left_output_dir):
                            os.makedirs(left_output_dir)

                        right_label_dir = os.path.join(result_dir, 'right_label')
                        if not os.path.exists(right_label_dir):
                            os.makedirs(right_label_dir)

                        right_input_dir = os.path.join(result_dir, 'right_input')
                        if not os.path.exists(right_input_dir):
                            os.makedirs(right_input_dir)

                        right_output_dir = os.path.join(result_dir, 'right_output')
                        if not os.path.exists(right_output_dir):
                            os.makedirs(right_output_dir)

                        left_label_path = os.path.join(left_label_dir, 'f_{}.png'.format(str(n_count*7+1).zfill(4)))
                        left_input_path = os.path.join(left_input_dir, 'f_{}.png'.format(str(n_count*7+1).zfill(4)))
                        left_output_path = os.path.join(left_output_dir, 'f_{}.png'.format(str(n_count*7+1).zfill(4)))
                        right_label_path = os.path.join(right_label_dir, 'f_{}.png'.format(str(n_count*7+1).zfill(4)))
                        right_input_path = os.path.join(right_input_dir, 'f_{}.png'.format(str(n_count*7+1).zfill(4)))
                        right_output_path = os.path.join(right_output_dir, 'f_{}.png'.format(str(n_count*7+1).zfill(4)))

                        left_label = left_label * 255.0
                        left_input = left_input * 255.0
                        left_output = batch_output['dehazed_left'] * 255.0
                        right_label = right_label * 255.0
                        right_input = right_input * 255.0
                        right_output = batch_output['dehazed_right'] * 255.0

                        left_label = left_label.round().clamp(0, 255).squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)   
                        left_input = left_input.round().clamp(0, 255).squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                        left_output = left_output.round().clamp(0, 255).squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                        right_label = right_label.round().clamp(0, 255).squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)   
                        right_input = right_input.round().clamp(0, 255).squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                        right_output = right_output.round().clamp(0, 255).squeeze().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

                        imageio.imwrite(left_label_path, left_label)
                        imageio.imwrite(left_input_path, left_input)
                        imageio.imwrite(left_output_path, left_output)
                        imageio.imwrite(right_label_path, right_label)
                        imageio.imwrite(right_input_path, right_input)
                        imageio.imwrite(right_output_path, right_output)

            print('*' * 50)
            print('Average Left PSNR before Restoring : %.4f' % (total_left_psnr_before / (len(validation_data_loader) * self.frame_num)))
            print('Average Left PSNR after Restoring : %.4f' % (total_left_psnr_after / (len(validation_data_loader) * self.frame_num)))
            print('Average Left PSNR Gain : %.4f' % (left_psnr_gain / (len(validation_data_loader) * self.frame_num)))
            print('Average Left SSIM before Restoring : %.4f' % (total_left_ssim_before / (len(validation_data_loader) * self.frame_num)))
            print('Average Left SSIM after Restoring : %.4f' % (total_left_ssim_after / (len(validation_data_loader) * self.frame_num)))
            print('Average Left SSIM Gain : %.4f' % (left_ssim_gain / (len(validation_data_loader) * self.frame_num)))
            print('*' * 50)
            print('Average Right PSNR before Restoring : %.4f' % (total_right_psnr_before / (len(validation_data_loader) * self.frame_num)))
            print('Average Right PSNR after Restoring : %.4f' % (total_right_psnr_after / (len(validation_data_loader) * self.frame_num)))
            print('Average Right PSNR Gain : %.4f' % (right_psnr_gain / (len(validation_data_loader) * self.frame_num)))
            print('Average Right SSIM before Restoring : %.4f' % (total_right_ssim_before / (len(validation_data_loader) * self.frame_num)))
            print('Average Right SSIM after Restoring : %.4f' % (total_right_ssim_after / (len(validation_data_loader) * self.frame_num)))
            print('Average Right SSIM Gain : %.4f' % (right_ssim_gain / (len(validation_data_loader) * self.frame_num)))
            f.write('*' * 50 + '\n')
            f.write('Average Left PSNR before Restoring : %.4f' % (total_left_psnr_before / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Left PSNR after Restoring : %.4f' % (total_left_psnr_after / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Left PSNR Gain : %.4f' % (left_psnr_gain / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Left SSIM before Restoring : %.4f' % (total_left_ssim_before / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Left SSIM after Restoring : %.4f' % (total_left_ssim_after / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Left SSIM Gain : %.4f' % (left_ssim_gain / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('*' * 50 + '\n')
            f.write('Average Right PSNR before Restoring : %.4f' % (total_right_psnr_before / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Right PSNR after Restoring : %.4f' % (total_right_psnr_after / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Right PSNR Gain : %.4f' % (right_psnr_gain / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Right SSIM before Restoring : %.4f' % (total_right_ssim_before / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Right SSIM after Restoring : %.4f' % (total_right_ssim_after / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.write('Average Right SSIM Gain : %.4f' % (right_ssim_gain / (len(validation_data_loader) * self.frame_num)) + '\n')
            f.close()


            

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--GPU', default='0', help='the GPU No. to use')
    subparsers = parser.add_subparsers(dest='mode')

    # evaluate mode:
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate the net. Type "python run.py evaluate -h" for more information.')
    parser_evaluate.add_argument('--fi', default=0.01, type=float, choices=[0.005, 0.01, 0.02], help='Fog intensity (default 0.001)')
    parser_evaluate.add_argument('-o', '--output', action='store_true', default=False, help='whether to output the results')
    
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        exit()
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    print('\n** GPU selection:', os.environ["CUDA_VISIBLE_DEVICES"])
    
    model = Model(fog_intensity=args.fi)
    model.test(need_output=args.output)
