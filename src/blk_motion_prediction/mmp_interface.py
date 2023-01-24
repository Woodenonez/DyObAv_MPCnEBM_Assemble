from typing import List

import numpy as np
import torch
import torchvision

import blk_motion_prediction.pre_load as pre_load
from blk_motion_prediction.util_mp import utils_np
from blk_motion_prediction._data_handle_mmp import data_handler
from blk_motion_prediction.pkg_net_module.net import UNetPos

class MmpInterface:
    def __init__(self, root_dir, config_file_name:str):
        self.net = pre_load.load_net(root_dir, config_file_name, Net=UNetPos)

    def get_motion_prediction(self, input_traj:List[tuple], ref_image:np.ndarray, pred_offset:int, rescale:float=1.0, batch_size:int=1):
        if input_traj is None:
            return None
            
        input_traj = [[x*rescale for x in y] for y in input_traj]
        transform = torchvision.transforms.Compose([data_handler.ToTensor()])
        if isinstance(ref_image, np.ndarray):
            ref_image = torch.from_numpy(ref_image.transpose((2, 0, 1)))
        elif not isinstance(ref_image, torch.Tensor):
            raise TypeError(f'The reference image should be np.ndarray, got {type(ref_image)}.')
        input_ = pre_load.traj_to_input(input_traj, ref_image=ref_image, transform=transform)
        
        hypos_list:List[np.ndarray] = []
        # XXX Batch inference
        input_all = input_.unsqueeze(0)
        for offset in range(1, pred_offset+1):
            input_[-1,:,:] = offset*torch.ones_like(input_[-1,:,:])
            input_all = torch.cat((input_all, input_.unsqueeze(0)), dim=0)
        input_all = input_all[1:]
        for i in range(pred_offset//batch_size):
            input_batch = input_all[batch_size*i:batch_size*(i+1), :]
            try:
                hyposM = np.concatenate((hyposM, self.net.inference(input_batch)), axis=0)
            except:
                hyposM = self.net.inference(input_batch)
        if pred_offset%batch_size > 0:
            input_batch = input_all[batch_size*(pred_offset//batch_size):, :]
            hyposM = np.concatenate((hyposM, self.net.inference(input_batch)), axis=0)
        for i in range(pred_offset):
            hypos_list.append(utils_np.get_closest_edge_point(hyposM[i,:], 255 - ref_image.numpy()) / rescale)
        # XXX

        # for offset in range(1, pred_offset+1):
        #     input_[-1,:,:] = offset*torch.ones_like(input_[-1,:,:])

        #     hyposM = self.net.inference(input_.unsqueeze(0))[0,:]

        #     hyposM = utils_np.get_closest_edge_point(hyposM, 255 - ref_image.numpy()) # post-processing
        #     hypos_list.append(hyposM/rescale)
        return hypos_list