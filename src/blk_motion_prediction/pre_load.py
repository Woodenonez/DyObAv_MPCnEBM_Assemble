import os, sys
import time
import pickle
from datetime import datetime
from typing import List

import torch
import matplotlib.pyplot as plt

from network_manager import NetworkManager
from blk_motion_prediction._data_handle_mmp import data_handler as dh
from blk_motion_prediction._data_handle_mmp import dataset as ds

from blk_motion_prediction.util_mp import utils_yaml
from blk_motion_prediction.util_mp import utils_np

import numpy as np


__PRT_NAME = '[PRE]'

def check_device():
    '''Check if GPUs are available. Record the current time.
    '''
    if torch.cuda.is_available():
        print('GPU count:', torch.cuda.device_count(),
              'First GPU:', torch.cuda.current_device(), torch.cuda.get_device_name(0))
    else:
        print(f'CUDA not working! Pytorch: {torch.__version__}.')
        sys.exit(0)
    torch.cuda.empty_cache()
    print(f'{__PRT_NAME} Pre-check at {datetime.now().strftime("%H:%M:%S, %D")}')

def load_config_fname(dataset_name:str, pred_range:tuple, mode:str):
    return f'{dataset_name.lower()}_{pred_range[0]}t{pred_range[1]}_{mode.lower()}.yml'

def load_param(root_dir, config_file, param_in_list=True, verbose=True):
    if param_in_list:
        param_list:List[dict] = utils_yaml.from_yaml_all(os.path.join(root_dir, 'config/', config_file), vb=verbose)
        return {k:v for x in param_list for k,v in x.items()}
    else:
        return utils_yaml.from_yaml(os.path.join(root_dir, 'config/', config_file), vb=verbose)

def load_path(param, root_dir):
    save_path = None # to save the model
    if param['model_path'] is not None:
        save_path = os.path.join(root_dir, param['model_path'])
    csv_path  = os.path.join(root_dir, param['label_path'])
    data_dir  = os.path.join(root_dir, param['data_path'])
    return save_path, csv_path, data_dir

def load_data(param, paths, transform, num_workers=0, T_range=None, ref_image_name=None, image_ext='png'):
    myDS = ds.ImageStackDataset(csv_path=paths[1], root_dir=paths[2], transform=transform,
                                pred_offset_range=T_range, ref_image_name=ref_image_name, image_ext=image_ext)
    myDH = dh.DataHandler(myDS, batch_size=param['batch_size'], num_workers=num_workers)
    print(f'{__PRT_NAME} Data prepared. #Samples(training, val):{myDH.get_num_data()}, #Batches:{myDH.get_num_batch()}')
    print(f'{__PRT_NAME} Sample (shape): \'image\':',myDS[0]['input'].shape,'\'label\':',myDS[0]['target'].shape)
    return myDS, myDH

def load_manager(param, Net:torch.nn.Module, loss:dict, encoder_channels=None, decoder_channels=None, verbose=True):
    if (encoder_channels is not None) & (decoder_channels is not None):
        net = Net(param['input_channel'], num_classes=param['pred_len'], 
                  en_channels=encoder_channels, de_channels=decoder_channels, out_layer=None)
        # en_chs = [16, 32,  64,  128, 256] # XXX
        # de_chs = en_chs[::-1]             # XXX
    else:
        net = Net(param['input_channel'], num_classes=param['pred_len'], ) # in, out channels
    myNet = NetworkManager(net, loss, training_parameter=param, device=param['device'], verbose=verbose)
    myNet.build_Network()
    return myNet

def save_profile(manager:NetworkManager, save_path:str='./'): # NOTE optional
    dt = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    manager.plot_history_loss()
    plt.savefig(os.path.join(save_path, dt+'.png'), bbox_inches='tight')
    plt.close()
    loss_dict = {'loss':manager.Loss, 'val_loss':manager.Val_loss}
    with open(os.path.join(save_path, dt+'.pickle'), 'wb') as pf:
        pickle.dump(loss_dict, pf)

def main_train(root_dir, config_file, transform, Net:torch.nn.Module, loss:dict, num_workers:int, batch_size:int=None, 
               T_range:tuple=None, ref_image_name:str=None, image_ext='png', runon='LOCAL'):
    ### Check and load
    check_device()
    param = load_param(root_dir, config_file)
    if batch_size is not None:
        param['batch_size'] = batch_size # replace the batch_size

    print(f'{__PRT_NAME} Model - {param["model_path"]}')
        
    paths = load_path(param, root_dir)
    _, myDH = load_data(param, paths, transform, num_workers, T_range, ref_image_name, image_ext)
    myNet = load_manager(param, Net, loss)

    ### Training
    start_time = time.time()
    myNet.train(myDH, myDH, param['batch_size'], param['epoch'], runon=runon)
    total_time = round((time.time()-start_time)/3600, 4)
    if (paths[0] is not None) & myNet.complete:
        torch.save(myNet.model.state_dict(), paths[0])
    nparams = sum(p.numel() for p in myNet.model.parameters() if p.requires_grad)
    print(f'\n{__PRT_NAME} Training done: {nparams} parameters. Cost time: {total_time}h.')

    save_profile(myNet)

def load_net(root_dir, config_file, Net):
    param = load_param(root_dir, config_file)
    paths = load_path(param, root_dir)
    myNet = load_manager(param, Net, {})
    myNet.build_Network()
    myNet.model.load_state_dict(torch.load(paths[0]))
    myNet.model.eval() # with BN layer, must run eval first
    return myNet

def main_test_pre(root_dir, config_file, transform, Net:torch.nn.Module, ref_image_name:str=None, verbose=False):
    ### Check and load
    param = load_param(root_dir, config_file)
    paths = load_path(param, root_dir)
    myDS, myDH = load_data(param, paths, transform, ref_image_name=ref_image_name)
    if Net is not None:
        myNet = load_manager(param, Net, {})
        myNet.model.load_state_dict(torch.load(paths[0]))
        myNet.model.eval() # with BN layer, must run eval first
    else:
        myNet = None
    return myDS, myDH, myNet

def main_test(dataset:ds.ImageStackDataset, net_manager:NetworkManager, idx:int):
    img:torch.Tensor   = dataset[idx]['input']  # originally np.ndarray
    label:torch.Tensor = dataset[idx]['target'] # originally np.ndarray
    traj  = dataset[idx]['traj']
    index = dataset[idx]['index']
    pred = net_manager.inference(img.unsqueeze(0))
    traj = torch.tensor(traj)
    try:
        ref = dataset[idx]['ref']
    except:
        ref = img[-1,:,:]
    return img, label, traj, index, pred.cpu(), ref

def traj_to_input(traj:list, ref_image:np.array, transform=None, obsv_len:int=5) -> torch.Tensor:
    '''Convert trajectory array to network input
    '''
    if len(traj)<obsv_len:
        traj = traj + [traj[-1]]*(obsv_len-len(traj))
    traj = traj[-obsv_len:]

    input_img = np.empty(shape=[ref_image.shape[0],ref_image.shape[1],0])
    for position in traj:
        this_x, this_y = position[0], position[1]

        obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(ref_image), sigmas=[20,20])
        obj_map = utils_np.np_dist_map((this_x, this_y), np.zeros_like(ref_image))
        input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)
    input_img = np.concatenate((input_img, ref_image[:,:,np.newaxis]), axis=2)
    input_img = np.concatenate((input_img, ref_image[:,:,np.newaxis]), axis=2) # placeholder for the pred_offset channel

    sample = {'image':input_img, 'label':0}
    if transform:
        sample = transform(sample)
    return sample['image']


