import os
import sys
import pathlib
from typing import List

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from blk_motion_prediction.pkg_net_module.net import UNetPos
from blk_motion_prediction._data_handle_mmp import data_handler as dh
from blk_motion_prediction._data_handle_mmp import dataset as ds

from blk_motion_prediction import pre_load
from blk_motion_prediction.util_mp import utils_test
from blk_motion_prediction.util_mp import zfilter
from blk_motion_prediction.util_mp import utils_np

from sklearn.mixture import GaussianMixture

print("Program: testing...\n")
pre_load.check_device()

### Test option
TEST_MODE = 3 # 1-Basic test, 2-Overall test, 3-CGF test
PRED_OFFSET = 20 # for Kalman filter

### Config
# config_file = 'sdd_1t12_train.yml'
# ref_image_name = 'label.png'
# config_file = 'gcd_1t20_train.yml'
# ref_image_name = None
# config_file = 'sidv2x_1t10_train.yml'
# ref_image_name = None
config_file = 'ald_1t20_train.yml'
ref_image_name = 'label.png'

root_dir = pathlib.Path(__file__).resolve().parents[1]
param = pre_load.load_param(root_dir, config_file, verbose=False)

composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = UNetPos

### Prepare
dataset, _, net_manager = pre_load.main_test_pre(root_dir, config_file, composed, Net, ref_image_name=ref_image_name)

if TEST_MODE==3:
    plot_row = 2
    plot_col = 2
    simple_plot = False
else:
    plot_row = 1
    plot_col = 3
    simple_plot = True

### Visualization option
idx_start = 8500
idx_end = len(dataset)
pause_time = 0

### Visualize
if plot_row == 1:
    fig, axes = plt.subplots(1, plot_col)
else:
    fig, axes = plt.subplots(plot_row, plot_col)
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    print(f'\r{idx}', end='')

    [ax.cla() for ax in axes.ravel()]
    
    img, label, traj, index, pred, ref = pre_load.main_test(dataset, net_manager, idx=idx)
    e_grid = net_manager.to_energy_grid(pred)
    prob_map = net_manager.to_prob_map(pred)

    ### Kalman filter
    # P0 = zfilter.fill_diag((1,1,1,1))
    # Q  = zfilter.fill_diag((0.1,0.1,0.1,0.1))
    # R  = zfilter.fill_diag((0.1,0.1))
    # KF_X, KF_P = utils_test.fit_KF(zfilter.model_CV, traj, P0, Q, R, PRED_OFFSET)

    if simple_plot:
        if TEST_MODE == 1:
            axes[0].imshow(ref, cmap='gray')
            axes[0].plot(traj[:-1,0], traj[:-1,1], 'm.', label='past')
            axes[0].plot(label[:,0], label[:,1], 'r-', linewidth=4, label="ground truth")
            axes[0].legend(prop={'size': 14}, loc='best')
            axes[0].set_aspect('equal', 'box')
            axes[0].set_title('Real world')

            axes[1].imshow(e_grid[0,-1,:], cmap='gray')
            axes[1].plot(label[-1,0], label[-1,1], 'rx')
            axes[1].set_title('Last energy grid')

            axes[2].imshow(prob_map[0,-1,:], cmap='gray')
            axes[2].plot(label[-1,0], label[-1,1], 'rx')
            axes[2].set_title('Last probability map')
        elif TEST_MODE == 2:
            axes[0].imshow(ref, cmap='gray')
            axes[0].plot(traj[:-1,0], traj[:-1,1], 'm.', label='past')
            axes[0].plot(label[:,0], label[:,1], 'r-', linewidth=4, label="ground truth")
            axes[0].legend(prop={'size': 14}, loc='best')
            axes[0].set_aspect('equal', 'box')
            axes[0].set_title('Real world')

            axes[1].imshow(torch.sum(e_grid[0,:], dim=0), cmap='gray')
            axes[1].plot(label[:,0], label[:,1], 'r.')
            axes[1].set_title('Energy grid')

            axes[2].imshow(torch.clamp(torch.sum(prob_map[0,:], dim=0), max=1) + ref/2, cmap='gray')
            axes[2].plot(label[:,0], label[:,1], 'r.')
            axes[2].set_title('Probability map')
    else:
        mu_list_list = []
        std_list_list = []
        conf_list_list = []
        gmm_list:List[GaussianMixture] = []
        traj_samples = net_manager.gen_samples(prob_map, num_samples=500, replacement=True)
        for i in range(prob_map.shape[1]):
            # goal_samples = utils_func.ynet_TTST(prob_map[:,i,:,:].unsqueeze(1), 20)
            # clusters = utils_test.fit_DBSCAN(goal_samples[:,0,0,:].numpy(), eps=5, min_sample=3)

            # prob_x, prob_y = traj_samples[0,i,:,0], traj_samples[0,i,:,1]
            # confidence = prob_map[0, i, prob_y.long(), prob_x.long()]

            clusters = net_manager.fit_DBSCAN(traj_samples[0,i,:].numpy(), eps=10, min_sample=5)

            if len(clusters) > 0:
                gmm = GaussianMixture(n_components=len(clusters)).fit(traj_samples[0,i,:])
                gmm_list.append(gmm)

            mu_list, std_list = net_manager.fit_cluster2gaussian(clusters)

            conf_list = []
            for mu in mu_list:
                conf_list.append(prob_map[0, i, int(mu[1]), int(mu[0])].item() + 1e-9)
            conf_list = [round(x/sum(conf_list),2) for x in conf_list]

            mu_list_list.append(mu_list)
            std_list_list.append(std_list)
            conf_list_list.append(conf_list)

        alpha_list = np.linspace(start=1, stop=0.5, num=len(mu_list_list)).tolist()

        utils_test.plot_on_ref([axes[0,0], axes[1,0], axes[1,1]], ref, traj, label,
                                e_grid[0,-1], prob_map[0,-1], traj_samples)

        for gmm in gmm_list:
            axes[1][0].plot(gmm.means_[:,0], gmm.means_[:,1], 'yo')
            [axes[1][0].text(gmm.means_[j,0], gmm.means_[j,1], round(gmm.weights_[j],2)) for j in range(len(gmm.weights_))]

        axes[0][1].imshow(ref, cmap='gray')
        axes[0][1].plot(traj[:-1,0], traj[:-1,1], 'm.', label='past') # past
        axes[0][1].plot(label[:,0], label[:,1], 'r-', label="ground truth")
        for mu_list, std_list, conf_list, alpha in zip(mu_list_list, std_list_list, conf_list_list, alpha_list):
            utils_test.plot_Gaussian_ellipses(axes[0][1], mu_list, std_list, conf_list, factor=3, expand=5, alpha=alpha)

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    if pause_time==0:
        plt.pause(0.01)
        while not plt.waitforbuttonpress():  # XXX press a button to continue
            pass
    else:
        plt.pause(pause_time)

plt.show()

