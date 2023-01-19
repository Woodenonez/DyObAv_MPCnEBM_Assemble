import os
from pathlib import Path

from blk_motion_prediction._data_handle_mmp.dataset import io, ImageStackDataset

from blk_util import utils_sl
from blk_util.basic_datatype import *

def prepare_map(scene:str, root_dir:FolderDir, inversed_pixel:bool=False):
    scene_dir_name = scene.lower()+'_sim_original'
    ref_map_path = os.path.join(root_dir, 'data', scene_dir_name, 'label.png')
    ref_map = ImageStackDataset.togray(io.imread(ref_map_path))
    if scene.lower() == 'assemble':
        map_file_path = os.path.join(root_dir, 'data', scene_dir_name, 'drawing_map.json')
        the_map_dict = utils_sl.read_obj_from_json(map_file_path)[0]
    else:
        map_file_path = os.path.join(root_dir, 'data', scene_dir_name, 'mymap.pgm')
        the_map = utils_sl.read_pgm_and_process(map_file_path, inversed_pixel=inversed_pixel)
        
        if scene.lower() == 'bookstore':
            the_map[0:40, 950:980] = 0 # for the bookstore scene
    return the_map, ref_map

def prepare_params(config_file_name:str, root_dir:FolderDir):
    cfg_path = os.path.join(root_dir, 'config', config_file_name)
    param_dict = utils_sl.from_yaml(cfg_path)
    return param_dict
