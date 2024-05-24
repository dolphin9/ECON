from lib.kornia.axis2matrix import axis_angle_to_rotation_matrix
import torch
import numpy as np


def loadnpz(npz_path):
    npz_data = np.load(npz_path,allow_pickle=True)
    if type(npz_data) == np.lib.npyio.NpzFile:
        param = {key: torch.as_tensor(npz_data[key]) for key in npz_data.keys()}
    elif type(npz_data) == np.ndarray:
        param = npz_data.tolist()

    for item in param:
        param[item] = torch.as_tensor(param[item])
        if param[item].shape[1] == 3:
            param[item] = axis_angle_to_rotation_matrix(param[item]).unsqueeze(0)

    return param