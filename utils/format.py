import numpy
from scipy.linalg import hadamard
import torch
import torch_ac
import gymnasium as gym

def get_lock_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_lock_images(obss, device=device)
            })

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_lock_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    rotation_matrix = hadamard(numpy.array(images).shape[1])
    images = numpy.matmul(numpy.array(images), rotation_matrix)
    return torch.tensor(images, device=device, dtype=torch.float)
