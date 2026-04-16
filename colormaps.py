import os

import cmocean
import numpy as np
from matplotlib.colors import ListedColormap


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_custom_colormap(filename):
    rgb_values = np.load(os.path.join(BASE_DIR, filename))
    return ListedColormap(rgb_values)


COLORMAP_CONFIGS = {
    "thermal": {
        "label": "cmocean thermal",
        "cmap": cmocean.cm.thermal,
        "image_source": "interpolated",
        "norm": None,
        "imshow_kwargs": {
            "interpolation": "nearest",
            "aspect": "auto",
        },
    },
    "oxy": {
        "label": "cmocean oxy",
        "cmap": cmocean.cm.oxy,
        "image_source": "interpolated",
        "norm": None,
        "imshow_kwargs": {
            "interpolation": "nearest",
            "aspect": "auto",
        },
    },
    "inferno": {
        "label": "matplotlib inferno",
        "cmap": "inferno",
        "image_source": "interpolated",
        "norm": None,
        "imshow_kwargs": {
            "interpolation": "nearest",
            "aspect": "auto",
        },
    },
    "flir": {
        "label": "FLIR",
        "cmap": load_custom_colormap("flir.npy"),
        "image_source": "interpolated",
        "norm": {
            "type": "logit",
            "weight": 0.29,
        },
        "imshow_kwargs": {},
    },
    "flir_high_contrast": {
        "label": "FLIR high contrast",
        "cmap": load_custom_colormap("flir_high_contrast.npy"),
        "image_source": "interpolated",
        "norm": {
            "type": "smooth_weighted_log",
            "weight_low": 2,
            "weight_high": 1.8,
            "midpoint": "median",
            "transition_width": 1,
        },
        "imshow_kwargs": {},
    },
    "fluke": {
        "label": "Fluke Ti10",
        "cmap": load_custom_colormap("fluke.npy"),
        "image_source": "sharpened",
        "norm": None,
        "imshow_kwargs": {},
    },
    "fluke_high_contrast": {
        "label": "Fluke high contrast",
        "cmap": load_custom_colormap("fluke_high_contrast.npy"),
        "image_source": "sharpened",
        "norm": None,
        "imshow_kwargs": {},
    },
    "seismic": {
        "label": "matplotlib seismic",
        "cmap": "seismic",
        "image_source": "interpolated",
        "norm": {
            "type": "smooth_weighted_log",
            "weight_low": 2,
            "weight_high": 1.8,
            "midpoint": "median",
            "transition_width": 1,
        },
        "imshow_kwargs": {},
    },
    "hot": {
        "label": "matplotlib hot",
        "cmap": "hot",
        "image_source": "interpolated",
        "norm": None,
        "imshow_kwargs": {},
    },
    "plasma": {
        "label": "matplotlib plasma",
        "cmap": "plasma",
        "image_source": "interpolated",
        "norm": {
            "type": "smooth_weighted_log",
            "weight_low": 6,
            "weight_high": 0.5,
            "midpoint": "median",
            "transition_width": 10,
        },
        "imshow_kwargs": {},
    },
}


def get_colormap_names():
    return tuple(sorted(COLORMAP_CONFIGS))


def get_colormap_config(name):
    return COLORMAP_CONFIGS[name]
