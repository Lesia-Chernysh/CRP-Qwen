import torch
import numpy as np
from typing import List
import os
from pathlib import Path
from collections import namedtuple

MaxStats = namedtuple("MaxStats", "sample_idx rel_or_act rf_idx")


def get_layer_names(model: torch.nn.Module, types: List):
    """
    Retrieves the layer names of all layers that belong to a torch.nn.Module type defined 
    in 'types'.

    Parameters
    ----------
    model: torch.nn.Module
    types: list of torch.nn.Module
        Layer types i.e. torch.nn.Conv2D

    Returns
    -------
    layer_names: list of strings


    """

    layer_names = []

    for name, layer in model.named_modules():
        for layer_definition in types:
            if isinstance(layer, layer_definition) or issubclass(layer.__class__, layer_definition):
                if name not in layer_names:
                    layer_names.append(name)

    return layer_names


def abs_norm(rel: torch.Tensor, stabilize=1e-10):
    """

    Parameter:
        rel: 1-D array
    """

    abs_sum = torch.sum(torch.abs(rel))

    return rel / (abs_sum + stabilize)

def max_norm(rel, stabilize=1e-10):
    
    return rel / (rel.max() + stabilize)


def get_output_shapes(model, single_sample: torch.tensor, record_layers: List[str]):
    """
    calculates the output shape of each layer using a forward pass.


    """

    output_shapes = {}

    def generate_hook(name):

        def shape_hook(module, input, output):
            output_shapes[name] = output.shape[1:]

        return shape_hook

    hooks = []
    for name, layer in model.named_modules():
        if name in record_layers:
            shape_hook = generate_hook(name)
            hooks.append(layer.register_forward_hook(shape_hook))

    _ = model(single_sample)

    [h.remove() for h in hooks]

    return output_shapes


from pathlib import Path
import re
import numpy as np


def load_maximization(path_folder, layer_name):
    path_folder = Path(path_folder)

    # Matches:
    # <layer_name>_0_100_data.npy
    # <layer_name>_100_200_data.npy
    #
    # and similarly rel/rf.
    pattern = re.compile(
        rf"^{re.escape(layer_name)}_(\d+)_(\d+)_data\.npy$"
    )

    checkpoint_files = []

    for path in path_folder.glob(f"{layer_name}_*_data.npy"):
        match = pattern.match(path.name)

        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            checkpoint_files.append((start, end, path))

    # If no checkpoint files exist, fall back to the old final-file format.
    if not checkpoint_files:
        print("no checkpoint files")
        filename = f"{layer_name}_"

        d_c_sorted = np.load(
            path_folder / f"{filename}data.npy",
            mmap_mode="r",
        )
        rel_c_sorted = np.load(
            path_folder / f"{filename}rel.npy",
            mmap_mode="r",
        )
        rf_c_sorted = np.load(
            path_folder / f"{filename}rf.npy",
            mmap_mode="r",
        )

        return MaxStats(
            d_c_sorted,
            rel_c_sorted,
            rf_c_sorted,
        )

    # Sort checkpoints by their start/end ranges:
    # 0_100, 100_200, 200_300, ...
    checkpoint_files.sort(key=lambda x: (x[0], x[1]))

    data_parts = []
    rel_parts = []
    rf_parts = []

    for start, end, data_path in checkpoint_files:
        prefix = f"{layer_name}_{start}_{end}_"

        rel_path = path_folder / f"{prefix}rel.npy"
        rf_path = path_folder / f"{prefix}rf.npy"

        if not rel_path.exists():
            raise FileNotFoundError(
                f"Missing relevance checkpoint: {rel_path}"
            )

        if not rf_path.exists():
            raise FileNotFoundError(
                f"Missing RF checkpoint: {rf_path}"
            )

        data_parts.append(np.load(data_path))
        rel_parts.append(np.load(rel_path))
        rf_parts.append(np.load(rf_path))

    d_c_sorted = np.concatenate(data_parts, axis=0)
    rel_c_sorted = np.concatenate(rel_parts, axis=0)
    rf_c_sorted = np.concatenate(rf_parts, axis=0)

    return MaxStats(
        d_c_sorted,
        rel_c_sorted,
        rf_c_sorted,
    )

def load_stat_targets(path_folder):

    targets = np.load(Path(path_folder) / Path("targets.npy")).astype(int)

    return targets


def load_statistics(path_folder, layer_name, target):

    filename = f"{target}_"

    d_c_sorted = np.load(Path(path_folder) / Path(layer_name) / Path(filename + "data.npy"), mmap_mode="r")
    rel_c_sorted = np.load(Path(path_folder) / Path(layer_name) / Path(filename + "rel.npy"), mmap_mode="r")
    rf_c_sorted = np.load(Path(path_folder) / Path(layer_name) / Path(filename + "rf.npy"), mmap_mode="r")

    return MaxStats(d_c_sorted, rel_c_sorted, rf_c_sorted)


def load_receptive_field(path_folder, layer_name):

    filename = f"{layer_name}.npy"

    rf_array = np.load(Path(path_folder) / Path(filename), mmap_mode="r")

    return rf_array


def find_files(path=None):
    """
    Parameters:
        path: path analysis results

    """
    if path is None:
        path = os.getcwd()

    folders = os.listdir(path)

    r_max, a_max, r_stats, a_stats, rf = [], [], [], [], []
    for name in folders:
        found_path = str(Path(path) / Path(name))
        if "RelMax" in name:
            r_max.append(found_path)
        elif "ActMax" in name:
            a_max.append(found_path)
        elif "RelStats" in name:
            r_stats.append(found_path)
        elif "ActStats" in name:
            a_stats.append(found_path)
        elif "ReField" in name:
            rf.append(found_path)

    return r_max, a_max, r_stats, a_stats, rf
