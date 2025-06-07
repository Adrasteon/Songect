import torch
import os  # Added for os.path.exists

from .unet import UNets
from .beat_net import BeatNet
from .chord_net import ChordNet
from .pitch_net import PitchNet
from .segment_net import SegmentNet


def get_model_cls(s):
    if s == 'unet':  # UNets is the wrapper for multiple UNet instances
        return UNets
    elif s == 'beat_net' or s == 'beat':  # Accept both "beat_net" and "beat"
        return BeatNet
    elif s == 'chord_net' or s == 'chord':  # Accept both "chord_net" and "chord"
        return ChordNet
    elif s == 'pitch_net' or s == 'pitch':  # Accept both "pitch_net" and "pitch"
        return PitchNet
    elif s == 'segment_net' or s == 'segment':  # Accept both "segment_net" and "segment"
        return SegmentNet
    else:
        raise ValueError(f'Invalid model name: {s}')


def get_model(model, config, model_path=None, is_train=True, device='cpu'):
    model_cls = None
    model_display_name = "UnknownModel"

    if isinstance(model, str):
        model_display_name = model
        model_cls = get_model_cls(model)
    elif hasattr(model, '__name__'):  # Check if it's a class
        model_cls = model
        model_display_name = model.__name__
    else:
        raise ValueError("'model' argument must be a string name or a model class.")

    net = model_cls(**config)

    weights_loaded_successfully = False  # Flag to indicate if weights were loaded
    if model_path and model_path.strip():  # Check if model_path is not None and not empty
        if os.path.exists(model_path):
            try:
                net.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded trained weights for {model_display_name} from {model_path}")
                weights_loaded_successfully = True
            except Exception as e:
                print(
                    f"Error loading weights for {model_display_name} from {model_path}: {e}. Using model with random weights.")
        else:
            print(
                f"Warning: Model path '{model_path}' not found for {model_display_name}. Using model with random weights.")
    else:
        print(
            f"Warning: No model path provided for {model_display_name}. Using model with random weights.")

    net.to(device)

    if is_train:
        net.train()
    else:
        net.eval()

    return net, weights_loaded_successfully  # Return model and loading status
