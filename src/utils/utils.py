import sys
sys.path.append('3D-reconstruction-from-single-image/')
import yaml
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

"""
Load yaml file.

Argument:
        file_path:      path to the yaml file
"""
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

"""
Print the device where each parameter of the model sits (cpu/gpu)
"""
def check_model_device(model):
    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")

"""
Print the gradient of each parameter of the model
"""
def check_gradients(model):
    print("Gradients:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.grad}")

"""
Print the update value of each weight of the model
"""
def print_weight_updates(model, prev_params):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Difference in {name}: {torch.sum(param.data - prev_params[name])}")

"""
Save a checkpoint of the model during training stage.

Arguments:
        epoch:          current epoch
        loss:           current loss
        model:          model
        optimizer:      optimization algorithm
"""
def save_checkpoint(epoch, loss, model, optimizer):
    config = load_config("config.yaml")

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, config['save_checkpoint'])

"""
Split dataset in Train, Eval, Test datasets

Argument:
        data (DataFrame):       DataFrame of the entire dataset. 
                                Splits always in the same Train, Eval, Test datasets
"""
def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    test_data, eval_data = train_test_split(test_data, test_size=0.5, random_state=42)

    return train_data, eval_data, test_data
