# 3D-reconstruction-from-single-image
This repo contains a PyTorch implementation of the first part of the paper "Attention-Based Dense Point Cloud Reconstruction from a Single Image", also available at the following [link](https://github.com/VIM-Lab/AttentionDPCR.git)

# Overview
This project implements a neural network to generate the 3D point cloud of an object, starting just from one single image of it. The network is composed by an encoder based on an attention mechanism, and a decoder which maps extracted features into the final point cloud. I have implemented and trained the network from scratch. The already trained model available here is trained on just 2/4 classes of objects, which are specified in the section Dataset.

# Preliminary
The file `config.yaml` contains all paths you have to specify in order to make the program running.

# Dataset
The dataset used to train the network is ShapeNetCore, which can be downloaded at the following [link](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) on huggingface. The only classes used to train the model available in the folder "trained_models" are (da inserire le classi). I trained just on these ones due to lack of very powerful hardware, but you can train on the classes you want. The reason why I decided these classes is because they represent very different classes of objects, so the network has more difficulty to generalize. Because this dataset does not contain already the rendering of the 3D models of the objects, I downloaded the rendered images at the following link (da inserire il link).

# Setup
If you want to train the network, once you have downloaded the dataset and the rendered images dataset, open the file `move_dirs.sh` inside `requirements` folder and put the correct paths of the datasets in the following lines:
```
# Define the main folders with absolute paths
shapes_folder= (put_your_path_to_ShapeNetCore_dataset)
images_folder= (put_your_path_to_rendered_images_dataset)
```
Then launch the program:
```
bash move_dirs.sh
```
that will move all the rendered images inside the corresponding models' folders. \\
Once the process is terminated, you can open the file `config.yaml`, where you can set the path where you want to save the csv file. Then, inside the `requirements` folder, execute the following script:
```
python create_csv_dataset.py
```


# Network

# Training

# Test

# Example
