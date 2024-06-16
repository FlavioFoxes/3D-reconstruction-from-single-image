# 3D-reconstruction-from-single-image
This repo contains a PyTorch implementation of the first part of the paper "Attention-Based Dense Point Cloud Reconstruction from a Single Image", also available at the following [link](https://github.com/VIM-Lab/AttentionDPCR.git)

# Overview
This project implements a neural network to generate the 3D point cloud of an object, starting just from one single image of it. The network is composed by an encoder based on an attention mechanism, and a decoder which maps extracted features into the final point cloud. The main structure of the network is shown below. \
I have implemented and trained the network from scratch. The already trained model available here is trained on just two classes of objects, which are specified in the section Dataset.

# Preliminary
The file `config.yaml` contains all paths you have to specify in order to make the program running.

# Dataset
The dataset used to train the network is **ShapeNetCore**, which can be downloaded at the following [link](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) on huggingface. The only classes used to train the model available in the folder "**trained_models**" are **02958343** and **03948459**. I trained just on these ones due to lack of very powerful hardware, but you can train on the classes you want. The reason why I decided these classes is because they represent very different classes of objects, so the network has more difficulty to generalize. Because this dataset does not contain already the renderings of the 3D models of the objects, I downloaded the rendered images at the following [link](https://github.com/Xharlie/ShapenetRender_more_variation.git), from the paper [DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction (NeurIPS 2019)](https://proceedings.neurips.cc/paper_files/paper/2019/file/39059724f73a9969845dfe4146c5660e-Paper.pdf)

# Setup
If you have already the csv file representing the dataset, you can skip directly to the next sections. \
If you want to train the network, once you have downloaded the dataset and the rendered images dataset, you have to create the csv file that will be used from the DataLoaders. So, open the file `move_dirs.sh` inside `requirements` folder and put the correct paths of the datasets in the following lines:
```
# Define the main folders with absolute paths
shapes_folder= (put_your_path_to_ShapeNetCore_dataset)
images_folder= (put_your_path_to_rendered_images_dataset)
```
Then launch the program:
```
bash move_dirs.sh
```
that will move all the rendered images inside the corresponding models' folders. \
Once the process is terminated, you can open the file `config.yaml`, where you can set the path where you want to save the csv file. Then, inside the `requirements` folder, execute the following script:
```
python create_csv_dataset.py
```
Every row of this file represents an image. The first column is the path to the image, the following ones represents the (x,y,z) coordinates of the ground truth points.\

Once you have the csv file, you can run the training phase, the test phase, or show some examples.

# Usage
Before launching one of the following processes, check (and eventually modify) the paths inside `config.yaml`.\
You can run `python main.py --help` to understand which arguments are available.

## Training
Before launching the training process, the file `training.yaml` inside the folder `src/train` can be modified according to own puproses. \
To start the training of a new network, run:
```
python main.py --train
```
To train a pretrained model, run:
```
python main.py --pretrain
```
where the network will be trained is the one specified in the path `load_model` inside `config.yaml`

## Test
To test a network, run:
```
python main.py --test
```
It returns the average loss on the test set, and the list of all indices of the samples inside the test set.

## Example
To show some examples, you can run the following commands:
* to plot the ground truth of a sample:
  ```
  python main.py --ground
  ```
* to plot the prediction of the network on a sample:
  ```
  python main.py --pred
  ```
* to compare the two plots on the same sample:
  ```
  python main.py --example
  ```
