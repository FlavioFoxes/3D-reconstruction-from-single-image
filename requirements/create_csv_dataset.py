import sys
sys.path.append('../')
import csv
import os
import numpy as np
from src.utils.mesh2point import mesh_to_point_cloud
from src.utils.utils import load_config


def create_dataset(root_path, csv_path, list_images):
    with open(csv_path, 'w', newline='') as file:
        os.chdir(root_path)
        writer = csv.writer(file)
        classes_list = [d for d in os.listdir(os.getcwd()) if os.path.isdir(d)]
        
        for dir in classes_list:

            os.chdir(dir)
            objects_list = os.listdir()
            for object_folder in objects_list:
                os.chdir(object_folder)
                os.chdir("models")
                points_cloud = mesh_to_point_cloud("model_normalized.obj")
                sorted_indices = np.argsort(points_cloud[:, 0])
                sorted_point_cloud = points_cloud[sorted_indices]

                os.chdir("..")
                os.chdir("renderings")
                images_list = os.listdir()
                for image in images_list:
                    image_name, image_ext = os.path.splitext(image)
                    if image_ext == ".png" and (int(image_name) in list_images):
                        name = dir + "/" + object_folder + "/renderings/" + image
                        row = [name]
                        for point in sorted_point_cloud:
                            for coordinate in point:
                                row.append(coordinate)
                
                        writer.writerow(row)
                        
                
                os.chdir(root_path+dir)   
            os.chdir(root_path)
        
if __name__ == "__main__":
    config = load_config("../config.yaml")
    DATASET_PATH = config['dataset_path']
    CSV_PATH = config['csv_path']
    list_images = [4,5,6,7,8,9,10,11]
    create_dataset(DATASET_PATH, CSV_PATH, list_images)
    print("Created successfully!")
