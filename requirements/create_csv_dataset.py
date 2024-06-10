import csv
import os
import numpy as np
from utils.mesh2point import mesh_to_point_cloud


def create_dataset(root_path, list_images):
    with open("dataset_ordered_two_classes.csv", 'w', newline='') as file:
        os.chdir(root_path)
        writer = csv.writer(file)
        classes_list = os.listdir()
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
                    if image_ext == ".png" and (int(image_name) in list_images): #or int(image_name)==1):
                        name = dir + "/" + object_folder + "/renderings/" + image
                        # print(name)
                        # this is the name file to be written in the csv file. Now we have to 
                        # extract from the corresponding mesh model the point cloud and put
                        # the coordinates of the points in the csv file
                        row = [name]
                        for point in sorted_point_cloud:
                            for coordinate in point:
                                row.append(coordinate)
                
                        # (row.append(coordinate) for point in sorted_point_cloud for coordinate in point)
                        writer.writerow(row)
                        # print(row)

                
                os.chdir(root_path+dir)   
            os.chdir(root_path)
        

PATH = "/home/flavio/Documenti/Datasets/ShapeNetCore_TwoClasses/"
list_images = [4,5,6,7,8,9,10,11]
create_dataset(PATH, list_images)





# path = input("Enter root path of the dataset...")
# with open("dataset.csv", 'w', newline='') as file:
#     writer = csv.writer(file)