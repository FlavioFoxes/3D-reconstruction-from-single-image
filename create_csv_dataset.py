import csv
import os
from mesh2point import mesh_to_point_cloud


def create_dataset(root_path):
    with open("dataset.csv", 'w', newline='') as file:
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
                os.chdir("..")
                os.chdir("renderings")
                images_list = os.listdir()
                for image in images_list:
                    image_name, image_ext = os.path.splitext(image)
                    if image_ext == ".png" and int(image_name)%5==0:
                        name = dir + "/" + object_folder + "/renderings/" + image
                        # print(name)
                        # this is the name file to be written in the csv file. Now we have to 
                        # extract from the corresponding mesh model the point cloud and put
                        # the coordinates of the points in the csv file
                        row = [name]
                        for point in points_cloud:
                            for coordinate in point:
                                row.append(coordinate)
                
                        # (row.append(coordinate) for point in points_cloud for coordinate in point)
                        writer.writerow(row)
                        # print(row)

                
                os.chdir(root_path+dir)   
            os.chdir(root_path)
        

PATH = "/home/flavio/Documenti/Datasets/ShapeNetCore/"
create_dataset(PATH)





# path = input("Enter root path of the dataset...")
# with open("dataset.csv", 'w', newline='') as file:
#     writer = csv.writer(file)