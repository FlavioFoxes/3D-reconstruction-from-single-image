#!/bin/bash

# Define the main folders
shapes_folder="/home/flavio/Documenti/Datasets/ShapeNetCore"
images_folder="/home/flavio/Documenti/Datasets/image"

# Check if the Shapes folder exists
if [ ! -d "$shapes_folder" ]; then
    echo "Error: Shapes folder does not exist."
    exit 1
fi

# Check if the Images folder exists
if [ ! -d "$images_folder" ]; then
    echo "Error: Images folder does not exist."
    exit 1
fi

# Iterate over the folders in the Images directory
for dir in "$images_folder"/*; do
    if [ -d "$dir" ]; then
        # Extract the directory name
        dir_name=$(basename "$dir")
        # Check if the corresponding directory exists in Shapes
        shapes_subfolder="$shapes_folder/$dir_name"
        if [ -d "$shapes_subfolder" ]; then
            # Copy the directory from Images to Shapes
            cp -r "$dir"/* "$shapes_subfolder/example/"
            echo "Copied '$dir_name' from 'Images' to '$shapes_subfolder'"
        fi
    fi
done
