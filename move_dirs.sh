#!/bin/bash

is_empty_directory() {
    local dir="$1"
    if [ -d "$dir" ] && [ -z "$(ls -A "$dir")" ]; then
        return 0
    else
        return 1
    fi
}

# Define the main folders
shapes_folder="/home/flavio/Documenti/Datasets/ShapeNetCore"
images_folder="/home/flavio/Documenti/Datasets/ShapeNetImages"


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
        #echo "$dir"
        # Extract the directory name
        dir_name=$(basename "$dir")
        # Check if the corresponding directory exists in Shapes
        shapes_subfolder="$shapes_folder/$dir_name"
        if [ -d "$shapes_subfolder" ]; then
            for subdir in "$dir"/*; do
                #echo "$subdir"
                subdir_name=$(basename "$subdir")
                shapes_sub2folder="$shapes_subfolder/$subdir_name"
                #echo "$shapes_sub2folder"
                if [ -d "$shapes_sub2folder" ]; then
                    if is_empty_directory "$subdir/easy"; then
                        rm -r "$shapes_sub2folder"

                    else
                        # Copy the directory from Images to Shapes
                        echo "Copied '$subdir/easy' from 'Images' to '$shapes_sub2folder'"
                        cp -r "$subdir/easy" "$shapes_sub2folder"
                        mv "$shapes_sub2folder/easy" "$shapes_sub2folder/renderings"
                    fi
                fi
            done
        fi
    fi
done
