"""This file, used for moving original images in one train folder to 
   different folders based on image's label.
"""

import os
import csv
import shutil

####################### Define your FULL_PATH, the path where you put the raw training dataset###############
FULL_PATH = "/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/train_data/"

####################### Define your FULL_PATH, the path where you target to put processed dataset############
DIR_PATH = "/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/train/"

# Construct a name-label map 
pic_label_map = {}
with open('labels.csv', 'r') as f:
	next(f)
	reader = csv.reader(f)
	for row in reader:
		name = row[0]+".jpg"
		pic_label_map[name] = row[1]

# Divide train image to its directory based on label, each directory contains all images 
# belonging to one label
list_dirs = os.walk(DIR_PATH)
for _, _, files in list_dirs:
    for f in files:
        name = os.path.basename(f)
        label = pic_label_map[name]
        image_src_path = os.path.join(DIR_PATH, name)
        image_dest_path = FULL_PATH+label+'./'
        if os.path.exists(image_dest_path):
            shutil.move(image_src_path, image_dest_path)
        else:
            os.mkdir(image_dest_path)
            shutil.move(image_src_path, image_dest_path)


