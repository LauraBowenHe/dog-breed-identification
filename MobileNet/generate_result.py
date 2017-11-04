"""This file is used to generate results, by calling scripts.label_image 
   for each image on test directory, it will generate a result csv of the kaggle required format.
   
   Remember to change the path to where you put the sample_submission.csv, which downloaded from kaggle.
"""


import subprocess
import csv

with open('/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/sample_submission.csv', 'r') as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            filename = "tf_files/test/"+row[0]+".jpg"
            image_dest = "--image="+filename
            subprocess.call(["python", "-m", "scripts.label_image",
                "--graph=tf_files/retrained_graph.pb",
                image_dest])
