import subprocess
import csv

with open('/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/sample_submission.csv', 'r') as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            filename = \
"/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/tensorflow-for-poets-2/tf_files/test/"+row[0]+".jpg"
            image_dest = "--image="+filename
            subprocess.call(["python", "-m", "scripts.label_image_inceptionV3",
                "--graph=/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/tensorflow-for-poets-2/tf_files/retrained_graph.pb",
                image_dest,
                "--label=/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/tensorflow-for-poets-2/tf_files/retrained_labels.txt"])
            break


