import argparse
import sys
import os
import csv

import tensorflow as tf


def label_format(label):
    label = list(label)
    for i in range(len(label)):
        if label[i] == '-' or label[i] == '_':
            label[i] = ' '
    label = "".join(label)
    return label

if __name__ == "__main__":
    # change this as you see fit
    image_path = '../tf_files/test/7d456cad378a38055723949d8cbbb811.jpg'
    graph_path = '../tf_files/retrained_graph.pb'
    label_lines_path = '../tf_files/retrained_labels.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path of image to be processed")
    parser.add_argument("--graph", help="path of graph/model to be executed")
    parser.add_argument("--labels", help="path of file containing labels")
    args = parser.parse_args()

    if args.image:
        image_path = args.image
    if args.graph:
        graph_path = args.graph
    if args.labels:
        label_lines_path = args.labels

        # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line in tf.gfile.GFile(
        label_lines_path)]

    # Unpersists graph from file
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        predictions = predictions[0]
        # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        # for node_id in top_k:
        #     human_string = label_lines[node_id]
        #     score = predictions[0][node_id]
        #     print('%s (score = %.5f)' % (human_string, score))


        with open('/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/sample_submission.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                header = row
                break
                
        label_score_map = {}
        for label, score in zip(label_lines, predictions):
            label_score_map[label] = score
        
        file_name_base = (image_path.split('/')[-1]).split('.')[0]
        image_labels = [file_name_base]

        if not os.path.exists('/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/tensorflow-for-poets-2/output/output.csv'):
            with open('/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/tensorflow-for-poets-2/output/output.csv', 'w') as outfile: 
                writer = csv.writer(outfile)
                writer.writerow(header)   

        with open('/Users/bowenhe/Downloads/ECS289/project/dog-breed-identification/tensorflow-for-poets-2/output/output.csv', 'a') as outfile:
            writer = csv.writer(outfile)
            for label in header[1:]:
                label = label_format(label)
                image_labels.append(label_score_map[label])
            writer.writerow(image_labels)
