import numpy as np
import os.path
from os import path
import pickle
import json
import os


data_path = "/home/training_pickle" #pickle_data_path
out_path = "/home/testing_json"  #json_output_path             

# Ensure the output directory exists
if not os.path.exists(out_path):
    os.makedirs(out_path)
for subdir, dirs, files in os.walk(data_path):
    for file in files:
            file_path = os.path.join(subdir, file)
            infile = open(file_path, 'rb')   
            annotations = pickle.load(infile)
            anno_dict = {}
            anno_list = []
            for frame in annotations:
                anno_list.append(annotations[frame])
            ids = list(set(key for dic in anno_list for key in dic.keys()))
            for person in ids:
                person_dict = {}
                for frame in annotations:
                    if person in annotations[frame]:
                        frame_dict = {}
                        frame_dict["keypoints"] = [float(x) for x in list(annotations[frame][person][1].flatten())]
                        frame_dict["scores"] = None
                        person_dict[str(frame)] = frame_dict
                    
                anno_dict[str(person)] = person_dict
            if len(anno_dict.keys()) != 0:
                output_file_path= os.path.join(out_path, file[:-3] + 'json')
                with open (output_file_path, 'w') as fp:       
                    json.dump(anno_dict, fp)
                
                    
            


