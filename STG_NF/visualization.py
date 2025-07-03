import json
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import re

#Visualizing some of the json files in the PoseLift dataset


json_file_path = "./data/PoseLift/pose/test/01_0257_alphapose_tracked_person.json" 
#05_0298_alphapose_tracked_person.json
#04_0308_alphapose_tracked_person.json

person_id = "1"  # ID of the person you want to visualize
output_dir = os.path.join("output_images",json_file_path.split('/')[-1].split('.')[0])  # Directory to save the images
num_frames = 12  # Number of consecutive frames to extract
video_filename = os.path.join(output_dir,json_file_path.split('/')[-1].split('.')[0]+".mp4") # "data_video.mp4"  # Output video file
fps = 5  # Adjust FPS

# COCO Skeleton Connection Mapping (for visualization)
skeleton_connections = [
    (0, 1), (0, 2),  # Nose to Eyes
    (1, 3), (2, 4),  # Eyes to Ears
    (5, 6),          # Shoulders
    (5, 7), (7, 9),  # Left Arm
    (6, 8), (8, 10), # Right Arm
    (5, 11), (6, 12),  # Shoulders to Hips
    (11, 12),        # Hip connection
    (11, 13), (13, 15),  # Left Leg
    (12, 14), (14, 16)   # Right Leg
]

with open(json_file_path, "r") as f:
    data = json.load(f)

if person_id not in data:
    raise ValueError(f"Person ID {person_id} not found in the dataset.")

frames = sorted(data[person_id].keys(), key=int)  
selected_frames = frames[:num_frames]  

os.makedirs(output_dir, exist_ok=True)

def plot_keypoints(keypoints, save_path):
    plt.figure(figsize=(12.8, 8))
    plt.axis("equal")
    plt.xlim(0, 1280)  
    plt.ylim(0, 800)
    plt.gca().invert_yaxis()  

    keypoints = np.array(keypoints).reshape(-1, 3) 
    y, x, confidence = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
    plt.scatter(x, y, c='red', s=10)

    for (i, j) in skeleton_connections:
        if i < len(keypoints) and j < len(keypoints):
            plt.plot([x[i], x[j]], [y[i], y[j]], 'b-', linewidth=1)

    plt.title(f"Frame: {os.path.basename(save_path)}", fontsize=10, y=-0.2)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

for i, frame in enumerate(selected_frames):
    keypoints = data[person_id][frame]["keypoints"]
    save_path = os.path.join(output_dir, f"frame_{i+1}.png")
    plot_keypoints(keypoints, save_path)


#Create a video 

# Function to extract numeric frame number from filenames like "frame_1"
def extract_frame_number(filename):
    match = re.search(r"frame_(\d+)", os.path.basename(filename))  
    return int(match.group(1)) if match else float("inf")  

# Get sorted list of image files
image_files = sorted(glob.glob(os.path.join(output_dir, "*.png")), key=extract_frame_number)


frame = cv2.imread(image_files[0])
height, width, layers = frame.shape
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)


for image_file in image_files:
    frame = cv2.imread(image_file)
    video_writer.write(frame)

video_writer.release()
print(f"Video saved as {video_filename}")


