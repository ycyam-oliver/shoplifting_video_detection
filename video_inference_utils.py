import cv2
import numpy as np

import torch
from mmpose.apis import inference_top_down_pose_model
from SmoothNet.lib.models.smoothnet import SmoothNet  
from SmoothNet.lib.utils.utils import slide_window_to_sequence

import yaml
import json

# ===================== For inference by ViTPose ===================== 

# Infering keypoints of a person in an image
def infer_keypoint(img, dets, pose_model, dataset_name, dataset_info):
    
    # takes an image (img) and detection (dets) from a YOLO model in supervision format
    # and use the yolo and pose model to estimate the keypoint 
    
    boxes = dets.xyxy
    scores = dets.confidence

    person_results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        score = scores[i]
        person_results.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2), float(score)],
            "track_id": i
        })

    pose_results, returned_scores = inference_top_down_pose_model(
        pose_model,
        img,
        person_results,
        bbox_thr=0.3,
        format="xyxy",
        dataset=dataset_name,
        dataset_info=dataset_info
    )
    return pose_results

# ===================== For inference by SmoothNet ===================== 
def load_smoothnet_from_yaml(yaml_path, checkpoint_path, device="cpu"):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg['MODEL']

    window_size = 32

    model = SmoothNet(
        window_size=window_size,
        output_size=window_size,
        hidden_size=model_cfg['HIDDEN_SIZE'],
        res_hidden_size=model_cfg['RES_HIDDEN_SIZE'],
        num_blocks=model_cfg['NUM_BLOCK'],
        dropout=model_cfg['DROPOUT']
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model, window_size, cfg['EVALUATE']['SLIDE_WINDOW_STEP_SIZE']

def apply_smoothnet_to_2d_seq(
    keypoints_seq: np.ndarray,        # (T, J, 2) in pixel coordinates
    yaml_path: str,
    checkpoint_path: str,
    image_shape: tuple,              # (height, width)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> np.ndarray:
    
    T_orig, J, D = keypoints_seq.shape
    assert D == 2, "Keypoints must be (T, J, 2)"

    model, window_size, step_size = load_smoothnet_from_yaml(yaml_path, checkpoint_path, device)
    height, width = image_shape

    # Normalize to [-1, 1] for each axis independently
    kp_norm = np.empty_like(keypoints_seq, dtype=np.float32)
    kp_norm[..., 0] = (keypoints_seq[..., 0] - width / 2) / (width / 2)
    kp_norm[..., 1] = (keypoints_seq[..., 1] - height / 2) / (height / 2)

    input_tensor = torch.tensor(kp_norm, dtype=torch.float32).to(device)  # (T, J, 2)

    # -------- Padding if sequence is too short --------
    padded = False
    pad_left = pad_right = 0
    if T_orig < window_size:
        pad_total = window_size - T_orig
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_left_tensor = input_tensor[0:1].repeat(pad_left, 1, 1)
        pad_right_tensor = input_tensor[-1:].repeat(pad_right, 1, 1)
        input_tensor = torch.cat([pad_left_tensor, input_tensor, pad_right_tensor], dim=0)
        padded = True

    T = input_tensor.shape[0]  # Updated T after padding
    input_flat = input_tensor.reshape(T, -1)  # (T, J*2)
    C = input_flat.shape[1]

    # Sliding window using as_strided
    stride = step_size

    num_windows = (T - window_size) // stride + 1
    
    starts = list(range(0, T - window_size + 1, stride))
    windows = torch.stack([
        input_tensor[i:i+window_size].reshape(window_size, -1)
        for i in starts
    ])  # (B, W, C)
    windows = windows.permute(0, 2, 1)  # (B, C, W)

    with torch.no_grad():
        out = model(windows)  # (B, C, W)
        out = out.permute(0, 2, 1)       # (B, W, C)
        out = out.reshape(num_windows, window_size, J, 2)

    # Reconstruct sequence from overlapping windows
    smoothed = slide_window_to_sequence(out, stride, window_size)  # (T, J, 2)

    # Remove padding to get back to original T
    if padded:
        smoothed = smoothed[pad_left:pad_left + T_orig]

    # Denormalize back to pixel coordinates
    if isinstance(smoothed, torch.Tensor):
        smoothed_cpu = smoothed.detach().cpu().numpy()
    else:
        smoothed_cpu = smoothed  # Already a NumPy array
    kp_denorm = np.empty_like(smoothed_cpu)

    kp_denorm[..., 0] = smoothed_cpu[..., 0] * (width / 2) + (width / 2)
    kp_denorm[..., 1] = smoothed_cpu[..., 1] * (height / 2) + (height / 2)

    return kp_denorm

# ===================== For outputting videos ===================== 

# Drawing keypoints on the video
def draw_keypoints(frame, keypoints, radius=2, color=(0, 255, 0)):
    """
    Draws keypoints on the given frame.

    Args:
        frame: The video frame (H, W, 3)
        keypoints: (J, 2) array of (x, y) pixel coordinates
        radius: Circle radius
        color: RGB tuple
    """
    for x, y in keypoints:
        if x > 0 and y > 0:  # skip invalid points
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    return frame

def overlay_keypoints_on_video(video_path, track_sessions, output_path="video_output/output_with_keypoints.mp4", color_map = None):
    """
    Overlays smoothed keypoints on the video and saves it.

    Args:
        video_path: path to original input video
        track_sessions: {tracker_id: collected_sessions} 
         with collected_sessions which is also a dict of {frame_ind_start: (frame_ind_end, smoothened_kpts)}
         where smoothened_kpts is of shape (T, J, 2) numpy array of keypoints
        output_path: where to save the video
        colormap: {tracker_id: (R,G,B)} a dict mapping tracker_id to the tuple of rgb color
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    track_start_indices = {}
    track_start_indices_ind = {}
    tracker_ids = list(track_sessions.keys())
    if color_map is not None:
        # to match the BGR convention in cv2
        color_map = {cc: (color_map[cc][2],color_map[cc][1],color_map[cc][0]) for cc in color_map} # {tracker_id: color}
    else:
        color_map = {cc: (0,255,0) for cc in tracker_ids} # {tracker_id: color}
    
    for tracker_id in tracker_ids:
        track_start_indices[tracker_id] = sorted(list(track_sessions[tracker_id].keys()))
        track_start_indices_ind[tracker_id] = 0

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        for tracker_id in tracker_ids:
            start_indices_ind = track_start_indices_ind[tracker_id]
            curr_start_ind = track_start_indices[tracker_id][start_indices_ind]
            curr_end_ind = track_sessions[tracker_id][curr_start_ind][0]

            if curr_start_ind<=frame_idx<=curr_end_ind:
                curr_smoothened_kpts = track_sessions[tracker_id][curr_start_ind][1]
                frame = draw_keypoints(frame, curr_smoothened_kpts[frame_idx-curr_start_ind], color=color_map[tracker_id])
                if frame_idx==curr_end_ind and start_indices_ind<len(track_start_indices[tracker_id])-1:
                    track_start_indices_ind[tracker_id]+=1
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved video with smoothed keypoints to: {output_path}")

# ===================== For outputting json file for STG-NF model ===================== 

def convert_sessions_to_stgnf_json(collected_sessions, person_id=1, output_path="converted_output.json"):
    """
    Convert collected_sessions into STG-NF-compatible json format:
    {person_id: {frame_id: {"keypoints": [...], "score": None}}}
    
    Args:
        collected_sessions (dict): {frame_start: (frame_end, np.ndarray of shape (T, 17, 2))}
        person_id (int): ID of the tracked person
        output_path (str): Path to save the output JSON file
    """
    output_data = {str(person_id): {}}

    for frame_start, (frame_end, smooth_kpts) in collected_sessions.items():
        T = smooth_kpts.shape[0]
        for i in range(T):
            frame_idx = frame_start + i
            kpts_xy = smooth_kpts[i]  # shape (17, 2)

            # Add confidence of 0.8 to each keypoint
            kpts_with_conf = np.concatenate(
                [kpts_xy, np.full((17, 1), 0.8)], axis=1
            ).reshape(-1).tolist()  # Flatten to length 51

            output_data[str(person_id)][str(frame_idx)] = {
                "keypoints": kpts_with_conf,
                "scores": None
            }

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

def output_keypoints_to_json(track_sessions, output_dir="video_output/output_keypoints/"):
    """
    Overlays smoothed keypoints on the video and saves it.

    Args:
        video_path: path to original input video
        track_sessions: {tracker_id: collected_sessions} 
         with collected_sessions which is also a dict of {frame_ind_start: (frame_ind_end, smoothened_kpts)}
         where smoothened_kpts is of shape (T, J, 2) numpy array of keypoints
        output_path: where to save the video
    """
    track_ids = list(track_sessions.keys())
    for track_id in track_ids:
        session = track_sessions[track_id]
        convert_sessions_to_stgnf_json(session, track_id, output_dir+'_'+str(f'{int(100*track_id):04}')+'_alphapose_tracked_person.json')
