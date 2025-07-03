import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
import csv

def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr

def save_predicted_scores_only(scores, metadata, seg_len=24, save_dir="results/csv_files/", specific_clip=None):
    """
    Save per-frame normality scores to CSV for a specific clip (e.g., "01_0222_alphapose_tracked_person").

    Args:
        scores: Flat list/array of normality scores.
        metadata: dataset['test'].metadata, shape [N, 4], each row = [scene_id, clip_id, person_id, frame_id]
        seg_len: segment length used in model input
        save_dir: directory to save CSV
        specific_clip: name of the input .json file without extension (e.g., "01_0222_alphapose_tracked_person")
    """
    os.makedirs(save_dir, exist_ok=True)
    metadata_np = np.array(metadata)
    scores = np.atleast_1d(scores)

    # Parse scene_id and clip_id from specific_clip
    if specific_clip is None:
        raise ValueError("specific_clip must be provided")
    try:
        scene_str, clip_str, *_ = specific_clip.split('_')
        scene_id = int(scene_str)
        clip_id = int(clip_str)
    except Exception as e:
        raise ValueError(f"Cannot parse scene_id and clip_id from '{specific_clip}'") from e

    # Filter metadata rows for just this clip
    clip_inds = np.where((metadata_np[:, 0] == scene_id) & (metadata_np[:, 1] == clip_id))[0]
    clip_meta = metadata_np[clip_inds]

    # Sort by frame_id to ensure consistent ordering
    sorted_inds = np.argsort(clip_meta[:, 3])
    clip_inds = clip_inds[sorted_inds]
    clip_meta = clip_meta[sorted_inds]

    # Create full-frame array and assign mid-frame scores
    max_frame = clip_meta[:, 3].max() + seg_len + 1
    clip_scores = np.ones(max_frame) * np.inf

    for i, idx in enumerate(clip_inds):
        frame_idx = metadata[idx][3] + seg_len // 2
        clip_scores[frame_idx] = scores[i]

    # Fill inf with neighboring values (match official logic)
    valid = clip_scores != np.inf
    if np.any(valid):
        clip_scores[~valid] = np.nanmax(clip_scores[valid])
        clip_scores[clip_scores == -np.inf] = np.nanmin(clip_scores[valid])
    else:
        clip_scores.fill(1.0)  # fallback

    # Save to CSV
    # output_path = os.path.join(save_dir, f"{scene_id:02d}_{clip_id:04d}.csv")
    output_path = os.path.join(save_dir, specific_clip+".csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for s in clip_scores:
            writer.writerow([s])

    print(f"Saved: {output_path}  ({len(clip_scores)} frames)")
    return clip_scores