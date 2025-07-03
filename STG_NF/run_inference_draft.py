import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace
from models.STG_NF.model_pose import STG_NF
from utils.train_utils import init_model_params
from utils.scoring_utils import get_dataset_scores, score_auc, smooth_scores
from dataset import get_dataset_and_loader
from utils.data_utils import trans_list
import os
import csv
from dataset import shanghaitech_hr_skip


# === CONFIGURATION ===
CHECKPOINT_PATH = 'data/exp_dir/PoseLift/Jul01_2235/Jul01_2235__checkpoint.pth.tar' # path to the pretrained .tar file
INPUT_JSON_DIR_PATH = "data/PoseLift/pose/test"  # path to input json file
OUTPUT_CSV_DIR = "check_inference_result/csv_files/"
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# === Step 1: Setup args to mimic official inference ===
args = SimpleNamespace(
    dataset='PoseLift',
    pose_path={'test': INPUT_JSON_DIR_PATH},  # path with your .json files
    vid_path={'test': INPUT_JSON_DIR_PATH},
    batch_size=1,
    num_workers=0,
    headless=False,
    norm_scale=1,
    prop_norm_scale=False,
    seg_len=24,
    return_indices=True,
    return_metadata=True,
    train_seg_conf_th=0.0,
    specific_clip=None,
    global_pose_segs=True,
    seg_stride=1,
    num_transform=1,
    pose_path_train_abnormal=None,
    checkpoint=CHECKPOINT_PATH
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Step 2: Load checkpoint and model args ===
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
ckpt_args = checkpoint['args']
if hasattr(ckpt_args, '__dict__'):
    ckpt_args = SimpleNamespace(**vars(ckpt_args))

ckpt_args.device = device
ckpt_args.edge_importance = False
ckpt_args.model_confidence = False
ckpt_args.norm_scale = 1
ckpt_args.seg_len = 24
ckpt_args.adj_strategy = 'uniform'
ckpt_args.max_hops = 1
ckpt_args.temporal_kernel = 13
ckpt_args.model_hidden_dim = getattr(ckpt_args, 'model_hidden_dim', 128)

# === Step 3: Load dataset and model ===
dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=True)
model_params = init_model_params(ckpt_args, dataset)
model = STG_NF(**model_params).to(device)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.set_actnorm_init()
model.eval()

# Accumulate all scores
all_scores = []

with torch.no_grad():
    for data_arr in loader['test']:
        data = [d.to(device, non_blocking=True) for d in data_arr]
        samp = data[0][:, :2]  # drop confidence
        score = data[-2].amin(dim=-1)
        label = torch.ones(data[0].shape[0])

        # Forward pass
        _, nll = model(samp.float(), label=label, score=score)
        all_scores.append(-nll.cpu())

# Flatten into 1D array (as expected by score_dataset)
normality_scores = torch.cat(all_scores, dim=0).squeeze().numpy().copy(order='C')

# Run scoring (this will also save per-frame CSVs)
auc_roc, scores_np, auc_pr, eer, eer_threshold = score_dataset(
    normality_scores,
    dataset['test'].metadata,
    args=args,
    save_results=True,
    directory=OUTPUT_CSV_DIR
)

# Print summary
print(f"ROC AUC: {auc_roc:.4f}")
print(f"PR AUC: {auc_pr:.4f}")
print(f"EER: {eer:.4f}, Threshold: {eer_threshold:.4f}")
print(f"Number of samples: {scores_np.shape[0]}")