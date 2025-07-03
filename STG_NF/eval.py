import os 
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, MinMaxScaler
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import scipy.stats as stats
import csv



class ScoreNormalization:
    def __init__(self, method="KDE", options=None):
        self.method = method
        self.options = {} if options is None else options
        self.name = "_".join([method]+["-".join([str(key), str(value)]) for key, value in self.options.items()])
        if self.method == "KDE":
            kernel = self.options.get("kernel", "gaussian")
            bandwidth = self.options.get("bandwidth", 0.75)
            self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        elif self.method == "histogram":
            raise ("Not implemented method {}".format(self.method))
        elif self.method == "gamma":
            self.k = self.options.get("k", 1)
            self.loc = self.options.get("loc", 0)
            self.theta = self.options.get("theta", 1.5)
        elif self.method == "chi2":
            self.df = self.options.get("df", 2)
            self.loc = self.options.get("loc", 0)
            self.scale = self.options.get("scale", 0.5)
        else:
            raise ("Invalid method {}".format(self.method))

    def fit(self, X):
        if self.method == "KDE":
            self.kde.fit(X)
        elif self.method == "histogram":
            raise ("Not implemented method {}".format(self.method))
        elif self.method == "gamma":
            self.fit_k, self.fit_loc, self.fit_theta = stats.gamma.fit(X, self.k, loc=self.loc, scale=self.theta)
        elif self.method == "chi2":
            self.fit_df, self.fit_loc, self.fit_scale = stats.chi2.fit(X, self.df, loc=self.loc, scale=self.scale)
            pass
        else:
            raise ("Invalid method {}".format(self.method))

    def score(self, x):
        if self.method == "KDE":
            return self.kde.score_samples(x)
        elif self.method == "gamma":
            return 1 - stats.gamma.cdf(x, self.fit_k, self.fit_loc, self.fit_theta)
        elif self.method == "chi2":
            return 1 - stats.chi2.cdf(x, self.fit_df, self.fit_loc, self.fit_scale)
        elif self.method == "histogram":
            raise ("Not implemented method {}".format(self.method))
        else:
            raise ("Invalid method {}".format(self.method))
    
    transform = score

    def get_fit_params_string(self):
        if self.method == "KDE":
            return ""
        elif self.method == "gamma":
            return "fit_k %3.1f fit_loc %3.1f fit_theta %3.1f" % (self.fit_k, self.fit_loc, self.fit_theta)
        elif self.method == "chi2":
            return "fit_df %s, fit_loc %s, fit_scale %s" % (self.fit_df, self.fit_loc, self.fit_scale)
        elif self.method == "histogram":
            return ""
        else:
            raise ("Invalid method {}".format(self.method))

    def get_fit_params(self):
        if self.method == "KDE":
            return ""
        elif self.method == "gamma":
            return self.fit_k, self.fit_loc, self.fit_theta
        elif self.method == "chi2":
            return self.fit_df, self.fit_loc, self.fit_scale
        elif self.method == "histogram":
            return ""
        else:
            raise ("Invalid method {}".format(self.method))

def score_norm (x, range=1, alpha=1, beta=0):
    return range/(1+np.exp(-alpha*x+beta))

def score_dataset(mask_root, score_vals, metadata, max_clip=None, scene_id=None, save_results=False, directory="results/chad_score_pose_no_vis_mse3/", seg_len=24):
    gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores(mask_root, score_vals, metadata, max_clip, scene_id, save_results=save_results, directory=directory, seg_len=seg_len)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    # scores_np = scores_arr
    auc, auc_pr, eer, eer_th = score_align(scores_np, gt_np, seg_len=seg_len) # fix the seg_len 
    return auc, auc_pr, eer, eer_th

def get_dataset_scores(mask_root, scores, metadata, max_clip=None, scene_id=None, save_results=False, directory="results/chad_score_pose_no_vis_mse3/", seg_len=24):
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    
    # directory = "results_ae_c2/"
    if save_results:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    # per_frame_scores_root = '/home/galinezh/anomaly_baseline/simple_ae/chad_data/data/c1/testing/test_frame_mask' # change this based on dir o masks 
    # TODO change it from config file
    per_frame_scores_root = mask_root  
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    for clip in clip_list:
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        scene_id, clip_id = [int(i) for i in clip.split('.')[0].split('_')[:2]]
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        clip_fig_idxs = set([arr[2] for arr in clip_metadata]) # finding the ids of the people in the clip
        scores_zeros = np.zeros(clip_gt.shape[0])
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs} # creating score for each person as the size of the whole clip
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_scores = scores[person_metadata_inds]
            pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])
            clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores
    
        if len (list(clip_person_scores_dict.values())) == 0:
            print ("found you motherfucker!")
            continue
        clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
        clip_score = np.amax(clip_ppl_score_arr, axis=0)
        fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
        
        if save_results:
            with open(directory+clip.split(".")[0]+".csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(zip(clip_gt, clip_score))
                
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr


def score_align(scores_np, gt, seg_len=30, sigma=40):
    scores_shifted = np.zeros_like(scores_np)
    shift = seg_len + (seg_len // 2) - 1
    scores_shifted[shift:] = scores_np[:-shift]
    scores_smoothed = gaussian_filter1d(scores_np, sigma)
    # scores_smoothed = scores_np
    auc_roc = roc_auc_score(gt, scores_smoothed)
    precision, recall, thresholds = precision_recall_curve(gt, scores_smoothed)
    
    auc_precision_recall = auc(recall, precision)

    fpr, tpr, threshold = roc_curve(gt, scores_smoothed, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return auc_roc, auc_precision_recall, EER, eer_threshold



def select_scaler_model(scaler_name):
    if scaler_name == 'standard':
        return StandardScaler()
    elif scaler_name == 'robust':
        return RobustScaler(quantile_range=(0.00, 50.0))
    elif scaler_name == 'quantile':
        return QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=42)
    elif scaler_name == 'max_abs':
        return MaxAbsScaler()
    elif scaler_name == 'min_max':
        return MinMaxScaler()
    elif scaler_name == 'kde':
        return ScoreNormalization(method='KDE')
    elif scaler_name == 'gamma':
        return ScoreNormalization(method='gamma')
    elif scaler_name == 'chi2':
        return ScoreNormalization(method='chi2')
    else:
        raise ValueError('Unknown scaler. Please select one of: standard, robust, quantile, max_abs, min_max, '
                         'kde, gamma, chi2.')
    
    return None
