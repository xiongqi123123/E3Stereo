"""
边缘检测评估指标：ODS / OIS。

ODS (Optimal Dataset Scale): 全数据集使用单一最优阈值时的 F1
OIS (Optimal Image Scale): 每张图各自最优阈值时的 F1 均值

基于距离阈值的匹配：预测边缘像素若在 GT 边缘的 dist_thresh 像素内则视为正确匹配。
默认 dist_thresh = max(2, 0.0075 * image_diagonal)，与 BSDS 惯例一致。
"""
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def _precision_recall_f1(pred_bin, gt_bin, dist_thresh_px):
    """
    基于距离阈值的 precision / recall / F1。
    pred_bin, gt_bin: [H,W] 二值 0/1
    dist_thresh_px: 匹配距离阈值（像素）
    """
    pred_bin = pred_bin.astype(np.uint8)
    gt_bin = gt_bin.astype(np.uint8)
    n_pred = pred_bin.sum()
    n_gt = gt_bin.sum()

    if n_pred == 0:
        prec = 1.0 if n_gt == 0 else 0.0
        rec = 1.0 if n_gt == 0 else 0.0
    elif n_gt == 0:
        prec = 0.0
        rec = 1.0
    else:
        # 到最近 GT 边缘的距离（背景=1 的区域做 EDT，再取反得到到前景的距离）
        dist_to_gt = distance_transform_edt(1 - gt_bin)
        dist_to_pred = distance_transform_edt(1 - pred_bin)
        # Precision: 预测边缘中有多少落在 GT 的 dist_thresh 内
        tp_prec = (pred_bin & (dist_to_gt <= dist_thresh_px)).sum()
        prec = float(tp_prec) / float(n_pred)
        # Recall: GT 边缘中有多少被预测边缘的 dist_thresh 内覆盖
        tp_rec = (gt_bin & (dist_to_pred <= dist_thresh_px)).sum()
        rec = float(tp_rec) / float(n_gt)

    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1


def ods_ois_single_image(pred_prob, gt_binary, dist_thresh_px, thresh_list=None):
    """
    单张图的 ODS/OIS 相关统计。
    pred_prob: [H,W] float [0,1]
    gt_binary: [H,W] 0/1 或 [0,1] 连续值（>0.5 视为边缘）
    dist_thresh_px: 距离阈值（像素）
    thresh_list: 用于扫阈值的列表，默认 0.01~0.99 步长 0.01

    Returns:
        best_f1_ois: 该图的最优 F1（OIS 用）
        best_thresh_ois: 该图的最优阈值
        curve: list of (thresh, prec, rec, f1) 用于聚合 ODS
    """
    gt_bin = (np.asarray(gt_binary, dtype=np.float32) > 0.5).astype(np.uint8)
    pred_prob = np.asarray(pred_prob, dtype=np.float32)
    if pred_prob.shape != gt_bin.shape:
        pred_prob = cv2.resize(
            pred_prob, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

    if thresh_list is None:
        thresh_list = np.linspace(0.01, 0.99, 99)

    curve = []
    best_f1 = 0.0
    best_thresh = 0.5
    for t in thresh_list:
        pred_bin = (pred_prob > t).astype(np.uint8)
        prec, rec, f1 = _precision_recall_f1(pred_bin, gt_bin, dist_thresh_px)
        curve.append((float(t), prec, rec, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_f1, best_thresh, curve


def compute_ods_ois(pred_list, gt_list, dist_thresh_frac=0.0075, thresh_list=None):
    """
    多张图聚合计算 ODS / OIS。
    pred_list: list of [H,W] float [0,1]
    gt_list: list of [H,W] 0/1 或 [0,1]
    dist_thresh_frac: 距离阈值 = max(2, dist_thresh_frac * image_diagonal)

    Returns:
        ods: float
        ois: float
        ods_thresh: ODS 对应的最优阈值
    """
    if thresh_list is None:
        thresh_list = np.linspace(0.01, 0.99, 99)

    n = len(pred_list)
    assert n == len(gt_list), "pred_list and gt_list length mismatch"

    # 按阈值聚合：每个阈值下，所有图的 (prec, rec, f1)
    thresh_to_f1 = {t: [] for t in thresh_list}
    ois_scores = []

    for pred, gt in zip(pred_list, gt_list):
        h, w = gt.shape[:2]
        diag = np.sqrt(h * h + w * w)
        dist_px = max(2, int(round(dist_thresh_frac * diag)))

        best_f1, _, curve = ods_ois_single_image(pred, gt, dist_px, thresh_list)
        ois_scores.append(best_f1)

        for (t, prec, rec, f1) in curve:
            thresh_to_f1[t].append(f1)

    # OIS: 每图最优 F1 的均值
    ois = float(np.mean(ois_scores))

    # ODS: 全数据集单一最优阈值
    best_ods = 0.0
    best_ods_thresh = 0.5
    for t in thresh_list:
        avg_f1 = np.mean(thresh_to_f1[t])
        if avg_f1 > best_ods:
            best_ods = avg_f1
            best_ods_thresh = t

    return float(best_ods), ois, best_ods_thresh
