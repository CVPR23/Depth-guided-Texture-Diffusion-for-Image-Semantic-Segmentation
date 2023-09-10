from py_sod_metrics import Emeasure as emeasure
from nest import export
from mmengine.evaluator import BaseMetric
from typing import Optional
import torch
import numpy as np

@export
class Emeasure(BaseMetric):
    """Emeasure Evaluation"""

    default_prefix = 'COD'    

    def __init__(self, collect_device: str = 'cpu', prefix: Optional[str] = None):
        super().__init__(collect_device, prefix)
        self.evaluator = emeasure()
    def process(self, data_batch, data_samples):
        with torch.no_grad():
            pred, gt = data_samples
            pred = pred.to(device=self.collect_device).numpy()
            gt = gt.to(device=self.collect_device).numpy() 
            pred = pred.squeeze(1)
            gt = gt.squeeze(1)
            pred = (pred * 255).astype(np.uint8)
            gt = (gt * 255).astype(np.uint8)
            assert pred.ndim == gt.ndim and pred.shape == gt.shape, (pred.shape, gt.shape)
            assert pred.dtype == np.uint8, pred.dtype
            assert gt.dtype == np.uint8, gt.dtype
            for x, y in zip(pred, gt):          
                self.evaluator.step(pred=x, gt=y)
            em = self.evaluator.get_results()["em"]['adp']
        self.results.append({'em': em})

    def compute_metrics(self, results: list) -> dict:
        mean_em = sum(r['em'] for r in results) / len(results)
        return dict(Emeasure=mean_em)





# from nest import export
# import numpy as np
# from mmengine.evaluator import BaseMetric
# import torch
# from typing import Optional

# EPS = np.spacing(1)
# TYPE = np.float64

# def prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
#     """
#     A numpy-based function for preparing ``pred`` and ``gt``.
#     - for ``pred``, it looks like ``mapminmax(im2double(...))`` of matlab;
#     - ``gt`` will be binarized by 128.
#     :param pred: prediction
#     :param gt: mask
#     :return: pred, gt
#     """
#     pred = pred.cpu().numpy()
#     gt = gt.cpu().numpy()
#     gt = gt > 128
#     # im2double, mapminmax
#     pred = pred / 255
#     if pred.max() != pred.min():
#         pred = (pred - pred.min()) / (pred.max() - pred.min())
#     return pred, gt


# def get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
#     """
#     Return an adaptive threshold, which is equal to twice the mean of ``matrix``.
#     :param matrix: a data array
#     :param max_value: the upper limit of the threshold
#     :return: min(2 * matrix.mean(), max_value)
#     """
#     return min(2 * matrix.mean(), max_value)

# @export
# class Emeasure(BaseMetric):
#     '''Emeasure Evaluation'''

#     default_prefix = 'COD'
#     def __init__(self, collect_device: str = 'cpu', prefix: Optional[str] = None,
#         data_range: Optional[float] = 1.0):
#         super().__init__(collect_device, prefix)

#         # self.adaptive_ems = []
#         # self.changeable_ems = []

#     def process(self, data_batch, data_samples):
#         with torch.no_grad():
#             pred, gt = data_samples
#             pred, gt = prepare_data(pred=pred, gt=gt)
#             # 检查
#             assert pred.ndim == gt.ndim and pred.shape == gt.shape, (pred.shape, gt.shape)
#             assert pred.dtype == np.uint8, pred.dtype
#             assert gt.dtype == np.uint8, gt.dtype

#             self.gt_fg_numel = np.count_nonzero(gt)
#             self.gt_size = gt.shape[0] * gt.shape[1]

#             changeable_em = self.cal_changeable_em(pred, gt)
#             # self.changeable_ems.append(changeable_ems)
#             adaptive_em = self.cal_adaptive_em(pred, gt)
#             # self.adaptive_ems.append(adaptive_em)            
#         self.results.append({'adaptive_em': adaptive_em,
#                              'changeable_em': changeable_em})   

#     def compute_metrics(self, results: list) -> dict:
#         """
#         Return the results about E-measure.
#         :return: dict(em=dict(adp=adaptive_em, curve=changeable_em))
#         """
#         adaptive_em = sum(r['adaptive_em'] for r in results) / len(results)
#         # changeable_em = sum(r['changeable_em'] for r in results) / len(results)
#         return dict(adp=adaptive_em)

#     def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
#         """
#         Calculate the adaptive E-measure.
#         :return: adaptive_em
#         """
#         adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
#         adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
#         return adaptive_em

#     def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
#         """
#         Calculate the changeable E-measure, which can be used to obtain the mean E-measure,
#         the maximum E-measure and the E-measure-threshold curve.
#         :return: changeable_ems
#         """
#         changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
#         return changeable_ems

#     def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
#         """
#         Calculate the E-measure corresponding to the specific threshold.
#         Variable naming rules within the function:
#         ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``
#         If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
#         """
#         binarized_pred = pred >= threshold
#         fg_fg_numel = np.count_nonzero(binarized_pred & gt)
#         fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

#         fg___numel = fg_fg_numel + fg_bg_numel
#         bg___numel = self.gt_size - fg___numel

#         if self.gt_fg_numel == 0:
#             enhanced_matrix_sum = bg___numel
#         elif self.gt_fg_numel == self.gt_size:
#             enhanced_matrix_sum = fg___numel
#         else:
#             parts_numel, combinations = self.generate_parts_numel_combinations(
#                 fg_fg_numel=fg_fg_numel,
#                 fg_bg_numel=fg_bg_numel,
#                 pred_fg_numel=fg___numel,
#                 pred_bg_numel=bg___numel,
#             )

#             results_parts = []
#             for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
#                 align_matrix_value = (
#                     2
#                     * (combination[0] * combination[1])
#                     / (combination[0] ** 2 + combination[1] ** 2 + EPS)
#                 )
#                 enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
#                 results_parts.append(enhanced_matrix_value * part_numel)
#             enhanced_matrix_sum = sum(results_parts)

#         em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)
#         return em

#     def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
#         """
#         Calculate the E-measure corresponding to the threshold that varies from 0 to 255..
#         Variable naming rules within the function:
#         ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``
#         If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
#         """
#         pred = (pred * 255).astype(np.uint8)
#         bins = np.linspace(0, 256, 257)
#         fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
#         fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
#         fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
#         fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

#         fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
#         bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

#         if self.gt_fg_numel == 0:
#             enhanced_matrix_sum = bg___numel_w_thrs
#         elif self.gt_fg_numel == self.gt_size:
#             enhanced_matrix_sum = fg___numel_w_thrs
#         else:
#             parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
#                 fg_fg_numel=fg_fg_numel_w_thrs,
#                 fg_bg_numel=fg_bg_numel_w_thrs,
#                 pred_fg_numel=fg___numel_w_thrs,
#                 pred_bg_numel=bg___numel_w_thrs,
#             )

#             results_parts = np.empty(shape=(4, 256), dtype=np.float64)
#             for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
#                 align_matrix_value = (
#                     2
#                     * (combination[0] * combination[1])
#                     / (combination[0] ** 2 + combination[1] ** 2 + EPS)
#                 )
#                 enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
#                 results_parts[i] = enhanced_matrix_value * part_numel
#             enhanced_matrix_sum = results_parts.sum(axis=0)

#         em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)
#         return em

#     def generate_parts_numel_combinations(
#         self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel
#     ):
#         bg_fg_numel = self.gt_fg_numel - fg_fg_numel
#         bg_bg_numel = pred_bg_numel - bg_fg_numel

#         parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

#         mean_pred_value = pred_fg_numel / self.gt_size
#         mean_gt_value = self.gt_fg_numel / self.gt_size

#         demeaned_pred_fg_value = 1 - mean_pred_value
#         demeaned_pred_bg_value = 0 - mean_pred_value
#         demeaned_gt_fg_value = 1 - mean_gt_value
#         demeaned_gt_bg_value = 0 - mean_gt_value

#         combinations = [
#             (demeaned_pred_fg_value, demeaned_gt_fg_value),
#             (demeaned_pred_fg_value, demeaned_gt_bg_value),
#             (demeaned_pred_bg_value, demeaned_gt_fg_value),
#             (demeaned_pred_bg_value, demeaned_gt_bg_value),
#         ]
#         return parts_numel, combinations

