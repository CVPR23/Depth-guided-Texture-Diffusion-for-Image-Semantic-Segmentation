# from py_sod_metrics import WeightedFmeasure as weightedfmeasure
# from nest import export
# from mmengine.evaluator import BaseMetric
# from typing import Optional
# import torch
# import numpy as np

# @export
# class WeightedFmeasure(BaseMetric):
#     """WeightedFmeasure Evaluation"""

#     default_prefix = 'COD'    

#     def __init__(self, collect_device: str = 'cpu', prefix: Optional[str] = None,
#         data_range: Optional[float] = 1.0):
#         super().__init__(collect_device, prefix)
#         self.evaluator = weightedfmeasure()
#     def process(self, data_batch, data_samples):
#         with torch.no_grad():
#             pred, gt = data_samples
#             pred = pred.to(device=self.collect_device).numpy()
#             gt = gt.to(device=self.collect_device).numpy()
#             pred = pred.squeeze(1)
#             gt = gt.squeeze(1)
#             pred = (pred * 255).astype(np.uint8)
#             gt = (gt * 255).astype(np.uint8)
#             assert pred.ndim == gt.ndim and pred.shape == gt.shape, (pred.shape, gt.shape)
#             assert pred.dtype == np.uint8, pred.dtype
#             assert gt.dtype == np.uint8, gt.dtype            
#             for x, y in zip(pred, gt):          
#                 self.evaluator.step(pred=x, gt=y)
#             wfm = self.evaluator.get_results()["wfm"]
#         self.results.append({'wfm': wfm})

#     def compute_metrics(self, results: list) -> dict:
#         mean_wfm = sum(r['wfm'] for r in results) / len(results)
#         return dict(WeightedFmeasure=mean_wfm)