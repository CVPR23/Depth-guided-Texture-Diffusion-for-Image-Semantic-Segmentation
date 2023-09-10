from py_sod_metrics import Fmeasure as fmeasure
from nest import export
from mmengine.evaluator import BaseMetric
from typing import Optional
import torch
import numpy as np

@export
class Fmeasure(BaseMetric):
    """Fmeasure Evaluation"""

    default_prefix = 'COD'    

    def __init__(self, collect_device: str = 'cpu', prefix: Optional[str] = None,
        data_range: Optional[float] = 1.0):
        super().__init__(collect_device, prefix)
        self.evaluator = fmeasure()
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
            fm = self.evaluator.get_results()["fm"]['adp']
        self.results.append({'fm': fm})

    def compute_metrics(self, results: list) -> dict:
        mean_fm = sum(r['fm'] for r in results) / len(results)
        return dict(Fmeasure=mean_fm)
# for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
#     mask_root = './data/TestDataset/{}/GT'.format(_data_name)
#     pred_root = './results/{}/{}/'.format(method, _data_name)
#     mask_name_list = sorted(os.listdir(mask_root))
#     FM = Fmeasure()
#     WFM = WeightedFmeasure()
#     SM = Smeasure()
#     EM = Emeasure()
#     M = MAE()
#     for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
#         mask_path = os.path.join(mask_root, mask_name)
#         pred_path = os.path.join(pred_root, mask_name)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
#         FM.step(pred=pred, gt=mask)
#         WFM.step(pred=pred, gt=mask)
#         SM.step(pred=pred, gt=mask)
#         EM.step(pred=pred, gt=mask)
#         M.step(pred=pred, gt=mask)

#     fm = FM.get_results()["fm"]
#     wfm = WFM.get_results()["wfm"]
#     sm = SM.get_results()["sm"]
#     em = EM.get_results()["em"]
#     mae = M.get_results()["mae"]

#     results = {
#         "Smeasure": sm,
#         "wFmeasure": wfm,
#         "MAE": mae,
#         "adpEm": em["adp"],
#         "meanEm": em["curve"].mean(),
#         "maxEm": em["curve"].max(),
#         "adpFm": fm["adp"],
#         "meanFm": fm["curve"].mean(),
#         "maxFm": fm["curve"].max(),
#     }

#     print(results)
#     file=open("evalresults.txt", "a")
#     file.write(method+' '+_data_name+' '+str(results)+'\n')