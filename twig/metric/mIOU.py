from nest import export
from mmengine.evaluator import BaseMetric
from typing import Optional
import torch
import torch.nn.functional as F

@export
class meanIntersectionOverUnion(BaseMetric):
    """mIOU Evaluation"""

    default_prefix = 'COD' 

    def __init__(self, collect_device: str = 'cpu', prefix: Optional[str] = None,
        data_range: Optional[float] = 1.0):
        super().__init__(collect_device, prefix)

    def confusion_matrix(self, input, target, num_classes):
        """
        input: torch.LongTensor:(N, H, W)
        target: torch.LongTensor:(N, H, W)
        num_classes: int
        results:Tensor
        """
        assert torch.max(input) < num_classes, f'{torch.max(input)}'
        assert torch.max(target) < num_classes, f'{torch.max(target)}'
        H, W = target.size()[-2:]
        results = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for i, j in zip(target.flatten(), input.flatten()):
            results[i, j] += 1
        return results

    def mean_iou(self, input, target):
        """
        input: torch.FloatTensor:(N, C, H, W)
        target: torch.LongTensor:(N, H, W)
        return: Tensor
        """
        assert len(input.size()) == 4
        assert len(target.size()) == 4
        target = target.squeeze(1)
        N, num_classes, H, W = input.size()
        input = F.softmax(input, dim=1)
        arg_max = torch.argmax(input, dim=1).to(torch.long)
        target = target*255
        target[target>num_classes-1]=num_classes-1
        target = target.to(torch.long)
        result = 0
        confuse_matrix = self.confusion_matrix(arg_max, target, num_classes)
        for i in range(num_classes):
            nii = confuse_matrix[i, i]
            # consider the case where the denominator is zero.
            if nii == 0:
                continue
            else:
                ti, tj = torch.sum(confuse_matrix[i, :]), torch.sum(confuse_matrix[:, i])
                result += (nii / (ti + tj - nii))

        return result / num_classes

    
    def process(self, data_batch, data_samples):
        with torch.no_grad():
            pred, gt = data_samples  
            pred = pred.to(device=self.collect_device)
            gt = gt.to(device=self.collect_device)         
            miou = self.mean_iou(pred, gt)
        self.results.append({'miou': miou})

    def compute_metrics(self, results: list) -> dict:
        mean_miou = sum(r['miou'] for r in results) / len(results)
        return dict(mIOU=mean_miou)