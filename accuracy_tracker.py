import torch


class AccuracyTracker():
    """
    Compute point-wise accuracy between network output and target
    """
    def __init__(self):
        self.num_total = 0.0
        self.num_TP = 0.0
        self.TP_pos = 0.0
        self.TP_neg = 0.0
        self.tot_pos = 0.0
        self.tot_neg = 0.0

    def onNewBatch(self, cls_out, target_out):
        """

        :param cls_out: (bz,1,grid_size[0],grid_size[1])}
        :param target_out: (bz,1,grid_size[0],grid_size[1])}
        :return:
        """
        assert cls_out.shape == target_out.shape
        assert cls_out.shape[1] == 1

        cls_out_bin = (cls_out > 0.5) * 1.0
        # only count positive and negative, target_out only contains -1, 0, 1
        gt_mask = target_out != -1
        result = (cls_out_bin[gt_mask].float() == target_out[gt_mask].float())        # torch.tensor dtype=torch.bool
        assert result.dim() == 1
        num_TP = torch.sum(result).item()
        self.num_TP = self.num_TP + num_TP
        self.num_total = self.num_total + result.shape[0]

        self.TP_pos = self.TP_pos + torch.sum(result*(target_out[gt_mask] == 1))
        self.TP_neg = self.TP_pos + torch.sum(result * (target_out[gt_mask] == 0))
        self.tot_pos = self.tot_pos + torch.sum(target_out[gt_mask] == 1)
        self.tot_neg = self.tot_neg + torch.sum(target_out[gt_mask] == 0)

    def getMetric(self):
        """
        Return the accuracy metric
        :return:
        """
        acc = self.num_TP / self.num_total
        return acc

    # def clear(self):
    #     self.num_total = 0.0
    #     self.num_TP = 0.0
