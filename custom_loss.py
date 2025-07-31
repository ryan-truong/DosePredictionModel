# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch
import torchio as tio
import lightning as L


class Loss(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='none')
        self.smoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        self.MSELoss_func_no_reduce = nn.MSELoss(reduction='none')
        self.MSELoss_func = nn.MSELoss()


    def forward(self, pred, gt_dose, batch):
        return_dict = {}
        pred_A = pred[0]
        pred_B = pred[1]

        gt_dose_scaled = torch.where(gt_dose > 10, 10, gt_dose) /10



        MSE_loss_no_reduction = 0.5 * self.MSELoss_func_no_reduce(pred_A, gt_dose_scaled) + self.MSELoss_func_no_reduce(pred_B, gt_dose_scaled)
        weighted_loss = torch.mean(MSE_loss_no_reduction)




        return_dict['loss'] = weighted_loss
        L1_loss = nn.L1Loss()

        mean_error = self.mean_error(gt_dose,pred_B)

        return_dict['mean_error'] = mean_error
        return_dict['mean_absolute_error_scaled'] = L1_loss(gt_dose_scaled,pred_B)
        return_dict['mean_absolute_error'] = L1_loss(gt_dose*100,pred_B*1000)


        return return_dict
