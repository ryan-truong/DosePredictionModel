import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from myLightningUtils import prepare_batch
from custom_loss import Loss
import torchio as tio
import lightning as L
import matplotlib.pyplot as plt
import numpy as np




class SingleConv(L.LightningModule):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation =1):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, groups=in_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True, dilation =1),
            # nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True, dilation =1),
            nn.InstanceNorm3d(in_ch, affine=True),
            # nn.BatchNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, stride=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            # nn.BatchNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.single_conv(x)


class UpConv(L.LightningModule):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, groups=in_ch, kernel_size=3, padding=1, stride=1, bias=True, dilation =1),
            # nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True, dilation =1),
            nn.InstanceNorm3d(in_ch, affine=True),
            # nn.BatchNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, stride=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            # nn.BatchNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.conv(x)
        return x


class Encoder(L.LightningModule):
    def __init__(self, in_ch, list_ch, dropout_prob):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding='same', dilation = 3),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding='same', dilation = 2),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.encoder_2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1), # 128 -> 64
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding="same", dilation = 2),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.encoder_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1), # 64 -> 32
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding='same', dilation = 2),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.encoder_4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1), # 32 -> 16
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.encoder_5 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1), # 16 -> 8
            SingleConv(list_ch[4], list_ch[5], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[5], list_ch[5], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation='gelu', batch_first=True, norm_first=True, )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=12, norm=nn.LayerNorm(512))
        self.positional_embedding = nn.Parameter(torch.randn(1, list_ch[5],512), requires_grad=True)



    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)

        # flatten the output (B, C, H, W, D) -> (B, C, N) where N = H*W*D = 512 and C = list_ch[5]
        out_encoder_5 = out_encoder_5.flatten(2)

        # add positional encoding

        out_encoder_5 = out_encoder_5 + self.positional_embedding

        # transformer encoder
        out_encoder_5 = self.transformer_encoder(out_encoder_5)
        # reshape back to (B, C, H, W, D)
        out_encoder_5 = out_encoder_5.reshape(out_encoder_5.shape[0], out_encoder_5.shape[1], 8, 8, 8)






        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5]



class Decoder(L.LightningModule):
    def __init__(self, list_ch, dropout_prob):
        super(Decoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])
        self.decoder_conv_4 = nn.Sequential(
            SingleConv(2 * list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.upconv_3 = UpConv(list_ch[4], list_ch[3])
        self.decoder_conv_3 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.upconv_2 = UpConv(list_ch[3], list_ch[2])
        self.decoder_conv_2 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout3d(p=dropout_prob)
        )
        self.upconv_1 = UpConv(list_ch[2], list_ch[1])
        self.decoder_conv_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5 = out_encoder

        out_decoder_4 = self.decoder_conv_4(
            torch.cat((self.upconv_4(out_encoder_5), out_encoder_4), dim=1)
        )
        out_decoder_3 = self.decoder_conv_3(
            torch.cat((self.upconv_3(out_decoder_4), out_encoder_3), dim=1)
        )
        out_decoder_2 = self.decoder_conv_2(
            torch.cat((self.upconv_2(out_decoder_3), out_encoder_2), dim=1)
        )
        out_decoder_1 = self.decoder_conv_1(
            torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1)
        )

        return out_decoder_1


class BaseUNet(L.LightningModule):
    def __init__(self, in_ch, list_ch, dropout_prob):
        super(BaseUNet, self).__init__()
        self.encoder = Encoder(in_ch, list_ch, dropout_prob)
        self.decoder = Decoder(list_ch, dropout_prob)
        # self.pool = nn.AdaptiveAvgPool3d((64,64,64))


        # init
        self.initialize()

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)


    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.encoder.modules)

    def forward(self, x):
        # x = self.pool(x)

        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)

        # Output is a list: [Output]
        return out_decoder


class Model(L.LightningModule):
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B, dropout_prob =0.0, train_batch_size=1, val_batch_size=1, val_plot_list = None):
    # def __init__(self, in_ch, out_ch, list_ch_A,dropout_prob =0.0):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.plot_list = val_plot_list
        # overall combined
        self.training_step_outputs = []
        self.validation_step_outputs = []





        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = BaseUNet(in_ch, list_ch_A, dropout_prob)
        self.net_B = BaseUNet(in_ch + list_ch_A[1], list_ch_B, dropout_prob)

        self.conv_out_A = nn.Sequential(
                        SingleConv(list_ch_A[1], list_ch_A[1]//2, kernel_size=3, stride=1, padding=1),
                        SingleConv(list_ch_A[1]//2, list_ch_A[1]//4, kernel_size=3, stride=1, padding=1),
                        SingleConv(list_ch_A[1]//4, out_ch, kernel_size=1, stride=1, padding=0),
                        nn.ReLU()
                        )
        self.conv_out_B = nn.Sequential(
                        SingleConv(list_ch_B[1], list_ch_B[1]//2, kernel_size=3, stride=1, padding=1),
                        SingleConv(list_ch_B[1]//2, list_ch_B[1]//4, kernel_size=3, stride=1, padding=1),
                        SingleConv(list_ch_B[1]//4, out_ch, kernel_size=1, stride=1, padding=0),
                        nn.ReLU()
                        )
        self.final_relu = nn.ReLU()
        self.custom_loss = Loss()




    def forward(self, x):
        out_net_A = self.net_A(x) ## 64,64,64

        output_A = self.conv_out_A(out_net_A)
        output_A = self.final_relu(output_A)

        out_net_B = self.net_B(torch.cat((out_net_A, x), dim=1))

        output_B = self.conv_out_B(out_net_B)
        output_B = self.final_relu(output_B)

        return [output_A, output_B]
        # return [output_A]
    def training_step(self, batch, batch_idx):
        inputs_, targets_ = prepare_batch(batch)
        outputs = self(inputs_) # the lightning module will automatically pass the inputs to forward function, could be called explicitly as well
        loss_dict = self.custom_loss(outputs, targets_, batch)
        # self.training_step_outputs.append(loss_dict["loss"])
        self.log('train_loss', loss_dict["loss"], batch_size=self.train_batch_size)
        self.log('train_mae_loss', loss_dict["mean_absolute_error"], batch_size=self.train_batch_size)
        self.log("train_mae_scaled", loss_dict["mean_absolute_error_scaled"], batch_size=self.train_batch_size)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'])

        return loss_dict["loss"]

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        '''
        This method is called before the optimizer step, can be used to check gradients.
        For debugging purposes, it can be used to check if gradients are None (i.e.
        unused parameters) which causes errors in data distributed training.

        '''
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)

    def validation_step(self, batch, batch_idx):
        inputs_, targets_ = prepare_batch(batch)
        outputs = self(inputs_)
        loss_dict = self.custom_loss(outputs, targets_, batch)
        # self.validation_step_outputs.append(loss_dict["loss"])
        self.log('val_loss', loss_dict["loss"], batch_size=self.val_batch_size, sync_dist=True)
        self.log('val_mae_loss', loss_dict["mean_absolute_error"], batch_size=self.val_batch_size, sync_dist=True)
        self.log("val_mae_scaled", loss_dict["mean_absolute_error_scaled"], batch_size=self.val_batch_size, sync_dist=True)


        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        inputs_, targets_ = prepare_batch(batch)
        outputs = self(inputs_)
        loss_dict = self.custom_loss(outputs, targets_, batch)
        self.validation_step_outputs.append(loss_dict["loss"])
        for key in loss_dict.keys():
            if key != "loss":
                # if loss_dict[key].size == torch.Size([]), make it torch.Size([1])
                if len(loss_dict[key].shape) == 0:
                    loss_dict[key] = loss_dict[key].unsqueeze(0)
                for i in range(inputs_.shape[0]):
                    self.log("test_combined_"+key, loss_dict[key][i])
                    self.log("test_" + batch["class"][i] + "_" + key, loss_dict[key][i])

        return loss_dict["loss"]
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs_, targets_ = prepare_batch(batch)
        outputsA, outputsB = self(inputs_)

        return outputsB

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
                    {'params': self.parameters(), 'lr': 1e-3},
                ],
                    weight_decay=1e-4,
                    amsgrad=False)
        return_dict = {'optimizer': optimizer}
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, verbose=True, threshold=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=1e-3,
                                                steps_per_epoch=78*4,
                                                epochs=5,
                                                pct_start=0.1,
                                                anneal_strategy='cos')
        return_dict['lr_scheduler'] = lr_scheduler
        # return_dict['monitor'] = 'val_loss'

        return return_dict
    def plotDoseSlice(self, dose, pred, patient_id):
        """
        This function finds the slice with the highest voxel value in the dose array and plots that slice for the dose, prediction, and the difference in one plot.
        The dose, and prediction are (H,W,D) tensors.
        """
        dose = dose.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        dose = dose.squeeze()
        pred = pred.squeeze()
        # dose = dose.transpose(1,2,0)
        # pred = pred.transpose(1,2,0)
        # find the max voxel index
        max_voxel_index = np.unravel_index(np.argmax(dose, axis=None), dose.shape)
        # get the slice with the max voxel
        dose = dose[max_voxel_index[0],:,:]
        pred = pred[max_voxel_index[0],:,:]
        # rotate 90 degrees clockwise
        dose = np.rot90(dose, k=3)
        pred = np.rot90(pred, k=3)
        # plot the dose, prediction, and difference in one plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,5))
        fig.suptitle(patient_id.replace("($)","_"))
        ax1.imshow(dose, vmin=0, vmax=1.5, cmap='jet')
        ax1.set_title("Dose")
        ax2.imshow(pred, vmin=0, vmax=1.5, cmap='jet')
        ax2.set_title("Prediction")
        ax3.imshow(dose-pred, vmin=-0.5, vmax=0.5, cmap='bwr')
        ax3.set_title("Difference")
        fig.colorbar(ax1.imshow(dose, vmin=0, vmax=1.5, cmap='jet'), ax=ax1)
        fig.colorbar(ax2.imshow(pred, vmin=0, vmax=1.5, cmap='jet'), ax=ax2)
        fig.colorbar(ax3.imshow(dose-pred, vmin=-0.5, vmax=0.5, cmap='bwr'), ax=ax3)
        # log the plot
        self.logger.experiment['images/DoseDistribution_'+patient_id.replace("($)","_")].append(fig)
        plt.close(fig)
        plt.clf()
    def plotIsoDSC(self, dose, pred, patient_id):
        """
        Plots the thresholded dose and prediction for a given patient.
        all plotted on the same plot.
        """
        dose = dose.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        dose = dose.squeeze()
        pred = pred.squeeze()
        max_voxel_index = np.unravel_index(np.argmax(dose, axis=None), dose.shape)
        dose = dose[max_voxel_index[0],:,:]
        pred = pred[max_voxel_index[0],:,:]
        dose = np.rot90(dose, k=3)
        pred = np.rot90(pred, k=3)
        # create a 2,2 plot
        fig, ax = plt.subplots(2, 2,figsize=(15,15))
        fig.suptitle(patient_id.replace("($)","_"))
        threshold_values = [0.8, 1.0, 1.25, 1.5]
        for i in range(len(threshold_values)):
            # threshold the dose and prediction
            dose_thresholded = np.where(dose>threshold_values[i], 1, 0)
            pred_thresholded = np.where(pred>threshold_values[i], 1, 0)
            # plot the dose and prediction
            ax[i//2,i%2].imshow(dose_thresholded, vmin=0, vmax=1, cmap='gray')
            ax[i//2,i%2].imshow(pred_thresholded, vmin=0, vmax=1, cmap='jet', alpha=0.5)
            ax[i//2,i%2].set_title("Threshold: " + str(threshold_values[i]))
            ax[i//2,i%2].axis('off')
        # log the plot
        self.logger.experiment['images/IsoDSC_'+patient_id.replace("($)","_")].append(fig)
        plt.close(fig)
        plt.clf()
