
import os
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils

from Utils.loss import l2_loss

from Models.GANomaly.networks_mvtec import NetD, NetG, weights_init


class GANomaly():
    def __init__(self, opt):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.log_path = opt.log_path
        self.start_epoch = opt.start_epoch
        self.finish_epoch = opt.finish_epoch
        self.cur_epoch = self.start_epoch
        self.cur_losses = {
            'gen_epoch_loss': float('inf'),
            'disc_epoch_loss': float('inf'),
            'adv_loss': float('inf'),
            'con_loss': float('inf'),
            'lat_loss': float('inf')
        }

        ## model
        self.gen = NetG(imageSize=opt.imageSize_h,
                        nz=opt.nz,
                        nc=opt.nc,
                        ngf=opt.nz,
                        ngpu=opt.ngf
                        ).to(self.device)
        self.disc = NetD(imageSize=opt.imageSize_h,
                        nc=opt.nc,
                        ngf=opt.nz,
                        ngpu=opt.ngf,
                        ).to(self.device)
        
        self.gen.apply(weights_init)
        self.disc.apply(weights_init)
        
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.disc_optimizer = optim.Adam(self.disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        
        try:
            self.load_checkpoint()
            print(self.cur_losses)
            print("Checpoint have been loaded.")
        except:
            print("Checkpoint load fail.")

               
        self.w_adv = opt.w_adv
        self.w_con = opt.w_con
        self.w_lat = opt.w_lat
            
        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_lat = l2_loss
        self.l_bce = nn.BCELoss()

        self.gen_loss_all_epoch = []
        self.disc_loss_all_epoch = []
        
    def save_checkpoint(self, is_best=False):
        if is_best:
            path = os.path.join(self.opt.checkpoint_save, f'checkpoint_best.pt')
        else:
            path = os.path.join(self.opt.checkpoint_save, f'checkpoint_{self.cur_epoch}.pt')

        torch.save({
            'epoch': self.cur_epoch,
            'gen_state_dict': self.gen.state_dict(),
            'disc_state_dict': self.disc.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'dictionary_losses': self.cur_losses
            }, path)
        print(f"Checkpoint saved successfully at epoch {self.cur_epoch}")
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.checkpoint_load, map_location=self.device)
        self.gen.load_state_dict(checkpoint['gen_state_dict'])
        self.disc.load_state_dict(checkpoint['disc_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_losses = checkpoint['dictionary_losses']
    
    def train(self, train_dataloader, test_neg_loader, test_pos_loader):
        ## Training
        record = 0
        best_auc = 0
        traindata_size = len(train_dataloader.dataset)
        with open(self.log_path, "a") as log_file:
            log_file.write("Training started\n")

        
        with tqdm(range(self.start_epoch, self.finish_epoch + 1)) as t:
            for self.cur_epoch in t:

                t.set_description(f"Epoch {self.cur_epoch} /{self.finish_epoch}")
                record = 0
                self.gen.train()
                self.disc.train()
                L_adv_epoch_loss = 0.0
                L_con_epoch_loss = 0.0
                L_lat_epoch_loss = 0.0
                gen_epoch_loss = 0.0
                disc_epoch_loss = 0.0
                
                for inputs in train_dataloader:
                    batchSize = inputs.size(0)
                    inputs = inputs.to(self.device)

                    # forward_g
                    # self.forward_g()
                    fakes, latent_i, latent_o = self.gen(inputs)

                    # forward_d
                    # self.forward_d()
                    pred_real, feat_real = self.disc(inputs)
                    pred_fake, feat_fake = self.disc(fakes.detach())


                    # Backward-pass
                    # # netg
                    self.gen_optimizer.zero_grad()
                    # self.backward_g()
                    err_g_adv = self.w_adv * self.l_adv(feat_fake, feat_real)
                    err_g_con = self.w_con * self.l_con(fakes, inputs)
                    err_g_lat = self.w_lat * self.l_lat(latent_o, latent_i)
                    err_g = err_g_adv + err_g_con + err_g_lat
                    err_g.backward(retain_graph=True)
                    self.gen_optimizer.step()
                    L_adv_epoch_loss += err_g_adv.item() * batchSize
                    L_con_epoch_loss += err_g_con.item() * batchSize
                    L_lat_epoch_loss += err_g_lat.item() * batchSize
                    gen_epoch_loss += err_g.item() * batchSize

                    # # netd
                    self.disc_optimizer.zero_grad()
                    # self.backward_d()
                    real_label = torch.ones (size=(batchSize,), dtype=torch.float32, device=self.device)
                    fake_label = torch.zeros(size=(batchSize,), dtype=torch.float32, device=self.device)
                    err_d_real = self.l_bce(pred_real, real_label)
                    err_d_fake = self.l_bce(pred_fake, fake_label)
                    err_d = (err_d_real + err_d_fake) * 0.5
                    err_d.backward()
                    self.disc_optimizer.step()
                    disc_epoch_loss += err_d.item() * batchSize
                    if err_d.item() < self.opt.reset_disc_value: 
                        self.disc.apply(weights_init)
                        print('Reloading d net')

                    ## record results
                    if self.cur_epoch % self.opt.sample_interval == 0 and record == 0:
                    # outputs.data = outputs.data.mul(0.5).add(0.5)
                        vutils.save_image(fakes[:24].view(-1, self.opt.nc, self.opt.imageSize_h, self.opt.imageSize_w),
                                          '{0}/outputs_{1}.png'.format(self.opt.experiment_path, self.cur_epoch), normalize=True)
                        vutils.save_image(inputs[:24].view(-1, self.opt.nc, self.opt.imageSize_h, self.opt.imageSize_w),
                                          '{0}/inputs_{1}.png'.format(self.opt.experiment_path, self.cur_epoch), normalize=True)
                    record += 1

                ## End of epoch
                L_adv_epoch_loss /= traindata_size
                L_con_epoch_loss /= traindata_size
                L_lat_epoch_loss /= traindata_size
                gen_epoch_loss /= traindata_size
                disc_epoch_loss /= traindata_size


                t.set_postfix(L_adv=L_adv_epoch_loss, L_con=L_con_epoch_loss, L_lat=L_lat_epoch_loss,
                              gen_loss=gen_epoch_loss, disc_loss=disc_epoch_loss)
               
                print("gen_epoch_loss ", gen_epoch_loss, self.cur_epoch)
                print("disc_epoch_loss", disc_epoch_loss, self.cur_epoch)

                auc, _, _, _, _ = self.test(test_neg_loader, test_pos_loader)
                # print(auc)

                text = f'{self.cur_epoch}:   {auc}'
                with open(self.log_path, "a") as log_file:
                    log_file.write('%s\n' % text)

                if auc > best_auc:
                    best_auc = auc
                    self.save_checkpoint(is_best=True)
                    with open(self.log_path, "a") as log_file:
                        log_file.write('-\n')

                self.cur_losses = {
                    'gen_epoch_loss': gen_epoch_loss,
                    'disc_epoch_loss': disc_epoch_loss,
                    'adv_loss': L_adv_epoch_loss,
                    'con_loss': L_con_epoch_loss,
                    'lat_loss': L_lat_epoch_loss
                }
                
                if (self.cur_epoch) % 100 == 0:
                    self.save_checkpoint()
    
    def test(self, test_neg_loader, test_pos_loader):
        self.gen.eval()
        self.disc.eval()
        with torch.no_grad():

            list_errors = torch.tensor([]).to(self.device)
            list_labels = torch.tensor([]).to(self.device)
            list_inputs = torch.tensor([]).to(self.device)
            list_outputs = torch.tensor([]).to(self.device)

            for inputs in test_neg_loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(self.device)
                labels = torch.zeros(batch_size).to(self.device)
                outputs, latent_in, latent_out = self.gen(inputs)

                error_latents = torch.mean(torch.pow((latent_in - latent_out), 2), dim=[1, 2, 3])
                error_images = torch.mean(torch.pow((inputs - outputs), 2), dim=[1, 2, 3])
                
                _, feature_fake = self.disc(inputs)
                _, feature_real = self.disc(outputs)
                adv_loss =  torch.mean(nn.functional.mse_loss(feature_fake, feature_real.detach(), reduction='none'),dim=[1, 2, 3])
                con_loss = torch.mean(nn.functional.l1_loss(outputs, inputs, reduction='none'), dim=[1, 2, 3])
                lat_loss = torch.mean(nn.functional.mse_loss(latent_out, latent_in, reduction='none'), dim=[1, 2, 3])
                total_loss = self.w_adv * adv_loss + \
                            self.w_con * con_loss + \
                            self.w_lat * lat_loss
                
                error = error_latents

                list_errors = torch.cat((list_errors, error.to(self.device)), dim=0)
                list_labels = torch.cat((list_labels, labels.to(self.device)), dim=0)
                list_inputs = torch.cat((list_inputs, inputs.to(self.device)), dim=0)
                list_outputs = torch.cat((list_outputs, outputs.to(self.device)), dim=0)


            for inputs in test_pos_loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(self.device)
                labels = torch.ones(batch_size).to(self.device)
                outputs, latent_in, latent_out = self.gen(inputs)

                error_latents = torch.mean(torch.pow((latent_in - latent_out), 2), dim=[1, 2, 3])
                error_images = torch.mean(torch.pow((inputs - outputs), 2), dim=[1, 2, 3])
                
                _, feature_fake = self.disc(inputs)
                _, feature_real = self.disc(outputs)
                adv_loss =  torch.mean(nn.functional.mse_loss(feature_fake, feature_real.detach(), reduction='none'),dim=[1, 2, 3])
                con_loss = torch.mean(nn.functional.l1_loss(outputs, inputs, reduction='none'), dim=[1, 2, 3])
                lat_loss = torch.mean(nn.functional.mse_loss(latent_out, latent_in, reduction='none'), dim=[1, 2, 3])
                total_loss = self.w_adv * adv_loss + \
                            self.w_con * con_loss + \
                            self.w_lat * lat_loss
                
                error = error_latents
                
                list_errors = torch.cat((list_errors, error.to(self.device)), dim=0)
                list_labels = torch.cat((list_labels, labels.to(self.device)), dim=0)
                list_inputs = torch.cat((list_inputs, inputs.to(self.device)), dim=0)
                list_outputs = torch.cat((list_outputs, outputs.to(self.device)), dim=0)

            # Scale error vector between [0, 1]
            list_scores = (list_errors - torch.min(list_errors)) / (torch.max(list_errors) - torch.min(list_errors))
            list_scores = list_scores.reshape(list_scores.size(0)).cpu().numpy()
            list_labels = list_labels.cpu().numpy()
            roc_auc = roc_auc_score(list_labels, list_scores)

            return roc_auc, \
                    list_scores, \
                    list_labels, \
                    list_inputs.cpu().numpy(), \
                    list_outputs.cpu().numpy() 