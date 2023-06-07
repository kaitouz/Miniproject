import numpy as np
import os
import cv2
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from tqdm.notebook import tqdm


class ResNet18_MS3(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet18_MS3, self).__init__()     
        net = models.resnet18(pretrained=pretrained)
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res


class STFPM():
    def __init__(self, opt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_path = opt.log_path
        self.weight_path = opt.weight_path
        self.checkpoint_save_path = opt.checkpoint_save
        self.checkpoint_load_path = opt.checkpoint_load

        self.lossSize_h = opt.lossSize_h
        self.lossSize_w = opt.lossSize_w

        # init
        self.teacher = ResNet18_MS3(pretrained=True).to(self.device)
        self.student = ResNet18_MS3(pretrained=False).to(self.device)
        self.start_epoch = opt.start_epoch
        self.finish_epoch = opt.finish_epoch
        self.optimizer = torch.optim.SGD(self.student.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

        self.cur_losses = {
            'Epoch_loss': float('inf'),
            'Val_loss': float('inf')
        }

        try:
            self.load_checkpoint()
            print(self.cur_losses)
            print("Checpoint have been loaded.")
        except:
            print("Checkpoint load fail.")

        try:
            self.load_weight()
            print("Student'weight have been loaded.")
        except:
            print("Student'weight load fail.")

    def save_checkpoint(self, is_best=False):
        if is_best:
            path = os.path.join(self.checkpoint_save_path, f'checkpoint_best.pt')
        else:
            path = os.path.join(self.checkpoint_save_path, f'checkpoint_{self.cur_epoch}.pt')

        torch.save({
                'epoch': self.cur_epoch,
                'teacher_state_dict': self.teacher.state_dict(),
                'student_state_dict': self.student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'dictionary_losses': self.cur_losses
            }, path)
        print(f"Checkpoint saved successfully at epoch {self.cur_epoch}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_load_path, map_location=self.device)
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_losses = checkpoint['dictionary_losses']

    def load_weight(self):
        self.student.load_state_dict(torch.load(self.weight_path)['state_dict'])

    def train(self, train_loader, val_loader, test_neg_loader, test_pos_loader):
        min_err = float('inf')
        best_auc = 0
        self.teacher.eval()
        self.student.train()
        with open(self.log_path, "a") as log_file:
            log_file.write("Training started\n")

        with tqdm(range(self.start_epoch, self.finish_epoch + 1)) as t:
            for self.cur_epoch in t:
                t.set_description(f"Epoch {self.cur_epoch} /{self.finish_epoch}")
                self.student.train()
                epoch_loss = 0

                for imgs in train_loader:
                    batch_size = imgs.size(0)
                    imgs = imgs.to(self.device)

                    with torch.no_grad():
                        teacher_feat = self.teacher(imgs)
                    student_feat = self.student(imgs)

                    loss = 0
                    for i in range(len(teacher_feat)):
                        teacher_feat[i] = F.normalize(teacher_feat[i], dim=1)
                        student_feat[i] = F.normalize(student_feat[i], dim=1)
                        loss += torch.sum((teacher_feat[i] - student_feat[i]) ** 2, 1).mean()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.detach().item() * batch_size

                epoch_loss /= len(train_loader.dataset)
                print('[%d/%d] loss: %f' % (self.cur_epoch, self.finish_epoch, epoch_loss))
                val_loss = self.test(val_loader).mean()
                print('Valid Loss: {:.7f}'.format(val_loss.item()))
                self.cur_losses = {
                    'Epoch_loss': epoch_loss,
                    'Val_loss': val_loss.item()
                }
                auc = self.cal_auc(test_neg_loader, test_pos_loader)
                print('AUC = ', auc)
                
                text = f'{self.cur_epoch}:   {auc}'
                with open(self.log_path, "a") as log_file:
                    log_file.write('%s\n' % text)
                
                t.set_postfix(epoch_loss=epoch_loss, val_loss=val_loss, auc=auc)
                
                if auc > best_auc:
                    best_auc = auc
                    self.save_checkpoint(is_best=True)
                    with open(self.log_path, "a") as log_file:
                        log_file.write('-\n')

                if self.cur_epoch % 50 == 0:
                    self.save_checkpoint()

    def test(self, test_loader):
        self.teacher.eval()
        self.student.eval()
        loss_map = np.zeros((len(test_loader.dataset), self.lossSize_h, self.lossSize_w))
        i = 0
        for imgs in test_loader:
            imgs = imgs.to(self.device)
            with torch.no_grad():
                teacher_feat = self.teacher(imgs)
                student_feat = self.student(imgs)
            score_map = 1.

            for j in range(len(teacher_feat)):
                teacher_feat[j] = F.normalize(teacher_feat[j], dim=1)
                student_feat[j] = F.normalize(student_feat[j], dim=1)
                sm = torch.sum((teacher_feat[j] - student_feat[j]) ** 2, 1, keepdim=True)
                sm = F.interpolate(sm, size=(self.lossSize_h, self.lossSize_w), mode='bilinear', align_corners=False)

                score_map = score_map * sm
            loss_map[i: i + imgs.size(0)] = score_map.squeeze().cpu().data.numpy()
            i += imgs.size(0)
        return loss_map
    

    def cal_auc(self, test_neg_loader, test_pos_loader):
        
        neg = self.test(test_neg_loader)
        pos = self.test(test_pos_loader)

        scores = []

        for i in range(len(neg)):
            temp = cv2.resize(neg[i], (128, 192))
            scores.append(temp)
        for i in range(len(pos)):
            temp = cv2.resize(pos[i], (128, 192))
            scores.append(temp)
        
        scores = np.stack(scores)
        gt_image = np.concatenate((np.zeros(neg.shape[0], dtype=np.bool_), np.ones(pos.shape[0], dtype=np.bool_)), 0)        

        # auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        scores.max(-1).max(-1)
        return roc_auc_score(gt_image, scores.max(-1).max(-1))
    
    
    def get_scores(self, test_neg_loader, test_pos_loader):
        
        neg = self.test(test_neg_loader)
        pos = self.test(test_pos_loader)

        scores_neg = []
        scores_pos = []

        for i in range(len(neg)):
            temp = cv2.resize(neg[i], (128, 192))
            scores_neg.append(temp)
        for i in range(len(pos)):
            temp = cv2.resize(pos[i], (128, 192))
            scores_pos.append(temp)
        
        scores_neg = np.stack(scores_neg)
        scores_pos = np.stack(scores_pos)

        return scores_neg.max(-1).max(-1), scores_pos.max(-1).max(-1)
