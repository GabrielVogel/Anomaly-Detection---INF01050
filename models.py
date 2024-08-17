import os
from torch import optim, nn, utils, Tensor
import lightning as L
import torch
import torch.nn.functional as F
from scipy.stats import beta


class StandardModel(L.LightningModule):
    def __init__(self, alpha,backbone,n_classes):
        super().__init__()
        self.nclasses = n_classes
        self.backbone = backbone
        self.l1 = nn.Linear(512,256)
        self.l2 = nn.Linear(256,self.nclasses)

    def accuracy(self,logits,labels):
        preds = torch.argmax(logits,dim=1)
        return (preds == labels).float().mean()

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    def partial_forward(self,x):
        x = self.backbone(x).view(x.size(0), -1)
        x = F.relu(self.l1(x))
        return x

    def training_step(self,batch,batch_idx):
        x,y = batch
        logits_nomix = self.forward(x)
        loss = F.cross_entropy(logits_nomix,y)
        self.log('train_loss',loss,on_epoch=True,prog_bar=True)
        self.log('train_acc',self.accuracy(logits_nomix,y),on_epoch=True,prog_bar=True)
        return {"loss":loss}


    def validation_step(self,batch,batch_idx):
        x,y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits,y)
        self.log('val_loss',loss,on_epoch=True,prog_bar=True)
        self.log('val_acc',self.accuracy(logits,y),on_epoch=True,prog_bar=True)
        return loss

    def predict_step(self,batch,batch_idx):
        x,y = batch
        logits = self.forward(x)
        return logits

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=3e-4)
        return optimizer

class ModelWithDropout(L.LightningModule):
    def __init__(self, alpha,backbone,n_classes):
        super().__init__()
        self.alpha = alpha
        self.nclasses = n_classes
        self.backbone = backbone
        self.l1 = nn.Linear(512,256)
        self.drop = nn.Dropout(0.5)
        self.l2 = nn.Linear(256,self.nclasses)
        self.mc_iterations = 50

    def accuracy(self,logits,labels):
        preds = torch.argmax(logits,dim=1)
        return (preds == labels).float().mean()

    def mixup_data(self, x, y,alpha):
        lam = beta.rvs(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        y_ = torch.zeros((batch_size, self.nclasses))
        y_[torch.arange(batch_size),y] = 1.0
        y_ = y_.to(self.device)
        mixup_x = lam * x + (1 - lam) * x[index, :]
        mixup_y = lam * y_ + (1 - lam) * y_[index]

        return mixup_x, mixup_y

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        x = self.drop(F.relu(self.l1(x)))
        x = self.l2(x)
        return x

    def partial_forward(self,x):
        x = self.backbone(x).view(x.size(0), -1)
        x = self.drop(F.relu(self.l1(x)))
        return x

    def training_step(self,batch,batch_idx):
        x,y = batch
        latent_rep = self.partial_forward(x)
        mix_x,mix_y = self.mixup_data(latent_rep,y,self.alpha)
        logits_nomix = self.l2(latent_rep)
        logits_mix = self.l2(mix_x)
        loss = F.cross_entropy(logits_nomix,y)
        self.log('train_loss',loss,on_epoch=True,prog_bar=True)
        self.log('train_acc',self.accuracy(logits_nomix,y),on_epoch=True,prog_bar=True)
        return {"loss":loss}


    def validation_step(self,batch,batch_idx):
        x,y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits,y)
        self.log('val_loss',loss,on_epoch=True,prog_bar=True)
        self.log('val_acc',self.accuracy(logits,y),on_epoch=True,prog_bar=True)
        return loss

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


    def predict_step(self,batch,batch_idx):
        x,y = batch
        #self.enable_dropout()
        self.drop.train()
        logits = torch.stack([torch.softmax(self.forward(x),dim=1) for _ in range(self.mc_iterations)])
        r = torch.mean(logits,dim=0)
        return r

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=3e-4)
        return optimizer


class ModelWithRegMixUp(L.LightningModule):
    def __init__(self, alpha,backbone,n_classes):
        super().__init__()
        self.alpha = alpha
        self.nclasses = n_classes
        self.backbone = backbone
        self.l1 = nn.Linear(512,256)
        self.drop = nn.Dropout(0.3)
        self.l2 = nn.Linear(256,self.nclasses)

    def accuracy(self,logits,labels):
        preds = torch.argmax(logits,dim=1)
        return (preds == labels).float().mean()

    def mixup_data(self, x, y,alpha):
        lam = beta.rvs(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        y_ = torch.zeros((batch_size, self.nclasses))
        y_[torch.arange(batch_size),y] = 1.0
        y_ = y_.to(self.device)
        mixup_x = lam * x + (1 - lam) * x[index, :]
        mixup_y = lam * y_ + (1 - lam) * y_[index]

        return mixup_x, mixup_y


    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        x = self.drop(F.relu(self.l1(x)))
        x = self.l2(x)
        return x

    def partial_forward(self,x):
        x = self.backbone(x).view(x.size(0), -1)
        x = self.drop(F.relu(self.l1(x)))
        return x

    def training_step(self,batch,batch_idx):
        x,y = batch
        latent_rep = self.partial_forward(x)
        mix_x,mix_y = self.mixup_data(latent_rep,y,self.alpha)
        logits_nomix = self.l2(latent_rep)
        logits_mix = self.l2(mix_x)
        loss = F.cross_entropy(logits_mix,mix_y) + F.cross_entropy(logits_nomix,y)
        self.log('train_loss',loss,on_epoch=True,prog_bar=True)
        self.log('train_acc',self.accuracy(logits_nomix,y),on_epoch=True,prog_bar=True)
        return {"loss":loss}


    def validation_step(self,batch,batch_idx):
        x,y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits,y)
        self.log('val_loss',loss,on_epoch=True,prog_bar=True)
        self.log('val_acc',self.accuracy(logits,y),on_epoch=True,prog_bar=True)
        return loss

    def predict_step(self,batch,batch_idx):
        x,y = batch
        logits = self.forward(x)
        return logits

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=3e-4)
        return optimizer
