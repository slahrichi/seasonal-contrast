from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser

import torch
from torch import nn, optim
from torchvision.models import resnet
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.eurosat_datamodule import EurosatDataModule
from models.moco2_module import MocoV2

import re
from collections import OrderedDict
from models import resnet

class Classifier(LightningModule):

    def __init__(self, backbone, in_features, num_classes):
        super().__init__()
        self.encoder = backbone
        self.classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(torch.argmax(logits, dim=1), y)
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.classifier.parameters())
        max_epochs = self.trainer.max_epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*max_epochs), int(0.8*max_epochs)])
        return [optimizer], [scheduler]

def _load_seco(model_path):
    checkpoint = torch.load(model_path)
    checkpoint_dict =  checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for key in list(checkpoint_dict.keys()):
        if key.startswith("encoder_q"):
            new_key = re.sub("encoder_q.","",key)
            new_key = re.sub("^0(?=\.weight)","conv1",new_key)
            new_key = re.sub("^1(?=\.[a-z_]*)","bn1",new_key)
            new_key  = re.sub("^4(?=\.\d\.(conv|bn|downsample))","layer1",new_key )
            new_key  = re.sub("^5(?=\.\d\.(conv|bn|downsample))","layer2",new_key )
            new_key  = re.sub("^6(?=\.\d\.(conv|bn|downsample))","layer3",new_key )
            new_key  = re.sub("^7(?=\.\d\.(conv|bn|downsample))","layer4",new_key )
            new_state_dict[new_key] = checkpoint_dict[key] 

    model = resnet.resnet50(inter_features=False)
    model.load_state_dict(new_state_dict)
    
    return model

if __name__ == '__main__':
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--backbone_type', type=str, default='imagenet')
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    datamodule = EurosatDataModule(args.data_dir)

    if args.backbone_type == 'random':
        backbone = resnet.resnet18(pretrained=False)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.backbone_type == 'imagenet':
        backbone = resnet.resnet18(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    elif args.backbone_type == 'pretrain':
        model = MocoV2.load_from_checkpoint(args.ckpt_path)
        backbone = deepcopy(model.encoder_q)
    elif args.backbone_type == "checkpoint":
        backbone = _load_seco(args.ckpt_path)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())

    else:
        raise ValueError()

    model = Classifier(backbone, in_features=2048, num_classes=datamodule.num_classes)
    model.example_input_array = torch.zeros((1, 3, 64, 64))

    experiment_name = args.backbone_type
    logger = TensorBoardLogger(save_dir=str(Path.cwd() / 'logs' / 'eurosat'), name=experiment_name)
    trainer = Trainer(gpus=args.gpus, logger=logger, checkpoint_callback=False, max_epochs=100, weights_summary='full')
    trainer.fit(model, datamodule=datamodule)
