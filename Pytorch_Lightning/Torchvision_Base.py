import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import random
import torchvision
from pytorch_lightning.trainer.supporters import CombinedLoader
import itertools
import torchvision.models as models

learning_state = 1
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

class Test_Load_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.img_list = glob(data_path)
        self.transform = transform
        self.lr_state = learning_state

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])

        spl = self.img_list[idx].split('/')
        spl2 = spl[-1].split('_')
        labels = np.long(spl2[1]) - 1
        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'labels': labels}
        return sample


class Cls_Load_Dataset2(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.img_list = glob(data_path)
        self.transform = transform
        self.lr_state = learning_state

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])

        spl = self.img_list[idx].split('/')
        spl2 = spl[-1].split('_')
        labels = np.long(spl2[-1][:-4]) - 1

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'labels': labels}
        return sample


class Cls_Load_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.img_list = glob(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])

        spl = self.img_list[idx].split('/')
        spl2 = spl[-1].split('_')
        labels = np.long(spl2[-1][:-4]) - 1

        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'labels': labels}
        return sample


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        transforms_Cls = transforms.Compose(
            [
                transforms.Grayscale(3),# If Value is 3, Input is RGB Image. If Value is 1, Input is Gray Image
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        self.Cls_Dataset = Cls_Load_Dataset2(data_path='C:/Users/user/Desktop/SSS/JH2/train/*', transform=transforms_Cls)
        self.test_Dataset = Cls_Load_Dataset2(data_path='C:/Users/user/Desktop/SSS/JH2/val/*', transform=transforms_Cls)
        self.test_Dataset2 = Test_Load_Dataset(data_path='C:/Users/user/Desktop/SSS/ClaDataset/*', transform=transforms_Cls)

        #If you want to sum dataset
        ######################################################################################################
        # self.one = Test_Load_Dataset(data_path='/JSS/Dataset/ClaDataset/1/*.png', transform=transforms_test)
        # self.two = Test_Load_Dataset(data_path='/JSS/Dataset/ClaDataset/2/*.png',transform=transforms_test)
        # self.three = Test_Load_Dataset(data_path='/JSS/Dataset/ClaDataset/3/*.png',transform=transforms_test)
        # self.four = Test_Load_Dataset(data_path='/JSS/Dataset/ClaDataset/4/*.png',transform=transforms_test)
        # self.five = Test_Load_Dataset(data_path='/JSS/Dataset/ClaDataset/5/*.png',transform=transforms_test)
        # self.test_dataset = ConcatDataset([self.one, self.two, self.three, self.four, self.five])
        ######################################################################################################

    def train_dataloader(self):
        return DataLoader(dataset=self.Cls_Dataset, batch_size=30, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_Dataset, batch_size=30, shuffle=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_Dataset2, batch_size=30, shuffle=True)


class Main_Network(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.Classifier = Classifier()
        self.Cls_criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['labels']
        out = self.Classifier(image)
        Loss = self.Cls_criterion(out, label)
        predictions = out.argmax(-1)
        correct = (predictions == label).sum().item()
        Acc = correct / image.shape[0]
        self.log_dict({"Loss": Loss, "Acc": Acc}, prog_bar=True)
        return Loss

    def validation_step(self, batch, batch_idx):
        img = batch['image']
        label = batch['labels']
        out = self.Classifier(img)
        Clsloss = self.Cls_criterion(out, label)
        predictions = out.argmax(-1)
        correct = (predictions == label).sum().item()
        acc = correct / img.shape[0]
        self.log_dict({"Val_Loss": Clsloss, "Val_Acc": acc}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        img = batch['image']
        label = batch['labels']
        out = self.Classifier(img)
        Clsloss = self.Cls_criterion(out, label)
        predictions = out.argmax(-1)
        correct = (predictions == label).sum().item()
        acc = correct / img.shape[0]
        self.log_dict({"Loss": Clsloss, "Acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        # Scheduler
        #############################################################################################################
        # optimizer = torch.optim.AdamW(self.parameters(), lr=0.1)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer ,lr_lambda=lambda epoch: 0.95 ** epoch)
        # return [optimizer], [scheduler]
        #############################################################################################################
        return torch.optim.AdamW(self.parameters(), lr=5e-5)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.Encoder = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',pretrained=False)
        self.Classifier = nn.Sequential(
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        x = self.Encoder(x)
        out = self.Classifier(x)
        return out


Callback = EarlyStopping(
    monitor='Val_Total_Loss',
    patience=5,
    strict=False,
    verbose=True,
    mode='min'
)


def Trainfunc(model, Dataset):
    set_seed(0)
    trainer.fit(model, Dataset)


def Testfunc(trainer, model, Dataset):
    set_seed(0)

    checkpoint = torch.load('Classification/version_0/checkpoints/epoch=147-step=12728.ckpt', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    # TEST after train [best weights]
    print("Train##########################################################")
    trainer.test(model=model, dataloaders=Dataset.train_dataloader())
    print("Validation##########################################################")
    trainer.test(model=model, dataloaders=Dataset.val_dataloader())
    print("Test##########################################################")
    trainer.test(model=model, dataloaders=Dataset.test_dataloader())
    print("##########################################################")


if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="Classification")
    Dataset = DataModule()
    model = Main_Network()

    set_seed(0)
    trainer = Trainer(gpus=1, max_epochs=200, callbacks=[Callback, ModelCheckpoint(monitor='Val_Acc', mode='max')], logger=logger, reload_dataloaders_every_n_epochs=1)
    Trainfunc(model, Dataset)
    #If you want to testing
    #Testfunc(trainer, model, Dataset)


