import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
from tempfile import TemporaryDirectory
import seaborn as sn

cudnn.benchmark = True

BATCH_SIZE = 32
NUM_CLASSES = 3
k = 0
REPO_PATH = '.../.../'
MODEL_WEIGHTS_PATH = REPO_PATH + '/weights/drawing-type classifier/ResNet-18_k'+str(k)

class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        Res_transform = transforms.Resize([224,224])
        image = Res_transform(image)
        
        return {'image': image,
                'labels': labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.frame.iloc[idx, 0]
        image = io.imread(img_name)
        labels = self.frame.iloc[idx, 1:]
        labels = np.array(labels, dtype=int)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

def create_dataloaders(batch_size=32):
    train_dataset = ImageDataset(csv_file='/train_data.csv',
                                            transform=transforms.Compose([ToTensor(), Resize()]))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    test_dataset = ImageDataset(csv_file='/test_data.csv',
                                            transform=transforms.Compose([ToTensor(), Resize()]))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    dataloaders = {
        'train': train_dataloader,
        'val': test_dataloader
    }

    return dataloaders

def load_model(model_weights_path, num_classes=3):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_test.load_state_dict(torch.load(model_weights_path+str(k)))
    return model_ft

def test_model(model_test, dataloaders):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions, labs = list(), list()
    for phase in ['val']:
        if phase == 'train':
            model_test.to(device).train()  # Set model to training mode
        else:
            model_test.to(device).eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, sample in enumerate(dataloaders[phase]):
            inputs = sample['image'].to(device).float()
            labels = sample['labels'].type(torch.LongTensor).squeeze(-1).to(device)

            # zero the parameter gradients
    #         optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model_test(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            predictions.extend(list(preds.detach().cpu().numpy()))
            labs.extend(list(labels.detach().cpu().numpy()))
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / len(dataloaders[phase])
        epoch_acc = running_corrects.double() / len(dataloaders[phase])

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return np.array(predictions), np.array(labs)

def main():
    train_data = pd.read_csv('/train_data.csv')
    test_data = pd.read_csv('/test_data.csv')
    dataloaders = create_dataloaders(batch_size=BATCH_SIZE)
    model = load_model(MODEL_WEIGHTS_PATH, NUM_CLASSES)
    model = test_model(model, dataloaders)

    return model


if __name__ == "__main__":
    main()


