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
import os, time, torch
from tempfile import TemporaryDirectory
import seaborn as sn

cudnn.benchmark = True

BATCH_SIZE = 32
NUM_CLASSES = 3
NUM_EPOCHS = 3
CRITERION = nn.CrossEntropyLoss()

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

    datasets = {
        'train': train_dataset,
        'val': test_dataset
    }

    dataloaders = {
        'train': train_dataloader,
        'val': test_dataloader
    }

    return datasets, dataloaders


def create_model(num_classes=3):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

def train_model(model, dataloaders, optimizer, scheduler, criterion, num_epochs=25, save_model=False):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.to(device).train()  # Set model to training mode
                else:
                    model.to(device).eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, sample in enumerate(dataloaders[phase]):
                    inputs = sample['image'].to(device).float()
                    labels = sample['labels'].type(torch.LongTensor).squeeze(-1).to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects.double() / len(dataloaders[phase])

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path)) #, weights_only=True))
    if save_model:
        torch.save(model.state_dict(), '/best_model')
    return model


def main():
    train_data = pd.read_csv('/train_data.csv')
    test_data = pd.read_csv('/test_data.csv')
    datasets, dataloaders = create_dataloaders(batch_size=BATCH_SIZE)
    model = create_model(NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, dataloaders, optimizer, scheduler, criterion=CRITERION, num_epochs=NUM_EPOCHS, save_model=True)

    return model


if __name__ == "__main__":
    main()
