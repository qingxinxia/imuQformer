import torch.cuda

from utils.prepare_imu import load_data
import pickle
from models.imu_qformer import IMUQformer
from torch.utils.data import DataLoader
from vet2text_utils.model import get_embeddings_openai
from models.contrastive import SimCLRLoss
from utils.augmentations import *
import torch.nn as nn

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load IMU and text data
    imu_dataset = load_data()

    # %% Create Train Dataloaders
    train_dataloader = DataLoader(imu_dataset, batch_size=20,
                                   shuffle=True, num_workers=0)

    # %% initialize model
    model = IMUQformer()
    model = model.to(device)
    model.train()
    mse_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    contrastive_model = SimCLRLoss()
    contrastive_model = contrastive_model.to(device)
    contrastive_model.train()
    contrast_optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=0.0001)

    # %% training
    for epoch in range(1):
        losses = 0
        for idx, sample in enumerate(train_dataloader):
            data = sample['data'].float()

            # %% step 1: train imu encoder and save checkpoint (contrastive loss)
            view1 = gen_aug(data, 'perm_jit').float()
            view2 = gen_aug(data, 'negate').float()
            view1, view2 = view1.to(device), view2.to(device)
            loss = contrastive_model(view1, view2)
            losses += loss.item()
            loss.backward()
            contrast_optimizer.step()
        print(losses)

    # save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': contrastive_model.state_dict(),
        'optimizer_state_dict': contrast_optimizer.state_dict()
    }
    torch.save(checkpoint, 'contrastive_model.pth')

    # %% step 2: train qformer (mse loss)

    # load checkpoint
    checkpoint = torch.load('contrastive_model.pth')

    # Load the model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Load any other information, like the epoch
    start_epoch = checkpoint['epoch']

    for epoch in range(1):
        # for i in range(len(imu_dataset)):
        for idx, sample in enumerate(train_dataloader):
            data = sample['data'].float()
            data = data.to(device)
            label = sample['label']
            embeddings = get_embeddings_openai(label)
            # print(embeddings.shape)
            loss = model(data, embeddings)
            loss.backward()
            mse_optimizer.step()

    return


if __name__ == '__main__':
    main()
