# Let's load dependencies

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch.nn as nn
import torch, torchvision
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm, trange
from torchvision import models

from sklearn.metrics import accuracy_score
from utils import loadDataset, EarlyStopping, accuracyMetrics
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from customCNN import CustomCNN
import argparse

DEVICE = None
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for training or testing a model")
    
    parser.add_argument('--task', choices=['train', 'test'], required=True, help="Specify whether to train or test the model")
    parser.add_argument('--train_path', help="Path to the training data")
    parser.add_argument('--test_path', help="Path to the test data")
    parser.add_argument('--model_name', choices=['custom', 'mobilenetv3'], help="Name of the model")
    parser.add_argument('--model_weights', help="Path to the model weights")
    parser.add_argument('--epochs', default=20, help="Number of epochs to train")
    
    args = parser.parse_args()
    
    if args.task == 'train':
        if not all([args.trainpath, args.testpath, args.modelname]):
            parser.error("--trainpath, --testpath, and --modelname are required for training")
    elif args.task == 'test':
        if not all([args.testpath, args.modelweights]):
            parser.error("--testpath and --modelweights are required for testing")
            
    return args


def train(train_loader, val_loader, model, num_epochs, optimizer, criterion, weights_dir, scheduler=None):
    writer = SummaryWriter('logs')
    early_stopping = EarlyStopping(patience=11, verbose=True)
    best_model_wts = model.state_dict()
    best_epoch = 0
    best_score = 0

    losses = {'train': [], "val": []}

    pbar = trange(num_epochs, desc="Epoch:")

    loaders = {"train": train_loader, "val": val_loader}

    for epoch in pbar:

        for k, dataloader in loaders.items():

            running_loss = 0.0
            epoch_preds, epoch_batches = [], []

            for batch in tqdm(dataloader, leave=False, desc=f"{k} iter:"):
                x_batch, y_batch = batch
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                if k == "train":
                    model.train()
                    optimizer.zero_grad()
                    outp = model(x_batch)
                    loss = criterion(outp, y_batch)
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch)
                        loss = criterion(outp, y_batch)

                preds = outp.argmax(-1)

                epoch_preds += preds.cpu()
                epoch_batches += y_batch.cpu()
                running_loss += loss.item() * x_batch.size(0)

            epoch_score = accuracy_score(epoch_preds, epoch_batches)
            epoch_loss = running_loss / len(dataloader.dataset)
            losses[k].append(epoch_loss)

            if k == "train":
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_score, epoch)
            if k == "val":
                writer.add_scalar('Loss/validation', epoch_loss, epoch)
                writer.add_scalar('Accuracy/validation', epoch_score, epoch)
                early_stopping(epoch_score)
                scheduler.step(epoch_loss)



            if k == 'val':
              print(f'Epoch: {epoch + 1} of {num_epochs}  Score: {epoch_score}')
#               if scheduler is not None:
#                 scheduler.step(epoch_score)

            pbar.set_description('{} Loss: {:.4f} Score: {:.4f}'.format(k, epoch_loss, epoch_score))

            if k == 'val' and epoch_score > best_score:
               best_score = epoch_score
               best_epoch = epoch + 1
               best_model_wts = model.state_dict()
               if not os.path.exists(weights_dir):
                   os.makedirs(weights_dir)
               torch.save(model.state_dict(), os.path.join(weights_dir, "best_model.pth"))

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f'Best score: {best_score}\nEpoch {best_epoch} of {num_epochs}')
    model.load_state_dict(best_model_wts)
    return model, losses, best_score


def generateOnnxModel(weights_dir):
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"]
    output_names = ["output"]

    # Export the model to ONNX format
    onnx_filename = os.path.join(weights_dir, "model_best.onnx")
    torch.onnx.export(model, dummy_input, onnx_filename, input_names=input_names, output_names=output_names)

if __name__ == "__main__":

    args = parse_arguments()

    model = None
    if args.model_name == "mobilenetv3":
        print("Loading with pretrained weights for better performance")
        model = models.mobilenet_v3_small(pretrained=True, quantize=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 7)
        model = model.to(DEVICE)
    elif args.model_name == "custom":
        if args.epochs <=30:
            print("Note: kindly set it above 30 as its training from scratch")
        model = CustomCNN(num_classes=7)
        model.to(DEVICE)

    trainloader, testloader = loadDataset(args.train_path, args.test_path)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    model, losses_mob_netv3, accuracy_mob_netv3 = train(trainloader,testloader, model=model, num_epochs=args.epochs, optimizer=optimizer, criterion=criterion, weights_dir="weights", scheduler=scheduler) # Training loop with 10 epochs

    # train and loss curve tensorboard logs
    print("Note: You can view the tensorboard logs by : tensorboard --logdir logs/")

    # prints accuracy metrics
    accuracyMetrics(model, testloader, args.test_path)

    

