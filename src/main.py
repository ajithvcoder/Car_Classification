import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm, trange
from torchvision import models

from sklearn.metrics import accuracy_score
from utils import load_train_dataset, load_test_dataset, EarlyStopping, accuracy_metrics
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from custom_cnn import CustomCNN
import argparse
from logger import logger

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
    parser.add_argument('--model_name', default="mobilenetv3", choices=['custom', 'mobilenetv3'], help="Name of the model")
    parser.add_argument('--model_weights', help="Path to the model weights")
    parser.add_argument('--epochs', default=20, type=int, help="Number of epochs to train")
    parser.add_argument('--tflite_model', default=False, type=bool, choices=[True, False], help="Generate tflite model or not")
    parser.add_argument('--output', default="weights", help="Output directory")
    
    args = parser.parse_args()
    
    if args.task == 'train':
        if not all([args.train_path, args.test_path, args.model_name]):
            parser.error("--train_path, --test_path, and --model_name are required for training")
    elif args.task == 'test':
        if not all([args.test_path, args.model_weights]):
            parser.error("--test_path and --model_weights are required for testing")
            
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
    logger.info(f"### Training Started ###")
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
                writer.add_scalar('Loss/train', epoch_loss, epoch+1)
                writer.add_scalar('Accuracy/train', epoch_score, epoch+1)
                logger.info(f"Epoch {epoch+1}: Train Loss: {epoch_loss}, Train Accuracy: {epoch_score}")
            if k == "val":
                writer.add_scalar('Loss/validation', epoch_loss, epoch+1)
                writer.add_scalar('Accuracy/validation', epoch_score, epoch+1)
                logger.info(f"Epoch {epoch+1}: Val Loss: {epoch_loss}, Val Accuracy: {epoch_score}")
                early_stopping(epoch_score)
                scheduler.step(epoch_loss)

            pbar.set_description('{} Loss: {:.4f} Score: {:.4f}'.format(k, epoch_loss, epoch_score))

            if k == 'val' and epoch_score > best_score:
               best_score = epoch_score
               best_epoch = epoch + 1
               best_model_wts = model.state_dict()
               if not os.path.exists(weights_dir):
                   os.makedirs(weights_dir)
               torch.save(model.state_dict(), os.path.join(weights_dir, "best_model.pth"))

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    logger.info(f'Best score: {best_score}\nEpoch {best_epoch} of {num_epochs}')
    model.load_state_dict(best_model_wts)
    writer.close()
    logger.info(f"### Training Ended ###")
    return model, losses, best_score


def generate_onnx_model(model, weights_dir):
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"]
    output_names = ["output"]

    # Export the model to ONNX format
    onnx_filename = os.path.join(weights_dir, "model_best.onnx")
    torch.onnx.export(model.to("cpu"), dummy_input, onnx_filename, input_names=input_names, output_names=output_names)
    return onnx_filename

if __name__ == "__main__":

    args = parse_arguments()

    model = None
    if args.model_name == "mobilenetv3":
        model = models.mobilenet_v3_small(weights=True, quantize=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 7)
        model = model.to(DEVICE)
        logger.info(f"Initalized Model - {args.model_name}")
        logger.info("Loading with pretrained weights for better performance")
    elif args.model_name == "custom":
        if args.task=="train" and args.epochs <=30:
            logger.info("Note: kindly set it above 30 as its training from scratch")
        model = CustomCNN(num_classes=7)
        model.to(DEVICE)
        logger.info(f"Initalized Model - {args.model_name}")
    
    # Custom model is configured for this input size
    RESCALE_SIZE = 224
    if args.task == "train":
        trainloader, testloader = load_train_dataset(args.train_path, RESCALE_SIZE), load_test_dataset(args.test_path, RESCALE_SIZE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        model, losses_model, accuracy_model = train(trainloader,testloader, model=model, num_epochs=args.epochs, optimizer=optimizer, criterion=criterion, weights_dir=args.output, scheduler=scheduler) # Training loop with 10 epochs

        # train and loss curve tensorboard logs
        logger.info("Note: You can view the tensorboard logs by : tensorboard --logdir logs/")

        # prints accuracy metrics
        accuracy_metrics(model, testloader, args.test_path, DEVICE)

        # generates onnx model
        onnx_filename = generate_onnx_model(model, args.output)
        logger.info(f"Generated onnx model at {onnx_filename}")
    elif args.task == "test":
        logger.info(f"### Testing Started ###")
        testloader = load_test_dataset(args.test_path, RESCALE_SIZE)
        model.load_state_dict(torch.load(args.model_weights))
        accuracy_metrics(model, testloader, args.test_path, DEVICE)

    # generates tflite model
    if args.tflite_model:
        logger.info("Note: Check if you have installed tensorflow and onnx2tf library to generate and test tflite model")
        import subprocess

        command = ['onnx2tf', '-i', onnx_filename]
        subprocess.run(command)
        logger.info("TFLITE file generated in saved_models dir")
