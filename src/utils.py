from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
import random
import numpy as np
import torchvision
from tqdm.autonotebook import tqdm
import torch 
from logger import logger
import os

def seed_worker(worker_id):
    np.random.seed(42)
    random.seed(42)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        score = val_score
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def maintainClassBalance(data_dir, dataset):
    # Get the class distribution
    class_counts = defaultdict(int)
    for _, label in dataset:
        class_counts[label] += 1

    # Find the maximum number of images among classes
    max_images_per_class = max(class_counts.values())

    for class_index, count in class_counts.items():
        class_dir = os.path.join(data_dir, dataset.classes[class_index])
        files = os.listdir(class_dir)
        num_augmentations = max_images_per_class - count
        augmented_samples = random.choices(files, k=num_augmentations)
        for file in augmented_samples:
            image_path = os.path.join(class_dir, file)
            dataset.samples.append((image_path, class_index))
    return dataset

def load_train_dataset(trainPath, RESCALE_SIZE, batch_size=32):
    train_transforms = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(degrees=10),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                    transforms.RandomResizedCrop(size=(RESCALE_SIZE, RESCALE_SIZE), scale=(0.8, 1.0)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    train_dataset = torchvision.datasets.ImageFolder(root=trainPath, transform = train_transforms)
    
    class_counts = defaultdict(int)
    class_names=sorted(os.listdir(trainPath))
    for _, label in train_dataset:
        class_counts[class_names[label]] += 1
    logger.info("Default dataset Info")
    logger.info(f"{class_counts}")
    train_dataset = maintainClassBalance(trainPath, train_dataset)
    class_counts = defaultdict(int)
    for _, label in train_dataset:
        class_counts[class_names[label]] += 1
    logger.info("Maintaing class balance and loading the dataset")
    logger.info(f"{class_counts}")
    trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 2, worker_init_fn=seed_worker)
    return trainloader

def load_test_dataset(testPath, RESCALE_SIZE, batch_size=32):
    test_transforms = transforms.Compose([transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_dataset = torchvision.datasets.ImageFolder(root=testPath, transform = test_transforms)
    testloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers = 2, worker_init_fn=seed_worker)
    return testloader


def get_val_labels(model, valDataloader, DEVICE):
    # mob_netv3_test = models.mobilenet_v3_small(pretrained=True, quantize=True)
    # mob_netv3_test.classifier[-1] = torch.nn.Linear(mob_netv3.classifier[-1].in_features, 7)
    # mob_netv3_test = mob_netv3.to(DEVICE)
    # mob_netv3_test.load_state_dict(torch.load("weights/best_model.pth"))
    predicted_labels = []
    val_labels = []
    epoch_preds = []
    epoch_batches = []
    for batch in tqdm(valDataloader, leave=False):
        x_batch, y_batch = batch
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        model.eval()
        with torch.no_grad():
            outp = model(x_batch)
        preds = outp.argmax(-1)

        epoch_preds += preds.cpu()
        epoch_batches += y_batch.cpu()


        # Apply argmax to get predicted labels
        _, preds_1 = torch.max(outp, 1)

        # Append current batch of predicted labels to the list
        predicted_labels.append(preds_1.cpu().numpy())  # Convert tensor to numpy array and move it to CPU
        val_labels.append(y_batch.cpu().numpy())
    epoch_score = accuracy_score(epoch_preds, epoch_batches)
    logger.info(f"Evaluation score: {epoch_score}")
    # Concatenate all the batches of predicted labels into a single numpy array
    predicted_labels = np.concatenate(predicted_labels, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    return predicted_labels, val_labels

def accuracy_metrics(model, testLoader, test_data_path, DEVICE):
    class_names=sorted(os.listdir(test_data_path))
    test_predicted_labels, test_true_labels = get_val_labels(model, testLoader, DEVICE)
    report = classification_report(test_true_labels, test_predicted_labels, target_names=class_names, digits=4)
    logger.info(f"{report}")
