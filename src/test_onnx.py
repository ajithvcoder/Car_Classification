import onnxruntime
import torch
import tqdm
from utils import loadDataset
import numpy as np

trainloader, testloader = loadDataset("Cars Dataset/train", "Cars Dataset/test")

# Load your ONNX model
onnx_model_path = 'weights/best_model.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)

# Initialize variables for accuracy calculation
total_samples = 0
correct_predictions = 0
loaders = {"val": testloader}

# Iterate over the test dataset
for k, dataloader in loaders.items():
    for batch in tqdm(dataloader, leave=False, desc=f"{k} iter:"):
        # inputs, labels
        inputs, labels = batch
        inputs = inputs.to("cpu")
        labels = labels.to("cpu")
        total_samples += labels.size(0)

        # Run inference on the ONNX model
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: inputs.numpy().astype(np.float32)})

        # Convert the output to torch tensor
        outputs = torch.tensor(outputs[0])

        # Get predicted labels
        _, predicted = torch.max(outputs, 1)

        # Count correct predictions
        correct_predictions += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = correct_predictions / total_samples
print("Accuracy:", accuracy)
