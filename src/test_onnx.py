import onnxruntime
import torch
from tqdm import tqdm
import os
import time
import argparse
from utils import load_test_dataset
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to use an ONNX model for testing")
    
    parser.add_argument('--onnx_model_path', required=True, help="Path to the ONNX model file")
    parser.add_argument('--test_dir', required=True, help="Directory containing test data")
    
    args = parser.parse_args()
    
    return args

def check_file_exists(file_path):
    if os.path.exists(file_path):
        print(f"The file '{file_path}' exists.")
    else:
        print(f"The file '{file_path}' does not exist.")
        return False
    return True

def test_onnx_accuracy(onnx_model_path, test_dir, RESCALE_SIZE):
    testloader = load_test_dataset(test_dir, RESCALE_SIZE, batch_size=1)

    session = onnxruntime.InferenceSession(onnx_model_path)
    total_samples = 0
    correct_predictions = 0
    loaders = {"val": testloader}

    # Iterate over the test dataset
    for k, dataloader in loaders.items():
        for batch in tqdm(dataloader, leave=False, desc=f"{k} iter:"):
            inputs, labels = batch
            inputs = inputs.to("cpu")
            labels = labels.to("cpu")
            total_samples += labels.size(0)

            # Run inference on the ONNX model
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            outputs = session.run([output_name], {input_name: inputs.numpy().astype(np.float32)})
            outputs = torch.tensor(outputs[0])
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    return accuracy

def check_inference_time(onnx_model_path, RESCALE_SIZE):

    # Generate a random tensor
    input_tensor = np.random.randn(1, 3, RESCALE_SIZE, RESCALE_SIZE).astype(np.float32)

    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    start_time = time.time()
    output = session.run([], {input_name: input_tensor})
    end_time = time.time()
    inference_time = end_time - start_time

    return inference_time

if __name__ == "__main__":
    args = parse_arguments()
    RESCALE_SIZE = 224

    if check_file_exists(args.onnx_model_path):
        inference_time = check_inference_time(args.onnx_model_path, RESCALE_SIZE)
        print(f"Inference time: {inference_time} seconds")

    if check_file_exists(args.onnx_model_path) and check_file_exists(args.test_dir):
        accuracy = test_onnx_accuracy(args.onnx_model_path, args.test_dir, RESCALE_SIZE)
        print(f"Accuracy of {args.onnx_model_path} is {accuracy}%")
        