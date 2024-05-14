# Accuracy test for mobilenetv3_onnx_prequant_float16.tflite model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import numpy as np
import argparse
import time
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to use a tflite model for testing")
    
    parser.add_argument('--tflite_model_path', required=True, help="Path to the tflite model file")
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

def test_tflite_accuracy(tflite_model_path, test_dir, RESCALE_SIZE)
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Data preprocessing
    img_size = (RESCALE_SIZE, RESCALE_SIZE)
    batch_size = 1
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Perform inference and calculate accuracy
    correct = 0
    total = 0
    print("total nums images - ", test_generator.n)

    for _ in range(test_generator.n):
        # Perform inference
        images, labels = next(test_generator)
        images = (images - MEAN) / STD
        images= np.array(images, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], images)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_labels = np.argmax(output_data, axis=1)

        # Compare with ground truth labels
        ground_truth_labels = np.argmax(labels, axis=1)
        correct += np.sum(predicted_labels == ground_truth_labels)
        total += len(images)
        count += 1

    # Calculate accuracy
    accuracy = correct / total
    return accuracy

def check_inference_time(tflite_model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Generate a random input tensor (assuming input tensor shape [batch_size, height, width, channels])
    input_shape = input_details[0]['shape']
    input_tensor = np.random.randn(*input_shape).astype(np.float32)

    # Set input tensor to the model
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference and measure time
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    output_tensor = interpreter.get_tensor(output_details[0]['index'])

    # Calculate inference time
    inference_time = end_time - start_time

    return inference_time

if __name__ == "__main__":

    args = parse_arguments()
    RESCALE_SIZE = 224

    if check_file_exists(args.tflite_model_path):
        inference_time = check_inference_time(args.tflite_model_path, RESCALE_SIZE)
        print(f"Inference time: {inference_time} seconds")

    if check_file_exists(args.tflite_model_path) and check_file_exists(args.test_dir):
        accuracy = test_tflite_accuracy(args.tflite_model_path, args.test_dir, RESCALE_SIZE)
        print(f"Accuracy of {args.tflite_model_path} is {accuracy}%")