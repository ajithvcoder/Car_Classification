# Accuracy test for mobilenetv3_onnx_prequant_float16.tflite model

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="saved_model/best_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Image size and batch size
img_size = (224, 224)  # Adjust this according to your model input size
batch_size = 1
test_dir = "./Cars Dataset/test"
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
# Prepare test generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Set shuffle to False to ensure correct label matching
)

# Perform inference and calculate accuracy
correct = 0
total = 0
count = 0
print("total nums images - ", test_generator.n)

for _ in range(test_generator.n):
    # Perform inference
    images, labels = next(test_generator)
    images = (images - MEAN) / STD
    images= np.array(images, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], images)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted labels
    predicted_labels = np.argmax(output_data, axis=1)

    # Compare with ground truth labels
    ground_truth_labels = np.argmax(labels, axis=1)
    correct += np.sum(predicted_labels == ground_truth_labels)
    total += len(images)
    count += 1
    # print(count)

# Calculate accuracy
accuracy = correct / total
print("Accuracy:", accuracy)
