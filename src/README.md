### Car Classification Model

Usage:

- TRAIN 

        python main.py --task "train" --train_path "./Cars Dataset/train" --test_path "./Cars Dataset/test" --model_name "mobilenetv3"

        You can also train with "custom" method

- TEST

        python main.py --task "test" --test_path "./Cars Dataset/test" --model_weights "weights/best_model.pth"


TFLITE Conversion:

    onnx2tf -i "weights/model_best.onnx"

- Note:

    You can use test_tflite.py to test your tflite model.

    You can use test_onnx.py to test your onnx model.


