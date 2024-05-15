### Car Classification Model

Kaggle Notebook : Available [here](./20240515-assignment-carclassification.ipynb)

Pretrained weights : Available [here](https://drive.google.com/file/d/11bGMM4WQkxoR30tCGGqpd2rvmU2xZdqh/view?usp=sharing)

Installation:

- GENERAL ONNX MODEL

        pip install -r requirements.txt

- FOR TFLITE MODEL

        pip install -r requirements_tflite.txt

Download Data:
        
        Note: Kindly download your kaggle.json file from kaggle.com to download the dataset

        python download_data.py --json_file "kaggle.json"

- DATA TREE

        |-- data
                |-- Cars Dataset
                                |--train
                                        Audi
                                        Hyundai Creta
                                        ...
                                |--test
                                        Audi
                                        Hyundai Creta
                                        ...



Usage:

- TRAIN 

        python main.py --task "train" --train_path "data/Cars Dataset/train" --test_path "data/Cars Dataset/test" --model_name "mobilenetv3" --epochs=20

        You can also train with "custom" method

- TEST

        python main.py --task "test" --model_name "mobilenetv3" --test_path "data/Cars Dataset/test" --model_weights "weights/best_model.pth"

- GENERATE TFLITE MODEL ALONG WITH TRAIN

        python main.py --task "train" --train_path "data/Cars Dataset/train" --test_path "data/Cars Dataset/test" --model_name "mobilenetv3" --epochs=20 --tflite_model True

- MANUAL TFLITE Conversion:

        onnx2tf -i "weights/model_best.onnx"

- ONNX_MODEL_TEST_ACCURACY_AND_INFERENCE_TIME

        python test_onnx.py --onnx_model_path "weights/model_best.onnx" --test_dir "data/Cars Dataset/test"

- TFLITE_MODEL_TEST_ACCURACY_AND_INFERENCE_TIME

        python test_tflite.py --tflite_model_path "saved_model/model_best_float16.tflite" --test_dir "data/Cars Dataset/test"

