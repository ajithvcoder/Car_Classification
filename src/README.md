### Car Classification Model

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

        python main.py --task "train" --train_path "data/Cars Dataset/train" --test_path "data/Cars Dataset/test" --model_name "mobilenetv3"

        You can also train with "custom" method

- TEST

        python main.py --task "test" --test_path "data/Cars Dataset/test" --model_weights "weights/best_model.pth"

- GENERATE TFLITE MODEL ALONG WITH TRAIN

        python main.py --task "train" --train_path "data/Cars Dataset/train" --test_path "data/Cars Dataset/test" --model_name "mobilenetv3" --tflite_model True

- MANUAL TFLITE Conversion:

        onnx2tf -i "weights/model_best.onnx"

- ONNX_MODEL_TEST_ACCURACY_AND_INFERENCE_TIME

        python test_onnx.py --onnx_model_path "weights/best_model.onnx" --test_dir "data/Cars Dataset/test"

- TFLITE_MODEL_TEST_ACCURACY_AND_INFERENCE_TIME

        python test_tfile.py --tfilte_model_path "saved_models/best_model.tflite" --test_dir "data/Cars Dataset/test"


