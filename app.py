from flask import Flask, request, jsonify, render_template
import numpy as np
import onnxruntime
from PIL import Image
import time

app = Flask(__name__)

MODEL_PATH = "./deployed_model/model_best.onnx"

# Load the ONNX model
sess = onnxruntime.InferenceSession(MODEL_PATH)

classes = ["Make: Audi - Model: Any", "Make: Hyundai - Model:Creta", "Make: Mahindra - Model:Scorpio", "Make: Rolls Royce - Model:Any", \
           "Make: Maruti Suzuki - Model: Swift","Make: Tata - Model: Safari", "Make: Toyota - Model: Innova"]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Normalization parameters
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict(file):
    try:
        image = Image.open(file).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = (image - MEAN) / STD
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :].astype(np.float32)        
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        pred = sess.run([output_name], {input_name: image})[0]
        predicted_class = classes[np.argmax(pred)]

        return jsonify({'class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/display', methods=['POST'])
def display():
    if request.method == 'POST':
        try:
            file = request.files['image']
            if allowed_file(file.filename) and file:
                start_time = time.time()
                response = predict(file)
                end_time = time.time()
                data = response.get_json()
                inference_time = end_time - start_time
                data['process_time'] = f"{inference_time:.2f} seconds [preprocess + infer + postprocess]"
                return render_template('display.html', result=data)
            else:
                raise Exception("Not a valid image file: supported 'png', 'jpg', 'jpeg'")
        except Exception as e:
            return render_template('display.html', result={'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
