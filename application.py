import pandas as pd
from flask import Flask, jsonify, request
#import pickle
from net_architecture import CRNN_model, CRNN_MODE
from net_config import FilePaths
from utils import Sample
from utils import load_data, TextSequenceGenerator, decode_predict_ctc, labels_to_text
import cv2
import numpy as np
from net_config import ArchitectureConfig
from data_preprocess import process_image_for_ocr
import json

# load model
#model = pickle.load(open('model.pkl','rb'))


FilePaths.fnSave = 'crnn_160ts.h5'

model = CRNN_model(CRNN_MODE.inference)
model.load_weights(FilePaths.fnSave)

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    input_path = "input.jpg"
    input_pre_path = "input_pre.jpg"
    try:
        # check if the post request has the file part
        filestr = request.files['image'].read()
        # convert string data to numpy array
        npimg = np.fromstring(filestr, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        cv2.imwrite(input_path, img)

        process_image_for_ocr(input_path, input_pre_path)
        data = [Sample("", input_pre_path)]
        test_set = TextSequenceGenerator(data)
        samples = test_set[0]
        img = samples[0]['the_input'][0]

        # img = image.load_img(new_file_path, target_size=ArchitectureConfig.IMG_SIZE[::-1], interpolation='bicubic')
        # img = image.img_to_array(img)
        # img = preprocess_input(img)
        # img = img.transpose((1, 0, 2))
        img = np.expand_dims(img, axis=0)
        img = img.transpose((0, 1, 2, 3))

        net_out_value = model.predict(img)
        pred_texts = decode_predict_ctc(net_out_value, ArchitectureConfig.CHARS)
        output = {'results': pred_texts[0]}


    except Exception as err:
        print(err)
        output = {'results': "Error!"}


    # convert data into dataframe
    #data.update((x, [y]) for x, y in data.items())
    #data_df = pd.DataFrame.from_dict(data)

    # predictions
    #result = model.predict(data_df)

    # send back to browser
    #output = {'results': 'OK'}

    # return data
    #return jsonify(results=output)
    print(output)
    return json.dumps(output, ensure_ascii=False).encode('utf8')

if __name__ == '__main__':
    app.run(port = 5000, debug=True)