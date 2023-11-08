import sys

from flask import Flask, render_template,request
import pickle
import numpy as np
import sklearn
import joblib
import spacy
import warnings
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from args import get_parser
from model import get_model
from torchvision import transforms
from utils.output_utils import prepare_output
from PIL import Image
import time
import requests
from io import BytesIO

from sklearn.neighbors import DistanceMetric

sys.modules['sklearn.neighbors._distance_metric'] = sklearn.neighbors._dist_metrics
# model = joblib.load('my_model.sav')
model = pickle.load(open('my_model.sav', 'rb'))
app = Flask(__name__)

use_gpu = True
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'
show_anyways = False
use_urls = True
image_folder = os.path.join('demo_imgs')

ingrs_vocab = pickle.load(open('ingr_vocab.pkl', 'rb'))
vocab = pickle.load(open('instr_vocab.pkl', 'rb'))
ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)
output_dim = instrs_vocab_size

args = get_parser()
args.maxseqlen = 15
args.ingrs_only=False
model2 = get_model(args, ingr_vocab_size, instrs_vocab_size)
model2_path = 'modelbest.ckpt'
model2.load_state_dict(torch.load(model2_path, map_location=map_loc))
model2.to(device)
model2.eval()
model2.ingrs_only = False
model2.recipe_only = False

transf_list_batch = []
transf_list_batch.append(transforms.ToTensor())
transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225)))
to_input_transf = transforms.Compose(transf_list_batch)

greedy = [True, False, False, False]
beam = [-1, -1, -1, -1]
temperature = 1.0
numgens = len(greedy)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictionpage')
def predictionpage():
    return render_template('predictionpage.html')

@app.route('/predictionimage')
def predictionimage():
    return render_template('predictionimage.html')

@app.route('/predict', methods=['POST'])
def predict_cuisine():
    ingredients = str(request.form.get('ingredients'))
    my_ing = ingredients.split(',')

    vecList = []
    nlp = spacy.load("en_core_web_lg")
    for i in range(0, len(my_ing)):
        myText = my_ing
        myString = " "
        myText = myString.join(myText)

    doc = nlp(myText)

    for token in doc:
        vecList.append(token.vector)

    vecList = np.array(vecList)
    avgWordVec = np.mean(vecList, axis=0)
    avgWordVec = avgWordVec.reshape(1, -1)

    result = model.predict(avgWordVec)

    return render_template('predictionpage.html', result=[my_ing, result[0]])


@app.route('/predict2', methods=['POST'])
def predict_recipes():
    demo_urls = []
    url = request.form.get('imagelink')

    demo_urls.append(url)
    demo_files = demo_urls

    for img_file in demo_files:

        response = requests.get(img_file)
        image = Image.open(BytesIO(response.content))

        transf_list = []
        transf_list.append(transforms.Resize(256))
        transf_list.append(transforms.CenterCrop(224))
        transform = transforms.Compose(transf_list)

        image_transf = transform(image)
        image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

        num_valid = 1
        result2 = []
        for i in range(numgens):
            with torch.no_grad():
                outputs = model2.sample(image_tensor, greedy=greedy[i],
                                        temperature=temperature, beam=beam[i], true_ingrs=None)

            ingr_ids = outputs['ingr_ids'].cpu().numpy()
            recipe_ids = outputs['recipe_ids'].cpu().numpy()

            outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)

            if valid['is_valid'] or show_anyways:
                var2 = outs['title']
                var3 = ', '.join(outs['ingrs'])
                var7 = '-' + '\n-'.join(outs['recipe'])

                result2.append([var2, var3, var7, url])

    return render_template("predictionimage.html", result=result2)


if __name__ == '__main__':
    app.run(debug=True)
