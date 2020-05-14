# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request, jsonify
import sqlite3
import ssl
import pandas as pd
import sys
import logging

model = None
app = Flask(__name__)
app.config["DEBUG"] = True


def load_model():
    global model
    # model variable refers to the global variable
    with open('active_model_tfidf_SVM.sav', 'rb') as f:
        model = pickle.load(f)

'''
@app.route('/', methods=['GET'])
def home_endpoint():
    return jsonify({'text': 'hello, world!'})
'''

@app.route('/query', methods=['POST'])
def get_query():
    # Works only for a single sample
    if request.method == 'POST':
        csv_file = request.files['file']
        try:
            df = pd.read_csv(csv_file, header=0)
            df.dropna(inplace=True)
            df['data_reuse'] = None

            print(df.shape)
            print(df.head(10))
            print("Count of missing values: \n" + str(df.isnull().sum()))

            
        except:
            return jsonify({'Reason': 'Error: ' + str(sys.exc_info())})

    #print(request.files['file'])
    return jsonify({'Test': 'Hello, World!'})

if __name__ == '__main__':
    # Set up SSL
    #context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    #context.load_cert_chain("cert.pem", "key.pem")

    load_model()  # load model at the beginning once only
    #app.run(threaded=True, host='192.168.1.152', port=8080)
    app.run(threaded=True, host='0.0.0.0', port=8080)
    #app.run(threaded=True, host='0.0.0.0', port=8443, ssl_context='adhoc')
    #app.run(threaded=True, host='0.0.0.0', port=8443, ssl_context=context)
