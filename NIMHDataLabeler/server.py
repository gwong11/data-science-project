# Serve model as a flask application

import pickle
import numpy as np
import sqlite3
import ssl
import pandas as pd
import sys
import os
import hashlib
#import logging
import spacy
from html import unescape
from flask import Flask, request, jsonify, g,  make_response
from logging.config import fileConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from ingest import DatabaseIngest
from contextlib import closing

model = None
app = Flask(__name__)
app.config["DEBUG"] = True
#logging.basicConfig(filename='labeler-app.log', 
#                    level=logging.DEBUG,
#                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
# app.logger.info

DATABASE = 'research.db'
DATABASE_TEMP = 'research_temp.db'

# create a custom analyzer class
class MyAnalyzer(object):

    # load spaCy's English model and define the tokenizer/lemmatizer
    def __init__(self):
        spacy.load('en_core_web_sm')
        self.lemmatizer_ = spacy.lang.en.English()

    # allow the class instance to be called just like
    # just like a function and applies the preprocessing and
    # tokenize the document
    def __call__(self, doc):
        doc_clean = unescape(doc).lower()
        tokens = self.lemmatizer_(doc_clean)
        return([token.lemma_ for token in tokens])

def vectorize(method, analyzer, data):
    vect = None
    if method.lower() == "tfidf":
        vect = TfidfVectorizer(analyzer=analyzer)
        return vectorizer.fit_transform(data)

def load_model():
    global model
    global vectorizer

    # vectorizer variable refers to the global variable
    with open("vectorizer.pkl", 'rb') as fd:
        vectorizer  = pickle.load(fd)

    # model variable refers to the global variable
    with open('active_model_tfidf_SVM.sav', 'rb') as fd2:
        model = pickle.load(fd2)
    

@app.before_request
def before_request():
    # Initiate database instance
    g.db = DatabaseIngest(DATABASE)
    #d.db_tmp = DatabaseIngest(DATABASE_TEMP)
    # Initiate analyzer instance for vectorizer
    analyzer = MyAnalyzer()
    global new
    if not os.path.exists(DATABASE):
        # create connection
        g.db.create_connection()
        # create table
        g.db.create_table('research', """
                                      Date,
                                      PMID NOT NULL,
                                      text PRIMARY KEY NOT NULL,
                                      data,
                                      data_reuse
                                      """)
        new = True
    else:
       g.db.create_connection()
       new = False
       

@app.teardown_request
def teardown_request(exception):
    if hasattr(g, 'db'):
        g.db.close()

'''
@app.route('/', methods=['GET'])
def home_endpoint():
    return jsonify({'text': 'hello, world!'})
'''
@app.route('/api/v1/query', methods=['POST'])
def get_query():
    # Works only for a single sample
    if request.method == 'POST':
        jsonData = []
        csv_file = request.files['file']
        try:
            df = pd.read_csv(csv_file, header=0)
            df.dropna(inplace=True)
            df.drop_duplicates(subset='text', keep='last', inplace=True)
            #df['text_md5'] = df.apply(lambda row: hashlib.md5(row.text.encode('utf-8')).digest(), axis=1)
            df['data_reuse'] = None
            #print(df.shape)
            #print(df.groupby(['text']).count())

            #df.loc[0, 'data_reuse'] = 1
            #print(df['text'].iloc[0])
            #print(df.head(10))
            #print("Count of missing values: \n" + str(df.isnull().sum()))
            if new:
                app.logger.info("Creating the database: research")
                g.db.insert_record('research', 'append', 1000, df)
                app.logger.info("Done")
            else:
                #app.logger.info("Inserting new records to research")
                #g.db.insert_record_unique('research', 'append', 1000, df)
                #app.logger.info("Done")
                pass

            # Vectorize the text
            X = vectorizer.fit_transform(df['text'])
            # Save vectorizer
            pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

            #print(X.toarray())
            #print(X.shape)
            #ordered_vocab = dict(sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:100])
            #print(str(ordered_vocab))
            for i in range(15):
                query_idx, query_inst = model.query(X)
                jsonData.append({'text': df['text'].iloc[query_idx[0]]})
                X = csr_matrix(np.delete(X.toarray(), query_idx, axis=0))

        except:
            return jsonify({'Reason': 'Error: ' + str(sys.exc_info())})

    #print(request.files['file'])
    return jsonify(jsonData)

@app.route('/api/v1/label', methods=['POST'])
def submit():
    data = request.json
    data['text'] = data['text'].replace("'","''")
    SQL_QUERY = "SELECT * FROM research WHERE text LIKE '%" + data['text'] + "%';"
    check_df = g.db.query_record(SQL_QUERY)
    #print(check_df['data_reuse'].values[0])
    if 'label' in data:
        if check_df['data_reuse'].values[0] is None:
            if data['label'] != None:
                if data['label'] in [0,1]:
                    try:
                        app.logger.info("Updating record with label!")
                        g.db.update_record('research', 'data_reuse', 'text', data['label'], data['text'])
                        app.logger.info("Done")
                        response = {'message': 'Label successfully captured!', 'code': 'SUCCESS'}
                        return make_response(jsonify(response), 200)
                    except:
                        return make_response(jsonify({'message': 'Error: ' + str(sys.exc_info()), 'code': 'FAILURE'}))
                elif int(data['label']) in [2]:
                    return make_response(jsonify({'message': 'Quiting labeling process', 'code': 'QUIT'}))
                else:
                    return make_response(jsonify({'message': 'Incorrect input', 'code': 'FAILURE'}))
            else:
                return make_response(jsonify({'message': 'Need to enter a value of 0 or 1.', 'code': 'FAILURE'}))
        else:
            return make_response(jsonify({'message': 'Already labeled. Skipping.', 'code': 'LABEL'}))

    return make_response(jsonify({'message': 'Missing label', 'code': 'FAILURE'}), 400)

if __name__ == '__main__':
    fileConfig('logging.cfg')

    # Set up SSL
    #context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    #context.load_cert_chain("cert.pem", "key.pem")

    load_model()  # load model at the beginning once only

    #app.run(threaded=True, host='192.168.1.152', port=8080)
    app.run(threaded=True, host='0.0.0.0', port=8080)
    #app.run(threaded=True, host='0.0.0.0', port=8443, ssl_context='adhoc')
    #app.run(threaded=True, host='0.0.0.0', port=8443, ssl_context=context)
