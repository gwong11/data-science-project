# Serve model as a flask application

import pickle
import numpy as np
import sqlite3
import ssl
import pandas as pd
import sys
import os
import re
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

VECTORIZER_RE = re.compile(r'.*vect.*', re.VERBOSE)
MODEL_RE = re.compile(r'.*model.*', re.VERBOSE)

app = Flask(__name__)
app.config["DEBUG"] = True
#logging.basicConfig(filename='labeler-app.log', 
#                    level=logging.DEBUG,
#                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
# app.logger.info

DATABASE = 'research.db'

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

'''
def load_model():

    # vectorizer variable refers to the global variable
    with open("vectorizer.pkl", 'rb') as fd:
        vectorizer  = pickle.load(fd)

    # model variable refers to the global variable
    with open('active_model_tfidf_SVM.sav', 'rb') as fd2:
        model = pickle.load(fd2)
'''

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

        g.db.create_table('research_tmp', """
                                      Date,
                                      PMID NOT NULL,
                                      text PRIMARY KEY NOT NULL,
                                      data,
                                      data_reuse
                                      """)

        g.db.create_table('setting', """
                                      type PRIMARY KEY NOT NULL,
                                      value NOT NULL
                                      """)

        new = True
    else:
       g.db.create_connection()
       new = False

@app.teardown_request
def teardown_request(exception):
    if hasattr(g, 'db'):
        g.db.close()

@app.route('/api/v1/setting', methods=['POST'])
def set_config():
    global model
    global vectorizer
    global numberOfQueries
    global modelName
    global vectorizerName

    upload_file = None
    data = None
    df = None
    set_vec = False
    set_mod = False
    numberOfQueries = 10
    if request.method == 'POST':
       if 'file' in request.files:
          upload_file = request.files['file']
          #print(upload_file.filename)
       try:
           SQL_NUMBER_QUERY = "SELECT * FROM setting WHERE type LIKE '%Queries%';"
           check_queries_df = g.db.query_record(SQL_NUMBER_QUERY)

           if request.json is not None:
               numberOfQueries = request.json['value']['numberOfQueries']

           if upload_file is not None:
              SQL_VEC_QUERY = "SELECT * FROM setting WHERE value LIKE '%vect%';"
              SQL_MODEL_QUERY = "SELECT * FROM setting WHERE value LIKE '%model%';"
              check_vect_df = g.db.query_record(SQL_VEC_QUERY)
              check_model_df = g.db.query_record(SQL_MODEL_QUERY)

              if check_vect_df['type'].values[0] == "vectorizer":
                 set_vec = True
                 vectorizerName = check_vect_df['value'].values[0]

                 with open(vectorizerName, 'rb') as fd:
                    vectorizer  = pickle.load(fd)

              if check_model_df['type'].values[0] == "model":
                 set_mod = True
                 modelName = check_model_df['value'].values[0]

                 with open(modelName, 'rb') as fd:
                    model  = pickle.load(fd)

              if re.match(VECTORIZER_RE, upload_file.filename) is not None and set_vec is not True:
                 app.logger.info("Setting vectorizer.")
                 data = [['vectorizer', upload_file.filename]]
                 vectorizerName = upload_file.filename
                 df = pd.DataFrame(data, columns = ['type', 'value']) 
                 g.db.insert_record('setting', 'append', 10, df)

                 fd2 = open(upload_file.filename, 'wb')
                 pickle.dump(vectorizer, fd2)
                 fd2.close()

                 with open(upload_file.filename, 'rb') as fd:
                    vectorizer  = pickle.load(fd)
              
              elif re.match(MODEL_RE, upload_file.filename) is not None and set_mod is not True:
                 app.logger.info("Setting Model.")
                 data = [['model', upload_file.filename]]
                 modelName = upload_file.filename
                 df = pd.DataFrame(data, columns = ['type', 'value'])
                 g.db.insert_record('setting', 'append', 10, df)

                 fd2 = open(upload_file.filename, 'wb')
                 pickle.dump(model, fd2)
                 fd2.close()

                 with open(upload_file.filename, 'rb') as fd:
                    model  = pickle.load(fd)

              elif set_vec is True or set_mod is True:
                 response = {'message': 'Already loaded!!', 'code': 'SUCCESS'}
                 return make_response(jsonify(response), 200)
              else:
                 return jsonify({'Reason': 'Please upload files that contain either vect (for vectorizer) or model (for model)'})
           elif numberOfQueries == 10 and len(check_queries_df['type'].values) == 0:
              app.logger.info("Setting numberOfQueries.")
              data = [['numberOfQueries', numberOfQueries]]
              df = pd.DataFrame(data, columns = ['type', 'value'])
              g.db.insert_record('setting', 'append', 10, df)
           elif numberOfQueries != 10:
              if(len(check_queries_df['type'].values) != 0):
                app.logger.info("Deleting numberOfQueries")
                g.db.delete_single_record('setting', 'type', 'numberOfQueries')

              app.logger.info("Setting numberOfQueries.")
              data = [['numberOfQueries', numberOfQueries]]
              df = pd.DataFrame(data, columns = ['type', 'value'])
              g.db.insert_record('setting', 'append', 10, df)
            
           app.logger.info("Successfully loaded!")
           response = {'message': 'Successfully loaded!!', 'code': 'SUCCESS'}
           return make_response(jsonify(response), 200)
       except: 
           return jsonify({'Reason': 'Error: ' + str(sys.exc_info())})

'''
@app.route('/', methods=['GET'])
def home_endpoint():
    return jsonify({'text': 'hello, world!'})
'''
@app.route('/api/v1/query', methods=['POST'])
def get_query():
    # Works only for a single sample
    global query_idx
    global query_inst
    setVectorizer = False
    setModel = False

    if request.method == 'POST':
        #print(numberOfQueries)
        if 'vectorizer' in globals():
            setVectorizer = True

        if 'model' in globals():
            setModel = True

        jsonData = []
        csv_file = request.files['file']

        if setVectorizer != False and setModel != False:
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
               SQL_COUNT_QUERY = "SELECT COUNT(*) FROM research;"
               check_count_df = g.db.query_record(SQL_COUNT_QUERY)
               if check_count_df["COUNT(*)"].values[0] == 0:
                  app.logger.info("Creating the database: research")
                  g.db.insert_record('research', 'append', 1000, df)
                  app.logger.info("Done")
               else:
                  print("Adding new record")
                  #app.logger.info("Inserting new records to research")
                  #g.db.insert_record('research_tmp', 'append', 1000, df)
                  #g.db.insert_record_unique('research_tmp', 'research')
                  #g.db.delete_record('research_tmp')
                  #app.logger.info("Done")

               # Vectorize the text
               X = vectorizer.fit_transform(df['text'])
               # Save vectorizer
               fd = open(vectorizerName, "wb")
               pickle.dump(vectorizer, fd)
               fd.close()

               #print(X.toarray())
               #print(X.shape)
               #ordered_vocab = dict(sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:100])
               #print(str(ordered_vocab))
               for i in range(int(numberOfQueries)):
                  query_idx, query_inst = model.query(X)
                  jsonData.append({'text': df['text'].iloc[query_idx[0]]})
                  X = csr_matrix(np.delete(X.toarray(), query_idx, axis=0))

               #print(request.files['file'])
               return jsonify(jsonData)
            except:
               return jsonify({'Reason': 'Error: ' + str(sys.exc_info())})
        else:
            return jsonify({'Reason': 'Make sure vectorizer, and model are configured.'})

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
                        app.logger.info("Training model with label.")
                        #print(query_inst)
                        #model.teach(query_inst.reshape(1, -1), data['label'])

                        # Save model
                        #fd = open(modelName, "wb")
                        #pickle.dump(model, fd)
                        #fd.close()

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

    #load_model()  # load model at the beginning once only

    #app.run(threaded=True, host='192.168.1.152', port=8080)
    app.run(threaded=True, host='0.0.0.0', port=8080)
    #app.run(threaded=True, host='0.0.0.0', port=8443, ssl_context='adhoc')
    #app.run(threaded=True, host='0.0.0.0', port=8443, ssl_context=context)
