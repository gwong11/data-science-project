import json
import os
import re
import string
import pandas as pd
import numpy as np
import sys 
import time
import pickle
import spacy
import collections
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm
from spacy.tokenizer import Tokenizer
from nltk.tokenize import sent_tokenize
from PIL import Image
from html import unescape
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
#from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from scipy.sparse import csr_matrix
from modAL.models import ActiveLearner
from modAL.uncertainty import classifier_entropy
from modAL.uncertainty import classifier_margin
from modAL.uncertainty import classifier_uncertainty
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import margin_sampling
from modAL.uncertainty import uncertainty_sampling

#pd.options.display.max_columns = None
#pd.set_option('display.max_colwidth', -1)
#pd.set_option('display.max_rows', None)

FILE_REGEX = re.compile(r'NIMH_research_papers_[0-9]+\.json', re.VERBOSE)
DATA_REGEX1 = re.compile(r'(github)', re.VERBOSE) 
DATA_REGEX2 = re.compile(r'(osf\.io)', re.VERBOSE) 
DATA_REGEX3 = re.compile(r'(nda\.nih\.gov)', re.VERBOSE) 
DATA_REGEX4 = re.compile(r'(openneuro)', re.VERBOSE)
DATA_REGEX5 = re.compile(r'[\s\'\"]+(ndar)[\s\'\"]+[?.!]*', re.VERBOSE)
DATA_REGEX6 = re.compile(r'(national\sdatabase\sfor\sautism\sresearch)', re.VERBOSE) 
DATA_REGEX7 = re.compile(r'(brain-map\.org)', re.VERBOSE)
DATA_REGEX8 = re.compile(r'(humanconnectome\.org)', re.VERBOSE)
DATA_REGEX9 = re.compile(r'(balsa\.wustl\.edu)', re.VERBOSE)
DATA_REGEX10 = re.compile(r'(loni\.usc\.edu)', re.VERBOSE)
DATA_REGEX11 = re.compile(r'(ida\.loni\.usc\.edu)', re.VERBOSE)
DATA_REGEX12 = re.compile(r'(fmridc)', re.VERBOSE)
DATA_REGEX13 = re.compile(r'(ccrns)', re.VERBOSE)
DATA_REGEX14 = re.compile(r'(datalad)', re.VERBOSE)
DATA_REGEX15 = re.compile(r'(dataverse)', re.VERBOSE)
DATA_REGEX16 = re.compile(r'(dbgap)', re.VERBOSE)
DATA_REGEX17 = re.compile(r'(nih\.gov\/gap)', re.VERBOSE)
DATA_REGEX18 = re.compile(r'(dryad)', re.VERBOSE)
DATA_REGEX19 = re.compile(r'(figshare)', re.VERBOSE)
DATA_REGEX20 = re.compile(r'(fcon_1000\.projects)', re.VERBOSE)
DATA_REGEX21 = re.compile(r'(nitrc)', re.VERBOSE)
DATA_REGEX22 = re.compile(r'(mcgill\.ca\/bic\/resources\/omega)', re.VERBOSE)
DATA_REGEX23 = re.compile(r'(xnat\.org)', re.VERBOSE) 
DATA_REGEX24 = re.compile(r'(zenodo)', re.VERBOSE)
DATA_REGEX25 = re.compile(r'(opendata\.aws)', re.VERBOSE)

#PREFIX_RE = re.compile(r'''^[\[\("']''', re.VERBOSE)
#SUFFIX_RE = re.compile(r'''[\]\)"']$''', re.VERBOSE)
#INFIX_RE = re.compile(r'''[-~]''', re.VERBOSE)
#SIMPLE_URL_RE = re.compile(r'''^(https?:/?/? ? ?)?(www)?''', re.VERBOSE)

def remove_punc(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

# create a custom analyzer class
class MyAnalyzer(object):

    # load spaCy's English model and define the tokenizer/lemmatizer
    def __init__(self):
        spacy.load('en')
        self.lemmatizer_ = spacy.lang.en.English()

    # allow the class instance to be called just like
    # just like a function and applies the preprocessing and
    # tokenize the document
    def __call__(self, doc):
        doc_clean = unescape(doc).lower()
        tokens = self.lemmatizer_(doc_clean)
        return([token.lemma_ for token in tokens])

def test_suitability(passage):
    #figures out whether a passage is suitable for what we're after.
    #probably still needs some fiddling.
    suitability = True
    try:
        if passage['infons']['section_type'].upper() == 'REF':
            suitability = False

        if passage['infons']['section_type'].upper() == 'TITLE':
            suitability = False

        if passage['infons']['type'].upper() == 'TABLE':
            suitability = False

        if 'TITLE' in passage['infons']['type'].upper():
            suitability = False
    except:
        suitability = False

    return(suitability)

'''
def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, prefix_search=PREFIX_RE.search,
                                suffix_search=SUFFIX_RE.search,
                                infix_finditer=INFIX_RE.finditer,
                                token_match=SIMPLE_URL_RE.match)
                                '''

# create a dataframe from a word matrix
def wm2df(wm, feat_names):

    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return(df)

def filter_text():
    data_dir = '/Users/G/git/repository/data-science-project/data/papers'
    files = sorted(os.listdir(data_dir))
    papers = pd.DataFrame(columns=['Date', 'PMID', 'text', 'data'])
    #papers = papers.append(pd.DataFrame({'Date': [20191214], 'PMID': ['00000000000'], 'text': ['This is a test']}), ignore_index = True)

    #no_match_filename = "/Users/G/git/repository/data-science-project/data/NIMH_pmids_noMatchRegex.txt"

    count = 0
    paper_count = 0
    for f in files:
        if re.match(FILE_REGEX, f) is not None:
            with open(os.path.join(data_dir, f)) as json_file:
                data = json.load(json_file)

            df = pd.io.json.json_normalize(data['papers'], record_path='documents', meta=['date'])
            print("Processed filename: " + f)

            for row in df.itertuples():
                pmid = ''
                first = True
                #print("Paper " + str(paper_count))
                for passage in row.passages:
                    try:
                        if first is True:
                            pmid = passage['infons']['article-id_pmid']
                            first = False

                        suitability = test_suitability(passage)
                        if suitability is True:
                            sentences = sent_tokenize(passage['text'])
                            for s in sentences:
                                #s = remove_punc(s)
                                m = (re.search(DATA_REGEX1, s.lower()) or re.search(DATA_REGEX2, s.lower()) or re.search(DATA_REGEX3, s.lower()) or 
                                        re.search(DATA_REGEX4, s.lower()) or re.search(DATA_REGEX5, s.lower()) or re.search(DATA_REGEX6, s.lower()) or 
                                        re.search(DATA_REGEX7, s.lower()) or re.search(DATA_REGEX8, s.lower()) or re.search(DATA_REGEX9, s.lower()) or 
                                        re.search(DATA_REGEX10, s.lower()) or re.search(DATA_REGEX11, s.lower()) or re.search(DATA_REGEX12, s.lower()) or 
                                        re.search(DATA_REGEX13, s.lower()) or re.search(DATA_REGEX14, s.lower()) or re.search(DATA_REGEX15, s.lower()) or 
                                        re.search(DATA_REGEX16, s.lower()) or re.search(DATA_REGEX17, s.lower()) or re.search(DATA_REGEX18, s.lower()) or
                                        re.search(DATA_REGEX19, s.lower()) or re.search(DATA_REGEX20, s.lower()) or re.search(DATA_REGEX21, s.lower()) or 
                                        re.search(DATA_REGEX22, s.lower()) or re.search(DATA_REGEX23, s.lower()) or re.search(DATA_REGEX24, s.lower()) or 
                                        re.search(DATA_REGEX25, s.lower()) )
                                if m is not None:
                                    papers = papers.append(pd.DataFrame({'Date': [row.date], 'PMID': [pmid], 'text': [s.strip()], 'data': [m.group(1)]}), 
                                            ignore_index = True)
                        #else:
                        #    print(passage['infons']['section_type'] + ',' + passage['infons']['type'] + ',' + passage['text'])
                    except:
                        continue
                paper_count += 1
            print("   - " + str(paper_count))
            paper_count = 0
            #if count == 4:
            #    print(papers.head())
            #    #print(papers.groupby(['PMID']).count())
            #    print(papers['text'][2])
            #    exit(1)
            count += 1
    #print(papers.head())
    #print(len(papers))
    #print(papers.groupby(['PMID']).count())
    #print(papers['text'][0])

    # pickle dataframe 
    papers.to_pickle("/Users/G/git/repository/data-science-project/data/NIMH_pmids_matched.pkl")

def read_df(filename="/Users/G/git/repository/data-science-project/data/NIMH_pmids_matched.pkl"):
    unpickled_df = pd.read_pickle(filename)
    print(unpickled_df.head())
    print(unpickled_df.tail())
    print(len(unpickled_df['PMID']))
    print(unpickled_df.groupby(['PMID']).count())
    print(unpickled_df['text'][4])

def write_df_to_csv(output_filename, filename="/Users/G/git/repository/data-science-project/data/NIMH_pmids_matched.pkl"):
    unpickled_df = pd.read_pickle(filename)
    unpickled_df.to_csv(output_filename, index=False)

def method_ML(model, ML_method, vectorizer_method, X_train, X_test, y_train, y_test, vis_filename, model_filename):
    # Fit the model on training set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy (" + vectorizer_method + "):" , score)
    print("Confusion matrix : ") 
    print(cm)
    print("Classification report (" + vectorizer_method + "):")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label = '%s (area = %0.2f)' % (ML_method, roc_auc))
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(vis_filename)
    print("Graph saved: ", vis_filename)
    print()
    #plt.show()
    plt.close()
    
    fd = open(model_filename, 'wb')
    pickle.dump(model, fd)
    fd.close()
    print("Model saved: ", model_filename)
    print()

def compare_models(models, seed, X, y, vectorizer):
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig('/Users/G/Loyola/Spring2020/DS796/compare_models_boxplot_' + vectorizer + '.png')
    print('Graph saved: /Users/G/Loyola/Spring2020/DS796/compare_models_boxplot_' + vectorizer + '.png')
    print()
    #plt.show()
    plt.close()

def prepare_data(filename, verbose=False):

    #fd = open(filename, 'r')
    #fd.close()
    df = pd.read_csv(filename, header=0)
    new_df = df[['Year', 'PMID', 'data_reuse', 'text']]
    #print(new_df.head(10))
    #print(new_df.tail(10))

    # Total number of records (with nulls)
    print("Count of records (with nulls): \n" + str(new_df.data_reuse.count()))
    print()

    # Checking missing values
    print("Count of missing values: \n" + str(new_df.isnull().sum()))
    print()
    print("Rows where text is missing: ")
    print(new_df[new_df['text'].isnull()])
    print()

    # Total number of records (without nulls)
    new_df2 = new_df.dropna(inplace=False)
    print("Count of records (without nulls): \n" + str(new_df2.data_reuse.count()))
    print()

    if verbose:
        # Balance of data
        print("Count by data_reuse: \n\n" + str(new_df2.groupby('data_reuse').count()))
        print()
        print("Normalized percentage by data_reuse: \n\n" + str(new_df2['data_reuse'].value_counts(normalize=True)))
        print()
        print("Count by pmid: \n\n" + str(new_df2.groupby('PMID').count()))
        print()
        print("Normalized percentage by pmid: \n\n" + str(new_df2['PMID'].value_counts(normalize=True).head(10)))
        print()
        print("Count by Year: \n\n: " + str(new_df2.groupby('Year').count()))
        print()
        print("Normalized percentage by Year: \n\n" + str(new_df2['Year'].value_counts(normalize=True)))
        print()
        print("Count of data_reuse by Year: \n\n" + str(new_df2.groupby(['Year', 'data_reuse']).count()))
        print()

        # Visualize
        plt.figure(figsize=(40,30))
        new_df2['data_reuse'].value_counts().plot(kind='bar')
        plt.xticks([0, 1], ('No', 'Yes'))
        plt.title("Number of data_reuse", fontsize=100)
        plt.xticks(fontsize=50)
        plt.yticks(fontsize=50)
        plt.xlabel("data_reuse", fontsize=70)
        plt.ylabel("count", fontsize=70)
        plt.savefig('/Users/G/Loyola/Spring2020/DS796/count_data_reuse_graph.png')
        print("Graph saved: /Users/G/Loyola/Spring2020/DS796/count_data_reuse_graph.png")
        print()
        #plt.show()

        plt.figure(figsize=(40, 30))
        new_df2.groupby(['Year', 'data_reuse'])['text'].count().unstack().plot(kind='bar', stacked=True)
        plt.title("Number of data_reuse by Year", fontsize=15)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("count", fontsize=12)
        plt.savefig('/Users/G/Loyola/Spring2020/DS796/count_data_reuse_year_graph.png')
        print("Graph saved: /Users/G/Loyola/Spring2020/DS796/count_data_reuse_year_graph.png")
        print()
        #plt.show()
        plt.close()

    return new_df2

def model(filename, model_type):
    try:

        # Initialize custom analyzer
        analyzer = MyAnalyzer()

        new_df2 = prepare_data(filename, True)

        # Split the data into train and test dataset
        X = np.array(new_df2['text'])
        y = np.array(new_df2['data_reuse'])
        print("X shape: " + str(X.shape))
        print("y shape: " + str(y.shape))
        print()
        
        sentences_train, sentences_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        print("Reuse (train): ")
        print(collections.Counter(y_train))
        print()
        print("Reuse (test): ")
        print(collections.Counter(y_test))
        print()

        # Tokenization/Normalization/Word Embedding
        # CountVectorizer
        # Shape without Custom tokenizer - train (1502, 7103)
        # Shape without Custom tokenizer - test (376, 7103)

        # Shape with Custom tokenizer - train (1502, 7553)
        # Shape with Custom tokenizer - test (376, 7553)

        #vectorizer_count = CountVectorizer()
        vectorizer_count = CountVectorizer(analyzer=analyzer)
        #sentence_vectors_count = vectorizer_count.fit_transform(new_df2['text'])
        #vectorizer_count.fit(sentences_train)
        vectorizer_count.fit(X)
        count_X_train = vectorizer_count.transform(sentences_train)
        count_X_test = vectorizer_count.transform(sentences_test)
        print("Count Vectorizer (train): ")
        print(count_X_train.toarray())
        print(str(count_X_train.shape))
        print()
        print("Count Vectorizer (test): ")
        print(count_X_test.toarray())
        print(str(count_X_test.shape))
        print()

        print("Vocabulary: ")
        ordered_count = dict(sorted(vectorizer_count.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:1000])
        print(str(ordered_count))
        print()

        #print(len(vectorizer_count.get_feature_names()))
        #print()

        # TFIDF
        #vectorizer_tfidf = TfidfVectorizer(norm = False, smooth_idf = False)
        vectorizer_tfidf = TfidfVectorizer(analyzer=analyzer)
        #sentence_vectors_tfidf = vectorizer_tfidf.fit_transform(new_df2['text'])
        #vectorizer_tfidf.fit(sentences_train)
        vectorizer_tfidf.fit(X)
        tfidf_X_train = vectorizer_tfidf.transform(sentences_train)
        tfidf_X_test = vectorizer_tfidf.transform(sentences_test)
        print("TFIDF Vectorizer (train): ")
        print(tfidf_X_train.toarray())
        print(str(tfidf_X_train.shape))
        print()
        print("TFIDF Vectorizer (test): ")
        print(tfidf_X_test.toarray())
        print(str(tfidf_X_test.shape))
        print()

        # TODO: Dimensionality Reduction?

        if model_type == "regular":
            # Machine learning
            models = []
            print("Logistic Regression: ")
            #logit_model = sm.Logit(y_train, count_X_train)
            #result = logit_model.fit()
            #print(result.summary2())
            #print()

            classifier_logistic = LogisticRegression()
            method_ML(classifier_logistic, "Logistic Regression", "CountVectorizer", count_X_train, 
                      count_X_test, y_train, y_test, "/Users/G/Loyola/Spring2020/DS796/count_log_roc_graph.png",
                        "/Users/G/Loyola/Spring2020/DS796/finalized_model_count_log.sav")
            method_ML(classifier_logistic, "Logistic Regression", "TfidfVectorizer", tfidf_X_train, 
                     tfidf_X_test, y_train, y_test, "/Users/G/Loyola/Spring2020/DS796/tfidf_log_roc_graph.png",
                        "/Users/G/Loyola/Spring2020/DS796/finalized_model_tfidf_log.sav")

            print("Naive Bayes: ")
            classifier_nb = MultinomialNB()
            method_ML(classifier_nb, "Naive Bayes", "CountVectorizer", count_X_train,
                       count_X_test, y_train, y_test, "/Users/G/Loyola/Spring2020/DS796/count_nb_roc_graph.png",
                        "/Users/G/Loyola/Spring2020/DS796/finalized_model_count_nb.sav")
            method_ML(classifier_nb, "Naive Bayes", "TfidfVectorizer", tfidf_X_train,
                        tfidf_X_test, y_train, y_test, "/Users/G/Loyola/Spring2020/DS796/tfidf_nb_roc_graph.png",
                        "/Users/G/Loyola/Spring2020/DS796/finalized_model_tfidf_nb.sav")
        
            print("Support Vector Machines: ")
            classifier_svm = SVC(kernel='linear', probability=True)
            method_ML(classifier_svm, "SVM", "CountVectorizer", count_X_train,
                        count_X_test, y_train, y_test, "/Users/G/Loyola/Spring2020/DS796/count_svm_roc_graph.png",
                     "/Users/G/Loyola/Spring2020/DS796/finalized_model_count_svm.sav")
            method_ML(classifier_svm, "SVM", "TfidfVectorizer", tfidf_X_train,
                        tfidf_X_test, y_train, y_test, "/Users/G/Loyola/Spring2020/DS796/tfidf_svm_roc_graph.png",
                     "/Users/G/Loyola/Spring2020/DS796/finalized_model_tfidf_svm.sav")

            print("Random Forest: ")
            classifier_rf = RandomForestClassifier()
            method_ML(classifier_rf, "Random Forest", "CountVectorizer", count_X_train,
                     count_X_test, y_train, y_test, "/Users/G/Loyola/Spring2020/DS796/count_rf_roc_graph.png",
                     "/Users/G/Loyola/Spring2020/DS796/finalized_model_count_rf.sav")
            method_ML(classifier_rf, "Random Forest", "TfidfVectorizer", tfidf_X_train,
                        tfidf_X_test, y_train, y_test, "/Users/G/Loyola/Spring2020/DS796/tfidf_rf_roc_graph.png",
                     "/Users/G/Loyola/Spring2020/DS796/finalized_model_tfidf_rf.sav")

            print("Comparing Models: ")
            models = []
            models.append(('LogisticRegression', classifier_logistic))
            models.append(('Naive Bayes', classifier_nb))
            models.append(('SVM', classifier_svm))
            models.append(('Random Forest', classifier_rf))
            print("Using Count:")
            compare_models(models, 7, count_X_train, y_train, 'count')
            print("Using TFIDF:")
            compare_models(models, 7, tfidf_X_train, y_train, 'tfidf')

        elif model_type == "active":
            print("Active Learning: ")
            accepted_vectorizers = ['COUNT', 'TFIDF']
            accepted_models = ['LR', 'NB', 'SVM', 'RF']
            accepted_query_strategies = ['CE', 'CM', 'CU', 'ES', 'MS', 'US']

            new_data_filename = input("Enter new data file: ")
            vectorizer = input("Enter the vectorizer to use (Default: COUNT): ").upper()
            model = input("Enter machine learning model to use (Default: LR): ").upper()
            query_strategy = input("Enter query_strategy to use (Default: US): ").upper()
            n_queries = input("Enter number of queries (Default: 10): ")
            print()

            if not vectorizer:
                vectorizer = 'COUNT'

            if not model:
                model = 'LR'

            if not query_strategy:
                query_strategy = 'US'

            if not n_queries:
                n_queries = int(10)

            if filename and new_data_filename and vectorizer.upper() in accepted_vectorizers and model.upper() in accepted_models and query_strategy.upper() in accepted_query_strategies:

                # Date,PMID,text,data
                df = pd.read_csv(new_data_filename, header=0)
                df['data_reuse'] = None
                print(df['text'].head(10))
                #print(df.tail(10))
                print()

                # Total number of records (with nulls)
                print("Count of records: \n" + str(df.text.count()))
                print()
                print("Count of unique records: \n" + str(df.groupby(['PMID']).count()))
                print()

                # Checking missing values
                print("Count of missing values: \n" + str(df.isnull().sum()))
                print()
                print("Rows where text is missing: ")
                print(df[df['text'].isnull()])
                print()

                if vectorizer == "COUNT":
                    # Using count vectorizer
                    print("##################################")
                    print("Using count vectorizer:")
                    print("##################################")
                    print()

                    # Tokenize, add vocabulary, and encode new data
                    count_X_new = vectorizer_count.fit_transform(df['text'])
                    # Save vectorizer
                    fd = open("/Users/G/Loyola/Spring2020/DS796/vectorizer_count.pkl", "wb")
                    pickle.dump(vectorizer_count, fd)
                    fd.close()
                    print("CountVectorizer saved: /Users/G/Loyola/Spring2020/DS796/vectorizer_count.pkl")
                    print()

                    # Re-encode training/test documents with the new vocabulary 
                    count_X_train = vectorizer_count.transform(sentences_train)
                    count_X_test = vectorizer_count.transform(sentences_test)

                    print("Count Vectorizer (New): ")
                    print(count_X_new.toarray())
                    print(str(count_X_new.shape))
                    print()

                    print("Vocabulary (New): ")
                    ordered_new_count = dict(sorted(vectorizer_count.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:1000])
                    print(str(ordered_new_count))
                    print()

                    active_learning("count", count_X_train, y_train, count_X_test, y_test, df, count_X_new, model, query_strategy, int(n_queries),
                                        "/Users/G/Loyola/Spring2020/DS796/active_model_count_" + model + ".sav",
                                        "/Users/G/git/repository/data-science-project/data/NIMH_pmids_matched_count_" + model + ".csv")
                    print()
                elif vectorizer == "TFIDF":
                    # Using tfidf vectorizer
                    print("##################################")
                    print("Using TFIDF vectorizer:")
                    print("##################################")
                    print()

                    # Tokenize, add vocabulary, and encode new data
                    tfidf_X_new = vectorizer_tfidf.fit_transform(df['text'])
                    # Save vectorizer
                    fd = open("/Users/G/Loyola/Spring2020/DS796/vectorizer_tfidf.pkl", "wb")
                    pickle.dump(vectorizer_tfidf, fd)
                    fd.close()
                    print("TFIDFVectorizer saved: /Users/G/Loyola/Spring2020/DS796/vectorizer_tfidf.pkl")
                    print()

                    # Re-encode training/test documents with the new vocabulary
                    tfidf_X_train = vectorizer_tfidf.transform(sentences_train)
                    tfidf_X_test = vectorizer_tfidf.transform(sentences_test)

                    print("Tfidf Vectorizer (New): ")
                    print(tfidf_X_new.toarray())
                    print(str(tfidf_X_new.shape))
                    print()

                    print("Vocabulary (New): ")
                    ordered_new_tfidf = dict(sorted(vectorizer_tfidf.vocabulary_.items(), key=lambda x: x[1], reverse=True)[:1000])
                    print(str(ordered_new_tfidf))
                    print()

                    active_learning("tfidf", tfidf_X_train, y_train, tfidf_X_test, y_test, df, tfidf_X_new, model, query_strategy, int(n_queries),
                                        "/Users/G/Loyola/Spring2020/DS796/active_model_tfidf_" + model + ".sav",
                                        "/Users/G/git/repository/data-science-project/data/NIMH_pmids_matched_tfidf_" + model + ".csv")
                    print()
                else:
                    print("Vectorizer not supported. Please see help for more info.")
                    print()

            else:
                print("Something went wrong with your input(s). Please see help for more info and check your inputs.")
                print()

    except:
        print("Encountered Error: ", sys.exc_info())
        raise

def active_learning(vectorizer_method, X_train, y_train, X_test, y_test, orig_df, X_new, model, qstrategy, n_queries, model_filename, df_filename):

    classifier = None
    strategy = None

    if model == 'LR':
        classifier = LogisticRegression()
    elif model == 'NB':
        classifier = MultinomialNB()
    elif model == 'SVM':
        classifier = SVC(kernel='linear', probability=True)
    elif model == 'RF':
        classifier = RandomForestClassifier()

    if qstrategy == 'CE':
        strategy = classifier_entropy
    elif qstrategy == 'CM':
        strategy = classifier_margin
    elif qstrategy == 'CU':
        strategy = classifier_uncertainty
    elif qstrategy == 'ES':
        strategy = entropy_sampling
    elif qstrategy == 'MS':
        strategy = margin_sampling
    elif qstrategy == 'US':
        strategy = uncertainty_sampling

    learner = ActiveLearner(
                 estimator=classifier,
                 query_strategy=strategy,
                 X_training=X_train, y_training=y_train
             )

    accuracy_scores = [learner.score(X_test, y_test)]
    recall_scores = [recall_score(y_test, learner.predict(X_test))]

    for i in range(n_queries):
        #print(X_train.shape)
        #print(X_new.shape)
        #print(orig_df.iloc[0])
        query_idx, query_inst = learner.query(X_new)
        #print(query_inst)
        #print(query_idx)
        print(orig_df['text'].iloc[query_idx[0]])
        print("Is this a data reuse statement or not (1=yes, 0=no)?")
        try:
            y_new = np.array([int(input())], dtype=int)
            if y_new in [0,1]:
                orig_df.loc[query_idx[0], 'data_reuse'] = y_new
                learner.teach(query_inst.reshape(1, -1), y_new)

                X_new = csr_matrix(np.delete(X_new.toarray(), query_idx, axis=0))

                accuracy_scores.append(learner.score(X_test, y_test))
                recall_scores.append(recall_score(y_test, learner.predict(X_test)))
                #print(accuracy_scores)
                #print(recall_scores)
                print()
            else:
                print("Input not accepted. Type '1' for yes or '0' for no. Skipping.")
                print()
        except:
            print("Encountered Error: " + str(sys.exc_info()))
            print()
            return

    # Performance of classier
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(10, 5))
        plt.title('Performance of the classifier during the active learning')
        #plt.plot(range(n_queries+1), accuracy_scores)
        #plt.scatter(range(n_queries+1), accuracy_scores)
        plt.plot(range(n_queries+1), recall_scores)
        plt.scatter(range(n_queries+1), recall_scores)
        plt.xlabel('Number of queries')
        plt.ylabel('Performance')
        plt.savefig('/Users/G/Loyola/Spring2020/DS796/active_model_' + vectorizer_method + '_' + model + '_performance.png')
        print("Graph saved: /Users/G/Loyola/Spring2020/DS796/active_model_" + vectorizer_method + '_' + model + "_performance.png")
        print()
        #plt.show()
        plt.close()

    fd = open(model_filename, 'wb')
    pickle.dump(learner, fd)
    fd.close()
    print("Model saved: ", model_filename)
    print()

    orig_df.to_csv(df_filename, index=False)
    print("Dataframe saved: ", df_filename)
    print()

def generateWordCloud(filename):
    df = pd.read_csv(filename, header=0)
    mask = np.array(Image.open('/Users/G/Loyola/Spring2020/DS796/Brain.jpg'))
    wordcloud = WordCloud(
            width = 3000,
            height = 2000,
            background_color = 'black',
            stopwords = STOPWORDS,
            mask = mask).generate(str(df['data']))
    fig = plt.figure(
            figsize = (40, 30),
            facecolor = 'k',
            edgecolor = 'k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('/Users/G/Loyola/Spring2020/DS796/wordcloud.png')
    print("Wordcloud saved: /Users/G/Loyola/Spring2020/DS796/wordcloud.png")
    print()
    #plt.show()
    plt.close(fig)

'''
def loadModel(filename, vectorizer_method):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    yPredict = loaded_model.predict(Xtest)
    print(yPredict)
    print()
    '''

def options(argument):

    switcher = {
        '1':  'Filter',
        '2':  'Read',
        '3':  'Write',
        '4':  'Model',
        '5':  'WordCloud',
        'q':  'Quit',
        'h':  'Help'
    }

    return switcher.get(argument, "Invalid argument")

def help():

    print("""
          Options available:
          1:  Filter
          2:  Read
          3:  Write
          4:  Model
              regular (using conventional Machine Learning techniques)
              active  (Accepted vectorizers (case insensitive): COUNT (Default), TFIDF,
                       Accepted models (case insensitive): LR, NB, SVM, RF,
                       Accepted query strategy (case insensitive): CE, CM, CU, ES, MS, US)
              LR - Logistic Regression (Default)
              NB - Naives Bayes
              SVM - Support Vector Machine
              RF - Random Forest
              CE - Classifier Entropy
              CM - Classifier Margin
              CU - Classifier Uncertainty
              ES - Entropy Sampling
              MS - Margin Sampling
              US - Uncertainty Sampling (Default)
              Default # of queries: 10
          5:  WordCloud
          q:  Quit
          h:  Help
          """)

if __name__ == "__main__":

    arg = input("Enter option or h for help: ")
    while(arg != 'q'):
        if options(arg) == 'Help':
            help()
        elif options(arg) == 'Filter':
            start = time.time()
            filter_text()
            print('It took', round((time.time()-start)/60, 2), 'minutes.')
            print()
        elif options(arg) == 'Read':
            filename = input("Enter pickled file: ")
            if filename:
                read_df(filename.strip())
            else:
                read_df()
        elif options(arg) == 'Write':
            output = input("Enter output file: ")
            if output:
                print("Writing dataframe to csv.")
                write_df_to_csv(output)
                print("Done.")
                print()
            else:
                print("Please provide an output file.")
                print()
        elif options(arg) == 'Model':
            build_model = input("Build (regular or active): ")
            filename = input("Enter input file to build model: ")

            try:
                if filename and build_model:
                    model(filename.strip(), build_model.lower())
                else:
                    print("Please enter a filename/model type to build model.")
                    print()
            except:
                pass
        elif options(arg) == 'WordCloud':
            filename = input("Enter input file to build wordcloud: ")
            try:
                if filename:
                    generateWordCloud(filename.strip())
                else:
                    print("Please enter a valid filename to begin.")
                    print()
            except:
                pass

        arg = input("Enter option or h for help: ")
