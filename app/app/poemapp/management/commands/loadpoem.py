from __future__ import division

import numpy as np
import pandas as pd

import operator

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import cross_val_score, KFold, train_test_split

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators

from sklearn.externals import joblib

from django.core.management.base import BaseCommand, CommandError
from poemapp.models import Poem




class Command(BaseCommand):
    help = 'Load poem'

    # def add_arguments(self, parser):
    #     parser.add_argument('poll_id', nargs='+', type=int)

    def handle(self, *args, **options):
        mvc = MajorityVoteClassifier()
        mvc.read_files()
        # self.stdout.write(mvc.read_files())
        self.stdout.write(self.style.SUCCESS('Successfully closed poll'))




def count_line(contents):
    if not contents: 
        return 0
    lines = contents.split('\n')
    return len(lines)

def mean_word(contents):
    if not contents: 
        return 0
    lines = contents.split('\n')
    words = [len(line.split(' ')) for line in lines]
    return np.mean(words)

def total_word(contents):
    if not contents:
        return 0
    lines = contents.split('\n')
    words = [len(line.split(' ')) for line in lines]
    return sum(words)

def std_w(data_w):
    if not data_w:
        return 0
    ws = map(int, data_w.split(','))
    return np.std(ws)

def mean_y(data_y):
    if not data_y:
        return 0
    ys = map(int, data_y.split(','))
    return np.mean([abs(ys[1:][i] - ys[:-1][i]) for i in range(0, len(ys[1:]))])

def mean_h(data_h):
    if not data_h:
        return 0
    hs = map(int, data_h.split(','))
    return np.mean(hs)

def mean_x(data_x):
    if not data_x:
        return 0
    xs = map(int, data_x.split(','))
    x_min = np.min(xs)
    return np.mean(map(lambda x:x-x_min, xs))

def get_page(page):
    return int(page.replace('Page', '').strip())

def convert_feature(df, target=1):
    df['count_line'] = df['content'].apply(count_line)
    df['mean_word'] = df['content'].apply(mean_word)
    df['total_word'] = df['content'].apply(total_word)
    df['std_w'] = df['data_w'].apply(std_w)
    df['mean_y'] = df['data_y'].apply(mean_y)
    df['mean_h'] = df['data_h'].apply(mean_h)
    df['mean_x'] = df['data_x'].apply(mean_x)
    df['page_num'] = df['page'].apply(get_page)
    df['target'] = target
    return df

class MajorityVoteClassifier(object):
    def __init__(self):
        self.classifiers = []
        base = '/mnt/dev/workspace/poem/data/'
        self.poem_filename = base + 'poem-austlit-20160412.csv'
        self.other_filename = base + 'others-201604120925.csv'

    def read_files(self):
        poems = pd.read_csv(self.poem_filename)
        convert_feature(poems, target=1)
        
        others = pd.read_csv(self.other_filename)
        convert_feature(others, target=0)
        
        self.all_df = pd.concat([poems, others])

        
    def process_data(self):
        format_features = ['count_line', 'mean_word', 'total_word', 'std_w', 'mean_y', 'mean_h', 'mean_x', 'page_num']
        content_features = 'content'
        y_feature = ['target']
        
        X_format = self.all_df[format_features].values
        X_format[X_format > 1000] = 1000
        X_format[np.isnan(X_format)] = 0
        
        X_content = self.all_df[content_features].values
        X_content = CountVectorizer(max_df=10, min_df=1).fit_transform(X_content)
        X_content = TfidfTransformer().fit_transform(X_content)
        
        y = self.all_df[y_feature].values
        y = y.flatten()
        
        self.X_format_train, self.X_format_test, self.y_format_train, self.y_format_test = train_test_split(X_format, y, test_size=0.20, random_state=42)
        self.X_content_train, self.X_content_test, self.y_content_train, self.y_content_test = train_test_split(X_content, y, test_size=0.20, random_state=42)
        
    def predict(self):
        count = len(self.y_content_test)
        predictions = [self._predict(self.X_format_test[i, :], self.X_content_test[i, :]) for i in range(0, count)]
        predictions = np.array(predictions)
        
        print(sum(predictions==self.y_content_test)/count)
        return sum(predictions==self.y_content_test)/count
    
    def _predict(self, X_format, X_content):
        return self.predict_proba(X_format, X_content)
    
    def predict_proba(self, X_format, X_content):
        probas = []
        predicts = []
        for name, clf in self.classifiers:
            if name == 'RandomForest':
                X = X_format
            else:
                X = X_content
            probas.append(clf.predict_proba(X))
            predicts.append(clf.predict(X))
            
        probas = np.asarray(probas)
        n = np.argmax(probas)
        if n in [0, 1]:
            return predicts[0][0]
        elif n in [2, 3]:
            return predicts[1][0]
        else:
            return predicts[2][0]
    
    def run(self):
        print('read **************************************')
        self.read_files()
        print('process_data ******************************')
        self.process_data()
        print('train_classifier **************************')
        self.triain_rfclassifier()
        self.triain_nbclassifier()
        self.triain_lrclassifier()
        print('predict ***********************************')
        print self.predict()
        
    def triain_rfclassifier(self):
        X, y = self.X_format_train, self.y_format_train
        clf = Pipeline([
                    ('clf', RandomForestClassifier()),
                ])

        parameters = {
                        'clf__n_estimators': (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70), 
                        'clf__criterion': ('gini', 'entropy'),
                        'clf__max_features': ('auto', 'sqrt', 'log2')
                     }    

        gs_clf = GridSearchCV(clf, parameters, cv=5)
        gs_clf.fit(X, y)

        best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        print('RandomForestClassifier')
        print('Score : ', score)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
        
        self.classifiers.append(('RandomForest', gs_clf.best_estimator_))
    
    def triain_nbclassifier(self):
        X, y = self.X_content_train, self.y_content_train
        
        clf = Pipeline([
                        ('clf', MultinomialNB()),
                    ])

        parameters = { 'clf__alpha': (0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4), 
                        'clf__fit_prior':(True, False)}    

        gs_clf = GridSearchCV(clf, parameters, cv=5)
        gs_clf.fit(X, y)

        best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        print('MultinomialNB')
        print('Score : ', score)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
        
        self.classifiers.append(('MultinomialNB', gs_clf.best_estimator_))
    
    def triain_lrclassifier(self):
        X, y = self.X_content_train, self.y_content_train
        
        clf = Pipeline([
                        ('clf', LogisticRegression()),
                    ])

        parameters = { 
#                         'clf__alpha': (0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4), 
#                         'clf__fit_prior':(True, False)
                     }    

        gs_clf = GridSearchCV(clf, parameters, cv=5)
        gs_clf.fit(X, y)

        best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        print('LogisticRegressionClassifier')
        print('Score : ', score)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
        
        self.classifiers.append(('LogisticRegression', gs_clf.best_estimator_))
    
    def save(self):
        joblib.dump(clf, 'filename.pkl') 
        joblib.dump(clf, 'filename.pkl')