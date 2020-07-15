from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
import h5py

from numpy import loadtxt
from keras.models import load_model
import tensorflow as tf
import keras

# Flask Application for REST service
from flask import Flask, request, Response, jsonify, redirect, render_template, url_for
import json

app = Flask(__name__)

FIELDS = {} # mandatory fields

attackmap = {'normal':0,'dos':1,'probe':2,'r2l':3,'u2r':4}
attackmap_rev = {v: k for k, v in attackmap.items()}

# load numeric_variables_zscore_attributes for preprocessing.
with open('numeric_variables_zscore_attributes.json','r') as f:
    num_var_zs_attr = json.load(f)

# load model
config = tf.ConfigProto(
    device_count={'CPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
keras.backend.set_session(session)
model = load_model('dnn1layer/dnn1layer_model.hdf5')

def create_validation_logic(body):
    if not body:
        return None

    return body

def preprocess(body):
    
    # for now body : let be a sample data for test.
    #dict_ = body
    
    # load and evaluate a saved model
    with open("pred_input.json",'r') as f:
        dict_ = json.load(f)

    data = pd.DataFrame([dict_])

    # Numeric Feature scale with Z-score
    def encode_numeric_zscore(data, name, mean=None, sd=None):
        if mean is None:
            mean = data[name].mean()

        if sd is None:
            sd = data[name].std()

        data[name] = (data[name] - mean) / sd

    encode_numeric_zscore(data, 'duration',num_var_zs_attr['duration']['mean'],num_var_zs_attr['duration']['sd'])
    encode_numeric_zscore(data, 'src_bytes',num_var_zs_attr['src_bytes']['mean'],num_var_zs_attr['src_bytes']['sd'])
    encode_numeric_zscore(data, 'dst_bytes',num_var_zs_attr['dst_bytes']['mean'],num_var_zs_attr['dst_bytes']['sd'])
    encode_numeric_zscore(data, 'wrong_fragment',num_var_zs_attr['wrong_fragment']['mean'],num_var_zs_attr['wrong_fragment']['sd'])
    encode_numeric_zscore(data, 'urgent',num_var_zs_attr['urgent']['mean'],num_var_zs_attr['urgent']['sd'])
    encode_numeric_zscore(data, 'hot',num_var_zs_attr['hot']['mean'],num_var_zs_attr['hot']['sd'])
    encode_numeric_zscore(data, 'num_failed_logins',num_var_zs_attr['num_failed_logins']['mean'],num_var_zs_attr['num_failed_logins']['sd'])
    encode_numeric_zscore(data, 'num_compromised',num_var_zs_attr['num_compromised']['mean'],num_var_zs_attr['num_compromised']['sd'])
    encode_numeric_zscore(data, 'root_shell',num_var_zs_attr['root_shell']['mean'],num_var_zs_attr['root_shell']['sd'])
    encode_numeric_zscore(data, 'su_attempted',num_var_zs_attr['su_attempted']['mean'],num_var_zs_attr['su_attempted']['sd'])
    encode_numeric_zscore(data, 'num_root',num_var_zs_attr['num_root']['mean'],num_var_zs_attr['num_root']['sd'])
    encode_numeric_zscore(data, 'num_file_creations',num_var_zs_attr['num_file_creations']['mean'],num_var_zs_attr['num_file_creations']['sd'])
    encode_numeric_zscore(data, 'num_shells',num_var_zs_attr['num_shells']['mean'],num_var_zs_attr['num_shells']['sd'])
    encode_numeric_zscore(data, 'num_access_files',num_var_zs_attr['num_access_files']['mean'],num_var_zs_attr['num_access_files']['sd'])
    encode_numeric_zscore(data, 'num_outbound_cmds',num_var_zs_attr['num_outbound_cmds']['mean'],num_var_zs_attr['num_outbound_cmds']['sd'])
    encode_numeric_zscore(data, 'count',num_var_zs_attr['count']['mean'],num_var_zs_attr['count']['sd'])
    encode_numeric_zscore(data, 'srv_count',num_var_zs_attr['srv_count']['mean'],num_var_zs_attr['srv_count']['sd'])
    encode_numeric_zscore(data, 'serror_rate',num_var_zs_attr['serror_rate']['mean'],num_var_zs_attr['serror_rate']['sd'])
    encode_numeric_zscore(data, 'srv_serror_rate',num_var_zs_attr['srv_serror_rate']['mean'],num_var_zs_attr['srv_serror_rate']['sd'])
    encode_numeric_zscore(data, 'rerror_rate',num_var_zs_attr['rerror_rate']['mean'],num_var_zs_attr['rerror_rate']['sd'])
    encode_numeric_zscore(data, 'srv_rerror_rate',num_var_zs_attr['srv_rerror_rate']['mean'],num_var_zs_attr['srv_rerror_rate']['sd'])
    encode_numeric_zscore(data, 'same_srv_rate',num_var_zs_attr['same_srv_rate']['mean'],num_var_zs_attr['same_srv_rate']['sd'])
    encode_numeric_zscore(data, 'diff_srv_rate',num_var_zs_attr['diff_srv_rate']['mean'],num_var_zs_attr['diff_srv_rate']['sd'])
    encode_numeric_zscore(data, 'srv_diff_host_rate',num_var_zs_attr['srv_diff_host_rate']['mean'],num_var_zs_attr['srv_diff_host_rate']['sd'])
    encode_numeric_zscore(data, 'dst_host_count',num_var_zs_attr['dst_host_count']['mean'],num_var_zs_attr['dst_host_count']['sd'])
    encode_numeric_zscore(data, 'dst_host_srv_count',num_var_zs_attr['dst_host_srv_count']['mean'],num_var_zs_attr['dst_host_srv_count']['sd'])
    encode_numeric_zscore(data, 'dst_host_same_srv_rate',num_var_zs_attr['dst_host_same_srv_rate']['mean'],num_var_zs_attr['dst_host_same_srv_rate']['sd'])
    encode_numeric_zscore(data, 'dst_host_diff_srv_rate',num_var_zs_attr['dst_host_diff_srv_rate']['mean'],num_var_zs_attr['dst_host_diff_srv_rate']['sd'])
    encode_numeric_zscore(data, 'dst_host_same_src_port_rate',num_var_zs_attr['dst_host_same_src_port_rate']['mean'],num_var_zs_attr['dst_host_same_src_port_rate']['sd'])
    encode_numeric_zscore(data, 'dst_host_srv_diff_host_rate',num_var_zs_attr['dst_host_srv_diff_host_rate']['mean'],num_var_zs_attr['dst_host_srv_diff_host_rate']['sd'])
    encode_numeric_zscore(data, 'dst_host_serror_rate',num_var_zs_attr['dst_host_serror_rate']['mean'],num_var_zs_attr['dst_host_serror_rate']['sd'])
    encode_numeric_zscore(data, 'dst_host_srv_serror_rate',num_var_zs_attr['dst_host_srv_serror_rate']['mean'],num_var_zs_attr['dst_host_srv_serror_rate']['sd'])
    encode_numeric_zscore(data, 'dst_host_rerror_rate',num_var_zs_attr['dst_host_rerror_rate']['mean'],num_var_zs_attr['dst_host_rerror_rate']['sd'])
    encode_numeric_zscore(data, 'dst_host_srv_rerror_rate',num_var_zs_attr['dst_host_srv_rerror_rate']['mean'],num_var_zs_attr['dst_host_srv_rerror_rate']['sd'])

    #protocol_type feature mapping
    pmap = {'icmp':0,'tcp':1,'udp':2}
    data['protocol_type'] = data['protocol_type'].map(pmap)

    #flag feature mapping
    fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
    data['flag'] = data['flag'].map(fmap)

    #protocol_type feature mapping
    attackmap = {'normal':0,'dos':1,'probe':2,'r2l':3,'u2r':4}
    data['Attack Type'] = data['Attack Type'].map(attackmap)
    data['Attack Type'].value_counts()

    reqd_cols_for_pred = ['duration','protocol_type','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_file_creations','num_shells','num_access_files','is_host_login','is_guest_login','count','srv_count','serror_rate','rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate']

    data = data[reqd_cols_for_pred]
    print(data.shape)

    testT = np.array(data)

    testT.astype(float)

    scaler = Normalizer().fit(testT)
    testT = scaler.transform(testT)

    X_test = np.array(testT)
    
    return X_test


@app.route('/')
def home():

    return render_template('home.html')

@app.route('/get_intrusion',methods = ['POST'])
def get_intrusion():
    with session.as_default():
        with session.graph.as_default():
            body = request.get_json()
            #validation_out = create_validation_logic(body)
            validation_out = True # for test
            if validation_out:
                # preprocess data
                X_test = preprocess(body)

                # make a prediction
                result = 'No prediction happen.'
                result = model.predict(X_test)

                return render_template('home.html', prediction_text="Intrusion Type : {}".format(attackmap_rev[result.argmax()]))
                #return jsonify({"Intrusion Type":attackmap_rev[result.argmax()]}), 201

    #return '{"error": "Bad request"}', 400
    return render_template('home.html', prediction_text="Bad Request.")

@app.route('/get_intrusion_api', methods=['POST'])
def get_intrusion_api():
    with session.as_default():
        with session.graph.as_default():
            body = request.get_json()
            #validation_out = create_validation_logic(body)
            validation_out = True # for test
            if validation_out:
                # preprocess data
                X_test = preprocess(body)

                # make a prediction
                result = 'No prediction happen.'
                result = model.predict(X_test)

                return jsonify({"Intrusion Type":attackmap_rev[result.argmax()]}), 201

    return '{"error": "Bad request"}', 400


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=8080)
