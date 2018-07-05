import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hickle as hkl
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
print 'Iniciando'
plot_save_dir = '/home/daniel/AnacondaProjects/prednet-master_2/kitti_results/hdf-result/'
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
n_plot = 40
batch_size = 15
nt = 15
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights_nt15_f5_b3_a3_pc.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model_nt15_f5_b3_a3_pc.json')
test_file = os.path.join(DATA_DIR, 'X_val.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)
# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)
test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
hkl.dump(X_hat,plot_save_dir+'X_hat_test_model_nt15_t5_a3.hkl')
print 'dump 1 pronto'
hkl.dump(X_test,plot_save_dir+'X_test_model_nt15_t5_a3.hkl')
print 'dump 2 pronto'
