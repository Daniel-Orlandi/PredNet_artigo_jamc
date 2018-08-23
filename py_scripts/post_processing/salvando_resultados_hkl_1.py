# coding: utf-8
import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
import os
from prednet import PredNet
from prednet.data_utils import SequenceGenerator
from prednet.kitti_settings import *

n_plot = 40
batch_size = 15
nt = 15

resp = raw_input('deseja mudar os nomes dos arquivos a serem carregados? Padrao (prednet_jamc_extrapfinetuned_a3_t5_fm) ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    peso_hdf = raw_input('weights_file:') 
    peso_json = raw_input('json weight file:')
    weights_file = os.path.join(WEIGHTS_DIR, str(peso_hdf))
    json_file = os.path.join(WEIGHTS_DIR,str(peso_json))
    test_file = os.path.join(DATA_DIR, 'X_val.hkl')
    test_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
else:
    weights_file = os.path.join(WEIGHTS_DIR, 'prednet_jamc_weights-extrapfinetuned_a3_t5_fm.hdf5')
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_jamc_model-extrapfinetuned_a3_t5_fm.json')
    test_file = os.path.join(DATA_DIR, 'X_val.hkl')
    test_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
print 'nome dos arquivos:'+ weights_file+'\n'+json_file



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



plot_save_dir = '/home/daniel/PredNet_artigo_jamc/kitti_results/hdf-result'
import hickle as hkl
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)

resp = raw_input('deseja mudar os nomes dos arquivos antes de salvar? ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    nome_xhat_saver = raw_input('digite nome X_hat: ')
    x_hat_saver = os.path.join (plot_save_dir,str(nome_xhat_saver))
else:
    x_hat_saver = os.path.join (plot_save_dir,'X_hat_jamc_t+5_a3_2.hkl')
print 'o arquivo será salvo com o nome:', x_hat_saver
hkl.dump(X_hat,x_hat_saver)
print 'Pronto!'

resp = raw_input('deseja mudar os nomes dos arquivos antes de salvar? ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    nome_xtest_saver = raw_input('digite nome X_test: ')
    x_test_saver = os.path.join (plot_save_dir,str(nome_xtest_saver))
else:
    x_test_saver = os.path.join (plot_save_dir,'X_test_jamc_t+5_a3_2.hkl')
print 'o arquivo será salvo com o nome:', x_test_saver
hkl.dump(X_test,x_test_saver)
print 'Pronto!'






