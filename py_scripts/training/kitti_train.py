'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

resp = raw_input('deseja mudar o nome com que os arquivos serao salvos? (padrao = prednet_jamc_a3_t1.hdf5): ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    peso_hdf = raw_input('weights_file:') 
    peso_json = raw_input('json weight file:')
    weights_file = os.path.join(DATA_DIR, str(peso_hdf))
    json_file = os.path.join(DATA_DIR,str(peso_json))
else:
    weights_file = os.path.join(DATA_DIR, 'prednet_jamc_a3_t1.hdf5')
    json_file = os.path.join(DATA_DIR, 'prednet_jamc_a3_t1.json')
    
print 'nome dos arquivos:'+ weights_file+'\n'+json_file
save_model = True  # if weights will be saved

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_test.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

# Training parameters
resp = raw_input('Deseja setar os parametros de treinamento? [Y,y,Yes,yes,S,s,Sim,sim]. Ou deseja escolher conf_monografia? ')
if  resp == 'y' or resp == 'sim' or resp == 's' or resp == 'yes' or resp == 'Yes' or resp == 'Sim':
    nb_epoch = raw_input('nb_epoch (padrao = 150): ')
    batch_size = raw_input('batch_size (padrao = 4): ')
    samples_per_epoch = raw_input('samples_per_epoch (padrao = 500): ')
    N_seq_val = raw_input('N_seq_val (padrao = 100): ' )
elif resp == 'conf_monografia':
    nb_epoch = 150
    batch_size = 4
    samples_per_epoch = 1000 #Numero de imagens a cada n batch_size (samples_per_epoch/batch_size = iterations)
    N_seq_val = 1000
    print ('nb_epoch = ' + str(nb_epoch) +';'+' batch_size = ' + str(batch_size)+';'+ ' samples_per_epoch = '+';'+ str(samples_per_epoch)+';'+ ' N_seq_val = ' + str(N_seq_val))
else:
    nb_epoch = 150
    batch_size = 4
    samples_per_epoch = 500 #Numero de imagens a cada n batch_size (samples_per_epoch/batch_size = iterations)
    N_seq_val = 100  # number of sequences to use for validation
#resp 2 = raw_input('Deseja comecar o treinamento com os parametros: '+'nb_epoch = ' + str(nb_epoch) +';'+' batch_size = ' + str(batch_size)+';'+ ' samples_per_epoch = '+';'+ str(samples_per_epoch)+';'+ ' N_seq_val = ' + str(N_seq_val)+'')
# Model parameters
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0, 0, 0])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 10  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
#Create Model
model = Model(inputs=inputs, outputs=final_errors)
# Loading Pretrained weights
#model.load_weights('/home/daniel/AnacondaProjects/prednet-master_2/model_data_keras2/prednet_kitti_weights-extrapfinetuned.hdf5')
# configuring Model
model.compile(loss='mean_absolute_error', optimizer='adam')

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size,N_seq=N_seq_val)

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
