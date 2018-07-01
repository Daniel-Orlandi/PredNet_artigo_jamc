import os
import requests
import numpy as np
from scipy.misc import imread, imresize, imshow
import hickle as hkl
from kitti_settings import *
from glob import glob
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec
from PIL import Image
print 'inicio'
pred = hkl.load('/home/daniel/AnacondaProjects/prednet-master_2/kitti_results/hdf-result/X_hat_model_nt15_f5_a3_pc.hkl')
obs = hkl.load('/home/daniel/AnacondaProjects/prednet-master_2/kitti_results/hdf-result/X_test_model_nt15_f5_a3_pc.hkl')
print pred.shape
print obs.shape
DATA_DIR = '/home/daniel/AnacondaProjects/prednet-master_2/kitti_results/prediction_plots_mono_tw/'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
nt=15
n_plot=15
plot_idx = np.random.permutation(pred.shape[0])[:n_plot]
print 'Iniciando loops'
for i in range(1):
    for t in range (nt): 
        img = pred[i,t]    
        plot.imshow(img,aspect='auto')
        xlim, ylim = plot.xlim(), plot.ylim()
        plot.xlim(xlim)
        plot.ylim(ylim)   
        plot.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        plot.savefig(DATA_DIR+'/'+'pred_'+str(i)+'_'+str(t)+'.png',dpi=600,bbox_inches='tight')

for i in range(1):
    for t in range (nt): 
        img = obs[i,t]    
        plot.imshow(img,aspect='auto')
        xlim, ylim = plot.xlim(), plot.ylim()
        plot.xlim(xlim)
        plot.ylim(ylim)   
        plot.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        plot.savefig(DATA_DIR+'/'+'obs_'+str(i)+'_'+str(t)+'.png',dpi=600,bbox_inches='tight')
    
print 'Imagens prontas'
print 'calculando mse '
mse_model = np.mean( (obs[:, 1:] - pred[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (obs[:, :-1] - obs[:, 1:])**2 )
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()

