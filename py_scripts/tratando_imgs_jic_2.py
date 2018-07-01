
# coding: utf-8

# In[4]:

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



# In[5]:

LOAD_DIR = '/home/daniel/PredNet_artigo_jamc/kitti_results/hdf-result/'
DATA_DIR = '/home/daniel/PredNet_artigo_jamc/kitti_results/'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
resp = raw_input('deseja mudar os nomes dos arquivos de entrada? Nome padrao,X_hat_test_model_t5_ptw.hkl e X_test_model_t5_ptw.hkl: ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    pred_hkl = raw_input('Previsto-X_hat:') 
    obs_hkl = raw_input('Observado-X_test:')
    pred_file = os.path.join(LOAD_DIR, str(pred_hkl))
    obs_file = os.path.join(LOAD_DIR,str(obs_hkl))
else:
    pred_file = os.path.join(LOAD_DIR,'X_hat_model_nt15_t5_a3_finetunning_pc.hkl')
    obs_file = os.path.join(LOAD_DIR,'X_test_model_nt15_t5_a3_finetunning_pc.hkl')
print 'nome dos arquivos:'+ pred_file+'\n'+obs_file
pred = hkl.load(pred_file)
obs = hkl.load(obs_file)
print pred.shape
print obs.shape
resp = raw_input('deseja mudar a pasta de salvamento das imagens? Pasta padrao, prediction_plots_artigo_jamc_finetunned: ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    save_dir_name = raw_input('Pasta onde deseja salvar ')
    SAVE_DIR = os.path.join(DATA_DIR,str(save_dir_name))
else:
    SAVE_DIR = DATA_DIR+'prediction_plots_artigo_jamc_finetunned'
print'os arquivos serão salvos em:'+ SAVE_DIR

resp = raw_input('deseja mudar a extensão dos arquivos de imagem? Extensão padrao,.png: ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    extensao = str(raw_input('Digite a extensão desejada:'))
else:
    extensao = '.png'
# In[3]:

print pred[:, 1:].shape
nt=15
n_plot=15
plot_idx = np.random.permutation(pred.shape[0])[:n_plot]
print 


# In[ ]:
print 'Começando o loop'
for i in range(4):
    for t in range (nt): 
        img = pred[i,t]    
        plot.imshow(img,aspect='auto')
        xlim, ylim = plot.xlim(), plot.ylim()
        plot.xlim(xlim)
        plot.ylim(ylim)   
        plot.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        plot.savefig(SAVE_DIR+'/'+'pred_'+str(i)+'_'+str(t)+extensao,dpi=300,bbox_inches='tight')

for i in range(4):
    for t in range (nt): 
        img = obs[i,t]    
        plot.imshow(img,aspect='auto')
        xlim, ylim = plot.xlim(), plot.ylim()
        plot.xlim(xlim)
        plot.ylim(ylim)   
        plot.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        plot.savefig(SAVE_DIR+'/'+'obs_'+str(i)+'_'+str(t)+extensao,dpi=300,bbox_inches='tight')
    
print 'Pronto'


# In[ ]:



