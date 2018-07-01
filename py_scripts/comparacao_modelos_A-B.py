
# coding: utf-8

# In[ ]:

from skimage.measure import compare_ssim as s
from skimage.measure import compare_mse as eqm
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os.path


# In[2]:

DATA_DIR='/home/daniel/AnacondaProjects/prednet-master_2/kitti_results/'
resp = raw_input('deseja mudar a pasta de salvamento ?'+' Padrão: '+ DATA_DIR+': ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    save_dir_name = raw_input('Pasta onde deseja salvar')
    SAVE_DIR = os.path.join(DATA_DIR,str(save_dir_name))
else:
    SAVE_DIR = DATA_DIR
print'os arquivos serão salvos em:'+ SAVE_DIR
resp = raw_input('deseja mudar o nome do arquivo ? Padrão, "twwat"')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    arquivo_name = raw_input('Nome com o qual deseja salvar os resultados ?')
    name = str(arquivo_name)
else:
    name = 'tw'
print'os arquivos terão em parte do nome:'+ name


# In[ ]:

obs = glob(SAVE_DIR+'obs_*_1*') 
pred = glob(SAVE_DIR+'pred_*_1*')
obs = sorted(obs)
pred= sorted(pred)
ssim=[]
mse=[]
obs_pred = zip(obs,pred)
for each_obs, each_pred in obs_pred:
    name_obs = os.path.basename(each_obs)
    name_pred = os.path.basename(each_pred)
    name_file =  name_obs+'_'+name_pred
    if name_file == 'obs_0_1.png_pred_0_1.png':
        print 'no'
        continue
    elif name_file == 'obs_1_1.png_pred_1_1.png':
        print 'no'
        continue
    elif name_file == 'obs_2_1.png_pred_2_1.png':
        print 'no'
        continue
    elif name_file == 'obs_3_1.png_pred_3_1.png':
        print 'no'
        continue
    else:
        obs_processing = io.imread(each_obs)
        pred_processing = io.imread(each_pred)
        m = eqm(obs_processing,pred_processing)
        m /= float (obs_processing.shape[0]*obs_processing.shape[1])
        mse.append(m)
        s1 = s(obs_processing,pred_processing,multichannel=True )
        ssim.append(s1)
        print'comparison done:',name_file ,m ,s1
print 'Pronto =)'
count = 15
mse_tempo = np.zeros((20,2))
i=0
tempo=15 
for each_mse in mse:
    if tempo > 75:
        tempo = 15
    mse_tempo[i][0]=each_mse
    mse_tempo[i][1]=tempo
    i+=1
    tempo+=15
np.savetxt(SAVE_DIR+'mse_'+name,mse_tempo, delimiter=',')
count = 15
ssim_tempo = np.zeros((20,2))
i=0
tempo=15 
for each_ssim in ssim:
    if tempo > 75:
        tempo = 15
    ssim_tempo[i][0]=each_ssim
    ssim_tempo[i][1]=tempo
    i+=1
    tempo+=15
np.savetxt(SAVE_DIR+'ssim_'+name,ssim_tempo, delimiter=',')
print 'arquivos prontos e salvos em: ',SAVE_DIR   


# In[ ]:



