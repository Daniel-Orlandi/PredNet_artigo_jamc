
# coding: utf-8



from skimage.measure import compare_ssim as s
from skimage.measure import compare_mse as eqm
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os.path




DATA_DIR='/home/daniel/PredNet_artigo_jamc/kitti_results/prediction_plots_artigo_jamc_finetunned_3/'
"""
resp = raw_input('deseja mudar a pasta de salvamento ?'+' Padrão: '+ DATA_DIR+': ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    save_dir_name = raw_input('Pasta onde deseja salvar')
    SAVE_DIR = os.path.join(DATA_DIR,str(save_dir_name))

else:
"""
SAVE_DIR = DATA_DIR

print'os arquivos serão salvos em:'+ SAVE_DIR

"""
resp = raw_input('deseja mudar o nome do arquivo ? Padrão, "tw": ')

if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    arquivo_name = raw_input('Nome com o qual deseja salvar os resultados ? ')
    name = str(arquivo_name)

else:
"""
name = '_jamc_ft_'

print'os arquivos terão em parte do nome:'+ name

resp = raw_input('Deseja extrair os scores para os frames? ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':

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

        elif '_1.' in name_file:
            print 'no'
            continue
        
        elif '_0.' in name_file:
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
    mse_tempo = np.zeros((400,2))
    i=0
    tempo=15 

    for each_mse in mse:
        if tempo > 75:
            tempo = 15
        mse_tempo[i][0]=each_mse
        mse_tempo[i][1]=tempo
        i+=1
        tempo+=15

    np.savetxt(SAVE_DIR+'mse_'+name, mse_tempo, delimiter=',')
    count = 15
    ssim_tempo = np.zeros((400, 2))
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

else:
     print 'Done! Exiting.'   

resp = raw_input('Deseja extrair os scores para Previous frames? ')
if str(resp) == 'yes' or str(resp) == 'sim' or str(resp) == 's' or str(resp) == 'y':
    
    obs = glob(SAVE_DIR+'obs_*_1*') 
    
    obs2 = glob(SAVE_DIR+'obs_*_9*')
         
    obs.append(obs2)
    obs = sorted(obs)

    ssim=[]
    mse=[]    

    for each_obs in obs:
        name_obs = os.path.basename(str(each_obs))
        
        #Ex: obs_5_1.png
        if len(name_obs) == 11:
            obs_idx = name_obs[4:5]
            idx_len_11 = name_obs[6:7]
            file_index_obs = int(idx_len_11)
            prev_index_obs = file_index_obs - 1
            
            
        elif len(name_obs) == 12:
            #Ex: obs_50_1.png
            obs_idx = (name_obs[4:6])
            idx_len_12_1d = (name_obs[7:8])

            if obs_idx.isdigit() == True and idx_len_12_1d.isdigit() == True:
                idx_len_12_1d = int(idx_len_12_1d)
                file_index_obs = idx_len_12_1d
                prev_index_obs = file_index_obs - 1
                print ('Ok.')
                
                
            else:
                print('Next option.')
                continue           

        elif len(name_obs) == 12:
            #Ex obs_5_10.png
            obs_idx = (name_obs[4:5])
            idx_len_12_2d = (name_obs[6:8])

            if obs_idx.isdigit() == True and idx_len_12_2d.isdigit() == True:
               idx_len_12_2d = int(idx_len_12_2d)
               
                
            else:
                print('Next option.')
                continue

        #Ex:obs_50_10.png
        elif len(name_obs) == 13:
            obs_idx = name_obs[4:6]
            idx_len_13 = int(name_obs[7:9])
            file_index_obs = idx_len_13
            prev_index_obs = file_index_obs - 1
            
            
        name_file =  name_obs[0:3]+ '_' + str(obs_idx) + '_' + str(file_index_obs)+ '_' + name_obs[0:3] + '_' + str(obs_idx) + '_' + str(prev_index_obs)
        
        #Avoinding incorrect files.
        if name_file[0:7] == name_obs[0:3] + str(obs_idx) + '_' + str(file_index_obs):
            print 'no'
            continue

        elif name_file == 'obs_0_10_obs_0_0':
            print 'no'
            continue

        elif name_file == 'obs_1_1_obs_1_0':
            print 'no'
            continue

        elif name_file == 'obs_2_1_obs_2_0':
            print 'no'
            continue

        elif name_file == 'obs_3_1_obs_3_0':
            print 'no'
            continue

        elif '_9_o' in name_file:
            print 'no'
            continue
            
        elif '_0_' in name_file:
            print 'no'
            continue
            
        elif '_1_' in name_file:
            print 'no'
            continue

        elif '_1.' in name_file:
            print 'no'
            continue

        elif '_0.' in name_file:
            print 'no'
            continue
        
        #Processing the correct ones.
        else:
            obs_processing = io.imread(DATA_DIR + name_obs[0:3]+ '_' + str(obs_idx) + '_' + str(file_index_obs) + '.png')
            pred_processing = io.imread(DATA_DIR + name_obs[0:3]+ '_' + str(obs_idx) + '_' + str(prev_index_obs) + '.png')
            m = eqm(obs_processing,pred_processing)
            m /= float (obs_processing.shape[0]*obs_processing.shape[1])
            mse.append(m)
            s1 = s(obs_processing,pred_processing,multichannel=True )
            ssim.append(s1)
            print'comparison done: ',name_file ,m ,s1

    print 'Pronto =)'

    count = 15

    #Empty matrix to stores score values.
    mse_tempo = np.zeros((400,2))

    i=0

    tempo=15 

    for each_mse in mse:
        if tempo > 75:
            tempo = 15
        mse_tempo[i][0]=each_mse
        mse_tempo[i][1]=tempo
        i+=1
        tempo+=15

    np.savetxt(SAVE_DIR+'mse_prev_'+name,mse_tempo, delimiter=',')

    count = 15

    #Empty matrix to stores score values.
    ssim_tempo = np.zeros((400,2))

    i=0

    tempo=15

    for each_ssim in ssim:
        if tempo > 75:
            tempo = 15
        ssim_tempo[i][0]=each_ssim
        ssim_tempo[i][1]=tempo
        i+=1
        tempo+=15

    np.savetxt(SAVE_DIR+'ssim_prev_f'+name,ssim_tempo, delimiter=',')
    print 'arquivos prontos e salvos em: ',SAVE_DIR   

else:
     print 'Done! Exiting.'
     




