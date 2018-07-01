
# coding: utf-8

# In[29]:

import skimage
import skimage.io as io
import skimage.transform
from scipy import misc
import scipy
import numpy as np
from glob import glob
from collections import OrderedDict


# In[38]:

dirAnos= glob('/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/*')

years=[]
for each_folder in dirAnos:
    years.append(each_folder)
years = years[:1]
print years
path =[]
for each_year in  years:
    path_to_img_folder = each_year+'/*/ams_temp_alta/*.jpg'
    print path_to_img_folder
    path.append(path_to_img_folder)
path = list(OrderedDict.fromkeys(path))#Remove duplicados
print 'Done!'
dirAnos = path
print dirAnos


# In[39]:


for item in dirAnos:
    diretorio=glob(item)
    for Dir in diretorio:
        try:
        
            IMAGE_LOCATION = '{}'.format(Dir)   
            satImg = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
            satImg = satImg[20:1320,:,:]
            print "Resizing to 512x1392"
            img256=skimage.transform.resize(satImg, (512, 1392))          
            io.imsave(IMAGE_LOCATION,img256)
            print "Saving"
            #misc.imsave(IMAGE_LOCATION,img256) Funciona igual ao de cima
        except:
               pass #ignora excessões:imagens com problema
    
print "Conversão Completa!"


# In[ ]:



