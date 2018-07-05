
# coding: utf-8

# In[1]:

import hickle as hkl
import random
from glob import glob 
import random
import os.path
from collections import OrderedDict


# In[4]:

DATA_DIR = glob('/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/dados_prednet_mono/raw/2013/*/*')
DATA_DIR_2 = glob('/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/dados_prednet_mono/raw/2012/*/*')
DATA_DIR_3 = glob('/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/dados_prednet_mono/raw/2011/*/*')
print len(DATA_DIR)
print len(DATA_DIR_2)
print len (DATA_DIR_3)
nt = 15
total = len(DATA_DIR)+len(DATA_DIR_2)+len(DATA_DIR_3)
test = 15*(total)/100
val =  5*(total)/100                         
print 'Imagens para teste:',float(test),',',float(test)/96,'Dias'
print 'Imagens para Validar:',float(val),',',float(val)/96,'Dias'
print 'Imagens no conjunto de treinamento:',(total-(test+val)),',',(total-(test+val))/96,'Dias'
print 'Sequencias Treinamento:',(total-(test+val))/nt
print 'Sequencias Teste:',float(test)/nt
print 'Sequencias Val:',float(val)/nt


# In[5]:


test=hkl.load('/home/daniel/AnacondaProjects/prednet-master_2/kitti_data/X_test.hkl')
train = hkl.load('/home/daniel/AnacondaProjects/prednet-master_2/kitti_data/X_train.hkl')
val = hkl.load('/home/daniel/AnacondaProjects/prednet-master_2/kitti_data/X_val.hkl')


# In[6]:

print train.shape
print test.shape
print val.shape


# In[33]:

#testando esse bloco(Seleciona imagens para teste e para validação)
test_recordings = []
val_recordings = []
Dir = glob('/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/imgs/raw/201*/*')
count = 0
while count <= 161:
    test_recordings.append(random.choice(Dir))
    count = count + 1
count = 0
while count <= 53:
    a = random.choice(Dir)
    for each_t in test_recordings:
        if a != each_t:
            val_recordings.append(a)

    count = count + 1
val_recordings=OrderedDict.fromkeys(val_recordings)

day=[]
year=[]

for each_day in test_recordings:
    day.append(os.path.basename(each_day))
    year.append(os.path.basename(os.path.dirname(each_day)))
year_day_test = zip(year, day)

day_v=[] 
year_v =[]
for each_day in val_recordings:
    day_v.append(os.path.basename(each_day))
    year_v.append(os.path.basename(os.path.dirname(each_day)))
year_day_val = zip(year_v, day_v)

test_recordings=year_day_test
val_recordings=year_day_val


# In[34]:

#Teste progressbar
import time
import progressbar

bar = progressbar.ProgressBar()
for i in bar(range(100)):
    time.sleep(0.02)


# In[39]:

count = 0
while count <= 590:
    count= count+15
    print count
    


# In[ ]:



