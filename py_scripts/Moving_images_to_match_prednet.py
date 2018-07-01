
# coding: utf-8

# In[7]:

from glob import glob
import os.path
from collections import OrderedDict
import shutil
import os.path
from collections import OrderedDict


# In[2]:

DATA_DIR='/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/imgs/'
data_dir=glob (DATA_DIR+'*')


# In[3]:

anos = []
for a in data_dir:
    anos.append(os.path.basename(a))
print anos


# In[4]:

dia_mes = []
for ano in anos:
    data_dir=glob(DATA_DIR+ano+'/*')
    for item in data_dir:
        dia_mes.append(os.path.basename(item))
print len(dia_mes)


# In[5]:

im_dir =[]
for each_ano in anos:
     imagem = os.path.join(DATA_DIR,each_ano)
     im_dir.append(imagem)
print im_dir


# In[6]:

part_dir = []
for each_element in im_dir:
    for each_day in dia_mes:
        a =os.path.join(each_element,each_day,'ams_temp_alta/')
        if os.path.exists(a):
            part_dir.append(a)
part_dir = list(OrderedDict.fromkeys(part_dir))
print part_dir[1]
    


# In[7]:

complete_dir= []
for each_element in part_dir:
    _,_,img_list = os.walk(each_element).next()
    for each_image in img_list:
        a = os.path.join(each_element,each_image)
        if os.path.exists(a): complete_dir.append(a)
print len (complete_dir)    
complete_dir=list(OrderedDict.fromkeys(complete_dir))
print len(complete_dir)


# In[8]:

print complete_dir[1]


# In[9]:

dest = []
base_dir=DATA_DIR+'raw/'
if not os.path.exists(base_dir): os.mkdir(base_dir)
    
for each_element in complete_dir:    
    file_name = os.path.basename(each_element)
    ano = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(each_element))))
    dia = os.path.basename(os.path.dirname(os.path.dirname(each_element)))
    destination = os.path.join(base_dir,ano,dia,file_name)
    destination_f = os.path.join(base_dir,ano,dia)
    if not os.path.exists(os.path.join(base_dir,ano)): os.mkdir(os.path.join(base_dir,ano))
    if not os.path.exists(destination_f): os.mkdir(destination_f)
    dest.append(destination)
    
print len(dest)
dest=list(OrderedDict.fromkeys(dest))
print len(dest)
    


# In[10]:

a=complete_dir
b=dest
for each_src,each_dest in zip(a,b):
    each_dest = os.path.dirname(each_dest)+'/'
    shutil.copy(each_src,each_dest)
print 'done'    
    


# In[4]:

anos = []
a=glob('/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/imgs/raw/*')
for each_year in a:
    ano = os.path.basename(each_year)
    anos.append(ano)
print anos


# In[6]:

selecao = anos [4:]
print selecao


# In[14]:

num_images = []
for each_year in selecao:
    a = '/media/daniel/Local-Disk-HDD-500GB/IC/IC_GEPAC/dados/imgs/raw/'+each_year+'/*/*.jpg'
    print a
    b = glob(a)
    for each_item in b:
        print each_item
        num_images.append(each_item)
num_images=list(OrderedDict.fromkeys(num_images))
print len(num_images)


# In[21]:

numero_images = len(num_images)
prop_test= (15*(numero_images))/100
prop_val=(5*(numero_images))/100
print prop_test/96, prop_val/96


# In[ ]:



