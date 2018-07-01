from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as eqm
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os.path
DATA_DIR='/home/daniel/AnacondaProjects/prednet-master_2/kitti_results/prediction_plots_mono/'
DATA_DIR_2='/home/daniel/AnacondaProjects/prednet-master_2/kitti_results/prediction_plots_mono_tw/'
def mse(imageA, imageB):
    	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
	#err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
    
    
    
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB,multichannel=True)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.6f, SSIM: %.4f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off") 
    # show the images
	SAVE_DIR='/home/daniel/AnacondaProjects/prednet-master_2/kitti_results/prediction_plots_mono_tw/scores_prednet_tw/'
	plt.savefig(SAVE_DIR+title,dpi=600,bbox_inches='tight')
	plt.close()
	print 'saving!!'

obs = glob(DATA_DIR_2+'obs_*_*') 
pred = glob(DATA_DIR_2+'pred_*_*')
obs = sorted(obs)
pred= sorted(pred)
obs_pred = zip(obs,pred)
for each_obs, each_pred in obs_pred:
    name_obs = os.path.basename(each_obs)
    name_pred = os.path.basename(each_pred)
    name_file =  name_obs+'_'+name_pred
    obs_processing = io.imread(each_obs)
    pred_processing = io.imread(each_pred)
    compare_images(obs_processing,pred_processing,name_file)
    print'comparison done:',name_file
print 'Pronto =)'