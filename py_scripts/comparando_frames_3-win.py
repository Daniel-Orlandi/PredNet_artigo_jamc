from skimage.measure import compare_ssim as ssim
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os.path

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

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
    plt.imshow(imageB, cmap = plt.cmap)
    plt.axis("off") 
    # show the images
    SAVE_DIR='D/IC/IC_GEPAC/PredNet_artigo_jamc/kitti_results/prediction_plots_artigo_jamc_finetunned_scores/'
    plt.savefig(SAVE_DIR+title,dpi=300,bbox_inches='tight')
    plt.close()
    print('saving!!')

DATA_DIR='/home/daniel/PredNet_artigo_jamc/kitti_results/prediction_plots_artigo_jamc_finetunned/'
DATA_DIR_2='D/IC/IC_GEPAC/PredNet_artigo_jamc/kitti_results/prediction_plots_artigo_jamc_finetunned/'
obs = glob(DATA_DIR_2+'obs_*_*') 
pred = glob(DATA_DIR_2+'pred_*_*')
obs = sorted(obs)
pred= sorted(pred)
obs_pred = zip(obs,pred)

#Current frame SSIM e RMSE
"""
for each_obs, each_pred in obs_pred:
    
     name_obs = os.path.basename(each_obs)
     name_pred = os.path.basename(each_pred)
     name_file =  name_obs+'_'+name_pred
     obs_processing = io.imread(each_obs)
     pred_processing = io.imread(each_pred)
     compare_images(obs_processing,pred_processing,str(name_file))
     print( 'image: '+str(name_file)+' saved!')
"""
#Previous Frame SSIM e RMSE
i = 1	 
for each_obs, each_pred in obs_pred:
    name_obs = os.path.basename(each_obs)
    current_frame_index = name_obs[6:7]
    if current_frame_index == '0':
        previous_frame_index = 0

    else:
        previous_frame_index = int(current_frame_index) - 1

    if i >= 10:
        current_frame_index = name_obs[6:8]
        
    new_file_name = name_obs[0:6]
    name_file =  new_file_name + str(current_frame_index) + '_' + new_file_name + str(previous_frame_index) + '.png'   
    obs_processing = io.imread(DATA_DIR_2 + new_file_name + str(current_frame_index)+'.png')
    pred_processing = io.imread(DATA_DIR_2 + new_file_name + str(previous_frame_index)+'.png')
    compare_images(obs_processing,pred_processing,str(name_file))
    i += 1
    print( 'image: '+str(name_file)+' saved!' )


print ('Pronto =)')

