import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import matplotlib.patches as mpatches

#Falta colocar nome do arquivo pra salvar. Onde o .csv tem que ser cortado. Opção para plotar o prev frame nos gráficos. Instalar no python. Colocar opção de dpi, posição das legendas, limites dos graficos etc...
class BAR_PLOT(object):

    def __init__(self, data_dir, file_1_mse, file_2_mse_prev, file_1_ssim, file_2_ssim_prev, previous_frame_res = False):

        self.data_dir = data_dir
        self.file_1_mse = file_1_mse
        self.file_2_mse_prev = file_2_mse_prev
        self.file_1_ssim = file_1_ssim
        self.file_2_ssim_prev = file_2_ssim_prev
        self.previous_frame_res = previous_frame_res

    def data_preparation(self):
        mse_df = pd.DataFrame(self.data_dir + self.file_1)
        ssim_df = pd.DataFrame(self.data_dir + self.file_1_ssim)
        ssim_df = ssim_df.loc[0:20]
        mse_df = mse_df.loc[0:20]
        
        if self.previous_frame_res == True:
            mse_df_prev = pd.DataFrame(self.data_dir + self.file_2_prev)
            ssim_prev_df = pd.DataFrame(self.data_dir + self.file_2_ssim_prev)
            ssim_prev_df = ssim_prev_df.loc[0:20]            
            mse_df_prev = mse_df_prev[0:20]

        else:
            ssim_prev_df, mse_df_prev = None


        
        return ssim_df, ssim_prev_df, mse_df, mse_df_prev

    def calc_mean(d_set):
        mean = []
        results_dict = {}
        time_axis = [15, 30, 45, 60, 75]
        
        for each_time_stamp in time_axis:
            pre_mean = d_set.loc[d_set[var_time] == each_time_stamp]
            out_value = np.asarray(pre_mean)
            out_value = np.mean(out_value,0)
            out_value = out_value[0]
            
            results_dict[each_time_stamp] = out_value
        
        return results_dict
    
    def create_means(self):   
        mean_SSIM = calc_mean(ssim_df, var_time='Time(min)', var_name='SSIM')
        mean_MSE = calc_mean(mse_df, var_time='Time(min)', var_name='MSE')
      
        if self.previous_frame_res == True:
            mean_MSE_Previous = calc_mean(mse_df_prev, var_time='Time(min)', var_name='MSE Previous')
            mean_SSIM_Previous = calc_mean(ssim_prev_df, var_time='Time(min)', var_name='SSIM Previous')
        
        else:
            mean_MSE_Previous, mean_SSIM_Previous = None

        return mean_MSE, mean_MSE_Previous, mean_SSIM, mean_SSIM_Previous

    def convert_to_list(dict0):
    res_list = list(reversed(dict0.values()))
    return res_list

    def mse_bar_plot(self, mean_file):
        d1 = convert_to_list(mean_file)
        #d2 = convert_to_list(mean_MSE_Previous)

        ind = np.arange(len(d1))
        width = 0.73
        fig, ax = plt.subplots() 
        p1 = ax.bar(ind, d1, width = -width*0.5, align = 'edge', color='xkcd:orange')
        #p2 = ax.bar(ind, d2, width = 0.5*width, align = 'edge', color='xkcd:azure')  # Uncomment for previous frames be included in plot.

        # Graph Title.
        ax.set_title('Mean MSE by forecast time', fontsize = 15)

        # Axis Label.
        ax.set_ylabel('MSE', fontsize = 13)
        ax.set_xlabel('Time', fontsize = 13)

        # x Ticks.
        ax.set_xticks(ind)
        ax.set_xticklabels(('15 min', '30 min', '45 min', '60 min', '75 min'))

        # y Ticks.
        ax.set_yticks(np.arange(0, 0.00032, 0.00002))
        ax.set_ylim(0.00000, 0.00038)
        ax.ticklabel_format(axis = 'y', style='sci', scilimits = (0 , 0.00015 ))

        # legend
        #azure_patch = mpatches.Patch(color='xkcd:azure', label='Previous Frame MSE')
        orange_patch = mpatches.Patch(color='xkcd:orange', label='Forecast MSE')
        #ax.legend(handles=[azure_patch,orange_patch]) # Uncomment for previous frames be included in plot.
        ax.legend(handles=[orange_patch]) # Forecast data only. Comment for previous frames be included in plot.

        # Text over bars.
        def label_above_bar (data):
            for each_data in data:         
                height = each_data.get_height()
                ax.text(each_data.get_x() + each_data.get_width()/2., 1.05*height, str('{0:.1E}'.format(Decimal(height))), ha = 'center', va='bottom')
        label_above_bar(p1)
        #label_above_bar(p2) # Uncomment for previous frames be included in plot.

        #Save and show barplot.
        plt.savefig('/home/daniel/PredNet_artigo_jamc/kitti_results/plots/mean_mse_by_forecast_time_ft_frcst_only_1.jpg',format='jpg', dpi = 400)

    def ssim_bar_plot(self, mean_ssim_file):
        d1 = convert_to_list(mean_SSIM)
        #d2 = convert_to_list(mean_SSIM_Previous)

        ind = np.arange(len(d1))
        width = 0.73
        fig, ax = plt.subplots() 
        p1 = ax.bar(ind, d1, width = -width*0.5, align = 'edge', color='xkcd:orange')
        #p2 = ax.bar(ind, d2, width = 0.5*width, align = 'edge', color='xkcd:azure')  Uncomment for previous frames be included in plot.

        # Graph Title.
        ax.set_title('Mean SSIM by forecast time', fontsize = 15)

        # Axis Label.
        ax.set_ylabel('SSIM', fontsize = 13)
        ax.set_xlabel('Time', fontsize = 13)

        # x Ticks.
        ax.set_xticks(ind)
        ax.set_xticklabels(('15 min', '30 min', '45 min', '60 min', '75 min'))

        # y Ticks.
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_ylim(0, 1.3)


        # legend
        #azure_patch = mpatches.Patch(color='xkcd:azure', label='Previous Frame SSIM') Uncomment for previous frames be included in plot.
        orange_patch = mpatches.Patch(color='xkcd:orange', label='Forecast SSIM')
        #ax.legend(handles=[azure_patch,orange_patch]) Comment for previous frames be included in plot.
        ax.legend(handles=[orange_patch])

        # Text over bars.
        def label_above_bar (data):
            for each_data in data:         
                height = each_data.get_height()
                ax.text(each_data.get_x() + each_data.get_width()/2., 1.05*height, str('%.2f' %(height)), ha = 'center', va='bottom')
        label_above_bar(p1)
        #label_above_bar(p2) Uncomment for previous frames be included in plot.


        plt.savefig('/home/daniel/PredNet_artigo_jamc/kitti_results/plots/mean_ssim_by_forecast_time_ft_rcst_only_1.jpg',format='jpg', dpi = 400)
        plt.show()

