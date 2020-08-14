# import modules
import matplotlib # comment out if running on tux
matplotlib.use('agg') # now it works via ssh connection; # comment out if running on tux


import os
import sys
import glob
import pickle
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/home/jalilov1/BB/AB_R/DvM') # Tux
# sys.path.append('/Volumes/psychology$/Researchers/alilovic/AB_R/DvM')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from visuals.visuals import MidpointNormalize
from support.FolderStructure import *
from support.support import *

from scipy import stats
from stats.nonparametric import *
from scipy.stats import spearmanr
import copy
import csv

data_path = ('/home/jalilov1/BB/AB_R/data') #data folder
os.chdir(data_path) #data folder

# set general plotting parameters
sns.set(font_scale=2.5)
sns.set_style('ticks', {'xtick.major.size': 20, 'ytick.major.size': 20})

# inspired by http://nipunbatra.github.io/2014/08/latexify/
params = {
    'axes.labelsize': 20, # fontsize for x and y labels (was 10)
    'axes.titlesize': 25,
    'font.size': 10, # was 10
    'legend.fontsize': 20, # was 10
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'font.family': 'arial',
}

matplotlib.rcParams.update(params)



# general plotting parameters
f = plt.figure(figsize = (40,30))
times = np.linspace(-200,900,140)
norm = MidpointNormalize(midpoint=1/2.0)
vmin, vmax = 0.45, 0.6
ylim = (0.45,0.6)




class AB_R_Plots(object):


    def clusterPlot(self, X1, X2, p_val, times, y, color, ls = '-', lw = 3):
        '''
        plots significant clusters in the current plot
        ''' 
        s = []
        e=[]
        # indicate significant clusters of individual timecourses
        sig_cl = clusterBasedPermutation(X1, X2, p_val = p_val)
        mask = np.where(sig_cl < 1)[0]
        sig_cl = np.split(mask, np.where(np.diff(mask) != 1)[0]+1)
        for cl in sig_cl:
            if len(cl) > 0:
                print 'plotting cluster from {} to {} ms'.format(times[cl][0],times[cl][-1])
            plt.plot(times[cl], np.ones(cl.size) * y, color = color, ls = ls, lw = lw)
        if len(cl)>0:
            s, e = times[cl][0], times[cl][-1] 

        return s, e

    def beautifyPlot( y = 0, xlabel = 'Time (ms)', ylabel = 'Mv', ls = '-'):
        '''
        Adds markers to the current plot. Onset placeholder and onset search and horizontal axis
        '''

    #     plt.axhline(y=y, ls = ls, color = 'black')
    #     plt.axvline(x=-250, ls = ls, color = 'black')
    #     plt.axvline(x=0, ls = ls, color = 'black')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        
    def colorbar( mappable, ticks):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.xaxis.set_ticks_position("top")
        return fig.colorbar(mappable, cax=cax, ticks = ticks) 




    def plotStimTemporalOrder(self):


        '''
        Plots targets and distractors, collapsed over all conditions; diagonal
        '''
        f = plt.figure(figsize = (12,5))
        times = np.linspace(-200,900,140)
        stim =  [['T1', 'T2', 'T3'], ['D1', 'D2'] ]
        colors = [['blue','green', 'purple'], ['red','pink','orange'] ]

        ylim = (0.47,0.55)
       
        for bidx, B in enumerate (['T', 'D']): 

            ax = plt.subplot(1,2, bidx+1)

            for i, T in enumerate(stim[bidx]):

                files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline','bdm','cross', 'T1detected', 'all'], filename = 'class_*-{}.pickle'.format(T)))

                bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
               
                X = np.stack([np.diag(b['collapsed']['standard']) for b in bdm])
                PO.clusterPlot(X, 1/2.0, p_val = 0.05, times = times, y = 0.48 + i * 0.002, color = colors[bidx][i])

                err_t, X_t = bootstrap(X)
                plt.plot(times, X_t, label = T, color = colors[bidx][i])
                plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = colors[bidx][i])
                
                plt.title(['Target representations', 'Distractor representations'][bidx], fontsize = 20)
             
                PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')

                plt.ylim(ylim)
                plt.yticks([ylim[0],1/2.0,ylim[1]])

                plt.xticks([0,300, 600, 900]) 

                plt.axhline(y = 0.5, ls = '--',color='k')
                plt.axvline(x = 0,  ls = '--',color='k')

                plt.legend(loc = 'best', frameon=None)
                sns.despine(offset=10, trim = False)


        # plt.tight_layout()
        plt.show()
        plt.savefig(FolderStructure.FolderTracker(filename = 'Stimuli_per_temp_positions.pdf'))

     
        
    def plotERPs(self, stim):

        # ***

        all_evokeds1 = []
        all_evokeds2 = []
        difference_wave =[]
        e_seen=[]
        e_unseen = []
        late=[]
        e_boost = []
        e_bounce = []

        cnd = ['TDTDD', 'TDDTD', 'TDTTD'] # just AB
        # cnd = ['T..DDDT', 'TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD']
        tshift = [0, -0.083, -0.083*2, -0.083*3, -0.083, -0.083, -0.083*2] 
        pos_info = {'pos_13': -0.664,'pos_14': -0.747,'pos_15': -0.83,'pos_16': -0.913}
        pos_header = 'Cnd_1-T2_pos'

        for sj, subjects in enumerate([1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35]):
        # for sj, subjects in enumerate([1,3,4]):

            # all_evokeds1 = []
            # all_evokeds2 = []

            files = glob.glob(FolderStructure.FolderTracker(['AB','processed'], filename = 'subject-{}_all-epo.fif'.format(subjects)))
            files_beh  = glob.glob(FolderStructure.FolderTracker(['AB','beh','processed'], filename = 'subject-{}_all-new.pickle'.format(subjects)))
            # Read in EEG epochs         
            epochs = mne.read_epochs(files[0]) 
            
            # time_windows = ((.2, .25), (.35, .45))
            # elecs = ["Fz", "Cz", "Pz"]
            # # display the EEG data in Pandas format (first 5 rows)
            # # print(epochs.to_data_frame()[elecs].head())

            epochs.crop(tmin = -0.2, tmax = 1.2)
                        
            #Read in behavior matrix to determine EEG trial selection
            beh_ = pickle.load(open(files_beh[0],'rb')) 
            beh = pd.DataFrame.from_dict(beh_)
            # embed()

            # select T2 correct trials
            outputs = []
            for session in [1,2]:
                file = FolderStructure.FolderTracker(extension = ['BEH' ,'results'], 
                    filename = 'abr_datamat_{}_sb{}.csv'.format(session,subjects))
                outputs.append(pd.read_csv(file))
            outputs = pd.concat(outputs)

            # get the clean trials and control for collapsing
            idx_collapse = np.where(np.diff(beh['nr_trials'])<0)[0][0] # end of the first session 1072
            trial_idx = beh['nr_trials'].values
            trial_idx[idx_collapse:] += 1072
            
            # now select the correct t2's
            beh['T2_correct'] = outputs['t2_ide_any'].values[trial_idx - 1]

            # now select the correct t1's
            beh['T1_correct'] = outputs['t1_ide_any'].values[trial_idx - 1]

        
            #OBJECT
            # seen = (beh['T2_correct']==1) & ( beh.condition.isin(cnd))
            # unseen = (beh['T2_correct']==2) & ( beh.condition.isin(cnd))
            # eeg_seen = epochs[seen]
            # eeg_unseen = epochs[unseen]
            # # Average over all  trigger types
            # evokeds_seen= eeg_seen['1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009'].average() 
            # evokeds_unseen= eeg_unseen['1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009'].average() 
      

            # Select conditions where targets appeared at the same lag, otherwise jitter dampens ERPs or shift data

            # RAW DATA
            # seen = (beh['T2_correct']==1) * (beh['condition'] == cnd)
            # unseen = (beh['T2_correct']==2) * (beh['condition'] ==cnd)
            if stim == 'T2':
                for cidx, c in enumerate (cnd):

                    # seen = (beh['T2_correct']==1) & ( beh.condition.isin(cnd))
                    # unseen = (beh['T2_correct']==2) & ( beh.condition.isin(cnd))

                    seen = (beh['T2_correct']==1) * (beh['condition'] == c) 
                    unseen = (beh['T2_correct']==2) * (beh['condition'] ==c)

                    # shift T2 timings to zero
                    if c != 'T..DDDT':
                        to_shift = int(np.diff([np.argmin(abs(epochs.times - t)) for t in (0,tshift[cidx])]  ))
                        if to_shift < 0:
                            print('EEG data is shifted backward in time for all {} trials'.format(c))
                        elif to_shift > 0:
                            print('EEG data is shifted forward in time for all {} trials'.format(c))  
                        
                        eeg_seen = np.roll(epochs[seen]._data, to_shift, axis=2) 
                        eeg_unseen = np.roll(epochs[unseen]._data, to_shift, axis=2)

                    else:
                        for pos in pos_info.keys():
                            shift = pos_info[pos]
                            to_shift = int(np.diff([np.argmin(abs(epochs.times - t)) for t in (0,shift)]))
                            if to_shift < 0:
                                print('EEG data is shifted backward in time for all {} trials'.format(pos))
                            elif to_shift > 0:
                                print('EEG data is shifted forward in time for all {} trials'.format(pos))

                            mask = (beh[pos_header] == int(pos[-2:]))
                            
                            mask_seen = seen * mask 
                            mask_unseen = unseen *mask
                            
                            eeg_seen = np.roll(epochs[mask_seen]._data, to_shift, axis=2)
                            eeg_unseen = np.roll(epochs[mask_unseen]._data, to_shift, axis=2)


                    # boost  = beh['condition']=='TTTDD'
                    # bounce  = beh['condition']=='TTDTD'
                    # # Select EEG based on behavior
                    # eeg_boost = epochs [boost]
                    # eeg_bounce = epochs [bounce]

                    # If you want to create ERPs per target type
                    # evokeds= [eeg_boost[name].average() for name in ('1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009')]
                    
                    #RAW
                    # seen = copy.copy(evokeds_seen)
                    # unseen = copy.copy(evokeds_unseen)

                    #this is raw data to plotting
                    e_seen.append(eeg_seen.mean(0)) # average over trials
                    e_unseen.append(eeg_unseen.mean(0)) # these data will be used with plt.plot and need to be in microvolts 
                    difference_wave.append(eeg_seen.mean(0) - eeg_unseen.mean(0))
            
            if stim == 'T3':

                boost =  (beh['condition'] == 'TDTTD') * (beh['T1_correct']==1)
                bounce = (beh['condition'] == 'TTDTD') * (beh['T1_correct']==1)
                eeg_boost = epochs[boost]._data
                eeg_bounce = epochs[bounce]._data
                e_boost.append(eeg_boost.mean(0)) # average over trials
                e_bounce.append(eeg_bounce.mean(0)) # these data will be used with plt.plot and need to be in microvolts 


            else: #T1

                seen = (beh['T2_correct']==1) * (beh['condition'] == 'TDTDD') * (beh['T1_correct']==1)
                unseen = (beh['T2_correct']==2) * (beh['condition'] =='TDTDD') * (beh['T1_correct']==1)
                eeg_seen = epochs[seen]._data
                eeg_unseen = epochs[unseen]._data
                e_seen.append(eeg_seen.mean(0)) # average over trials
                e_unseen.append(eeg_unseen.mean(0)) # these data will be used with plt.plot and need to be in microvolts 
                

                # create an ERP for a late T2 condition (on T1 correctly identified trials)
                late_seen = (beh['T2_correct']==1) * (beh['condition'] == 'T..DDDT') * (beh['T1_correct']==1)    
                late_seen_epochs = epochs[late_seen]._data   
                late.append(late_seen_epochs.mean(0)) 
                difference_wave.append(eeg_seen.mean(0) - eeg_unseen.mean(0))


                #OBJECT
                # evokeds_seen._data/=1e6 
                # evokeds_unseen._data/=1e6 
                # print('Microvrolts changed to volts') #default MNE functions expect Volts
                # #Add subjects together - these are mne objects
                # all_evokeds1.append(evokeds_seen)
                # all_evokeds2.append(evokeds_unseen)


            # # Desired time range to plot
            # lat1 = [-0.2, 1]
            # tx1=[]
            # for x, l in enumerate(lat1):
            #     tx1.append(int(np.argmin(abs(epochs.times - l)))) 
            # time = range(tx1[0],tx1[1])   
            # embed()
            # evokeds_seen.times = evokeds_seen.times[time] # change the object
            # evokeds_unseen.times = evokeds_unseen.times[time]
            
            
            # chanSel['OCC'] = ['Oz','O1','O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Iz']
            # chanSel['PAR'] = ['P1', 'P3', 'P5', 'P7', 'Pz', 'P2', 'P4', 'P6', 'P8']
            # chanSel['FRO'] = ['Fp1', 'AF7', 'AF3', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz']
            # chanSel['TMP'] = ['FT7', 'C5', 'T7', 'TP7', 'CP5', 'FT8', 'C6', 'T8', 'TP8', 'CP6']
            # chanSel['OPA'] = ['P1', 'P3', 'P5', 'P7', 'P9', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO7', 'PO3'
            #                        'O1', 'Iz', 'Oz', 'POz', 'PO8', 'PO4', 'O2', 'PO9', 'PO10' ]
            # chanSel['CDA'] = ['P5', 'P6', 'P7', 'P8', 'PO7', 'PO8', 'O1', 'O2', 'PO9', 'PO10']
            # channels = ['Fp1', 'AF7', 'AF3', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz']
    

        channels  = ['POz', 'Pz', 'CPz', 'CP1', 'CP2', 'P1', 'P2', 'PO3', 'PO4']
            # channels = ['FCz','Cz','C1','C2','FC1','FC2']
            # channels = ['FCz', 'FC1', 'FC3']
            # evokeds[0].plot_joint(times=[.15, .3], picks = [12]); 


        # pool = mne.viz.plot_compare_evokeds(dict(seen=all_evokeds1, unseen=all_evokeds2), show_sensors='upper left',  truncate_yaxis=False, truncate_xaxis=True,
                                       # picks=[epochs.ch_names.index(chans) for chans in channels])

        # works for single subject data objects
        # evoked_diff = mne.combine_evoked([evokeds_seen, -evokeds_unseen], weights='equal')
        # evoked_diff.plot_topo(color='r', legend=False)
        # plt.show()
        # plt.savefig(FolderStructure.FolderTracker(filename = 'base22.pdf'))

        #topomaps
        # evokeds_seen.plot_topomap (times=[0., 0.08, 0.1, 0.12, 0.2], ch_type='eeg')
        # pool.plot_topomap(times=[0., 0.08, 0.1, 0.12, 0.2], ch_type='eeg')


        #PLOT ERPS RAW
        embed()


        f = plt.figure(figsize = (12,5))

        ax = plt.subplot(1,2,1) 

        time = epochs.times * 1000
        ch = [epochs.ch_names.index(chans) for chans in channels]

        erp1 =np.stack(e_seen)[:,ch,:]
        erp2 = np.stack(e_unseen)[:,ch,:]
        erp3 = np.stack(late)[:,ch,:]
        diff_erp = np.stack(difference_wave)[:,ch,:]
        plt.plot(time, erp1.mean(0).mean(0), color = 'green', linewidth=2, markersize = 14, label = 'T2 seen')
        plt.plot(time, erp2.mean(0).mean(0), color = 'orange', linewidth=2, markersize = 14, label = 'T2 unseen')
        plt.plot(time, erp3.mean(0).mean(0), color = 'blue', linewidth=2, markersize = 14, label = 'T2 late')

        err_t, X_t = bootstrap(diff_erp.mean(1))
        plt.plot(time, X_t, color = 'purple', linewidth=2, markersize = 14, label = 'Difference')
        plt.fill_between(time, X_t + err_t, X_t - err_t, color = 'purple', alpha = 0.2)

        # cl = PO.clusterPlot(diff_erp.mean(1), 0, p_val = 0.05, times = time, y = -1 , color = 'purple')

        plt.axhline(y = 0, ls = '--',color='k')
        plt.axvline(x = 0,  ls = '--',color='k')

        # plt.ylim(ylim)
        # plt.yticks([ylim[0],1/2.0,ylim[1]])
        # plt.yticks(np.arange(0.48,0.5,0.54))
        plt.xlim(-200, 900)
        plt.xticks([0,300,600, 900])
        plt.xlabel ('Time (ms)')
        plt.ylabel('Microvolts (/V)')
        sns.despine(offset = 10, trim = False)

        plt.title('P3 over CP channels', fontsize = 20)

        plt.legend(loc = 'best')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, bbox_to_anchor=(1.9,1))

        # plt.tight_layout()
        plt.show()

        plt.savefig(FolderStructure.FolderTracker(filename ='T1_P3seenunseen in AB conditions.pdf'))


        # plt.title ('T2 seen versus unseen in AB conditions', fontsize = 20)
        # plt.tight_layout()
        # plt.show()
        # plt.savefig(FolderStructure.FolderTracker(filename = 'ERPs_T2seen_unseen_P3b_channels_all.pdf'.format(subjects)))


    # data = eeg_boost['1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009'].get_data(pick =['Oz','O1','O2','Iz','I1','I2'])

        ## Test the difference between T1 in T2 seen versus unseen trials
        # Tind the time windows for averaging

        lat1 = [350, 550] 
        win_T1 =[]
        for x, l in enumerate(lat1):
            win_T1.append(int(np.argmin(abs(time - l)))) 

        lat2 = [600, 800] 
        win_T2 =[]
        for x, l in enumerate(lat2):
            win_T2.append(int(np.argmin(abs(time - l)))) 

        # 350: 550
        avg_seen_lat1 = erp1[:, :, win_T1[0]:win_T1[1]].mean(1).mean(1)
        avg_unseen_lat1 = erp2[:, :, win_T1[0]:win_T1[1]].mean(1).mean(1)
        t_t1, p_t1 = stats.ttest_rel(avg_seen_lat1, avg_unseen_lat1 )
        
        # 600: 800
        avg_seen_lat2 = erp1[:, :, win_T2[0]:win_T2[1]].mean(1).mean(1)
        avg_unseen_lat2 = erp2[:, :, win_T2[0]:win_T2[1]].mean(1).mean(1)
        t_t2, p_t2 = stats.ttest_rel(avg_seen_lat2, avg_unseen_lat2)




    def plotBehavior2(self):

        # 

        file = FolderStructure.FolderTracker(extension = ['BEH' ,'results'], filename = 'plot_data_behavior_clean.csv') 
        DF = pd.read_csv(file)
        df_beh = DF.drop('subjects',axis=1)
        beh=[]
        df=[]
        #reorder behavior 
        df.append(DF.filter(like='T1'))
        df.append(DF.filter(like='T2'))
        df.append(DF.filter(like='T3'))


        beh = pd.concat(df, axis=1).T
        #remove variables with no data
        beh=beh.loc[(beh!=0).any(axis=1)]
        # Calculate SEM for errorbars
        yerr = beh.sem(axis = 1, skipna = False) 


        beh_cond = beh.mean(1) # mean over subjects so you get averages per TP across conditions
        beh_plot = np.zeros((5,8))
        beh_plot[0,:] = beh_cond[:8]
        beh_plot[1,:7] = beh_cond[8:15]
        beh_plot[2,4:7] = beh_cond[15:]
        BEH = pd.DataFrame (beh_plot, columns =['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD'] )

        # restructure the data for plotting
        BEH['DDTDD'][4] = BEH['DDTDD'][0]
        BEH['DDTDD'][0,1,2,3]=np.nan 

        BEH['T..DDDT'][4]=BEH['T..DDDT'][1]
        BEH['T..DDDT'][1,2,3]=np.nan   

        BEH['TDTDD'][2]=BEH['TDTDD'][1]
        BEH['TDTDD'][1,3,4]=np.nan         

        BEH['TDDTD'][3]=BEH['TDDTD'][1]
        BEH['TDDTD'][1,2,4]=np.nan 

        BEH['TTDTD'][3]=BEH['TTDTD'][2]
        BEH['TTDTD'][2,4]=np.nan 

        BEH['TDTTD'][3]=BEH['TDTTD'][2]
        BEH['TDTTD'][2]=BEH['TDTTD'][1]
        BEH['TDTTD'][1,4]=np.nan 
        BEH['TTDDD'][2,3,4] = np.nan
        BEH['TTTDD'][3,4] = np.nan


        yerrdf = np.zeros((5,8))
        yerrdf[0,:] = yerr[:8]
        yerrdf[1,:7] = yerr[8:15]
        yerrdf[2,4:7] = yerr[15:]
        rois_error = pd.DataFrame (yerrdf, columns =['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD'] )
        # restructure the data for plotting
        rois_error['DDTDD'][4] = rois_error['DDTDD'][0]
        rois_error['DDTDD'][0,1,2,3]=np.nan 

        rois_error['T..DDDT'][4]=rois_error['T..DDDT'][1]
        rois_error['T..DDDT'][1,2,3]=np.nan   

        rois_error['TDTDD'][2]=rois_error['TDTDD'][1]
        rois_error['TDTDD'][1,3,4]=np.nan         

        rois_error['TDDTD'][3]=rois_error['TDDTD'][1]
        rois_error['TDDTD'][1,2,4]=np.nan 

        rois_error['TTDTD'][3]=rois_error['TTDTD'][2]
        rois_error['TTDTD'][2,4]=np.nan 

        rois_error['TDTTD'][3]=rois_error['TDTTD'][2]
        rois_error['TDTTD'][2]=rois_error['TDTTD'][1]
        rois_error['TDTTD'][1,4]=np.nan 
        rois_error['TTDDD'][2,3,4] = np.nan
        rois_error['TTTDD'][3,4] = np.nan

    
        f = plt.figure(figsize = (12,5))
        ax = plt.subplot(1,2,1, title = 'Behavior: discrimination accuracy', ylabel = 'Accuracy (%)', xlabel = 'Temporal positions')

        xlim = (-0.5, 5.5)
        ylim =(0,100)

        marker = ['*', '*', 'o', '*','>', '>', '>', 'o'] 
        color = ['k', 'r', 'g', 'orchid', 'c', 'b', 'y', 'k']
        ticks = ['TP1', 'TP2', 'TP3', 'TP4', 'Late TP']

        for i, cnd in enumerate (['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD']):
            # plt.plot(np.arange(0.25,5,1), rois[cnd], marker[i],  linewidth=3, markersize = 14)
            plt.errorbar(np.arange(0.25,5,1), BEH[cnd], yerr=rois_error[cnd], uplims=False, lolims=False, marker = marker[i], color = color[i], linewidth = 3, markersize = 12)

        plt.title('Behavior: discrimination accuracy')
           

        plt.ylim(ylim)
        plt.yticks([ylim[0],20,40,60,80,ylim[1]])  
        plt.xlim(xlim)
        plt.xticks(np.arange(0.25,5,1), ticks) 


        plt.legend( bbox_to_anchor=(1.9, 1), loc='upper right', frameon=False)
        sns.despine(offset=10, trim = False)

        plt.tight_layout()
        plt.show()
        plt.savefig(FolderStructure.FolderTracker(filename = 'BEH_new.pdf'))



    def BoostBounceNEW(self):

        # 

        # DIAGONAL
        f = plt.figure(figsize = (12,5))
        ylim = (0.47,0.55)
        times = np.linspace(-200,900,140)


        w = [['D2','D1'],['D3','D2']] # distractor in the order you need to load them in

        for r_idx in range (2):

            diag1=[]
            s_dec=[]
            e_dec=[]

            for bidx, B in enumerate (['D']): # 'D' 

                # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'boostbounce', 'all_collapsed', 'T2s'], filename = 'class_*-{}.pickle'.format(B)))
                # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'within', 'boostbounce', 'binary', 'TTTvsTDT'], filename = 'class_*.pickle'.format(B)))
                # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'boostbounce', 'neutral-bounce', 'DDDvsTDD'], filename = 'class_*.pickle'.format(B)))
                
                # for plt_idx, cnd in enumerate(['all']): 
                # for plt_idx, cnd in enumerate(['boost', 'bounce']): 
                for plt_idx, cnd in enumerate(['DDTDD', 'TDDTD']):
                    # this is the correct folder; no separate analysis but you just plotted 2 specific conditions from all data folder for D1 and D2
                    files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'T1detected', 'all'], filename = 'class_*-{}.pickle'.format(w[r_idx][plt_idx])))

                    ax = plt.subplot(1,2, r_idx+1)
                    bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
               
                    if cnd not in bdm[0].keys():
                        continue

                    X = np.stack([b[cnd]['standard'] for b in bdm]) 
                    #Get diagonals for all subjects

                    diagonal = np.stack([np.diag(par) for par in X])

                    # diagonal=X

                    diag1.append(diagonal)

                    s, e = PO.clusterPlot(diagonal, 1/2.0, p_val = 0.05, times = times, y = 0.49 + plt_idx * 0.002, color =  [['black','red' ],['black','orange']][r_idx][plt_idx])
                    s_dec.append(s)
                    e_dec.append(e)

                    err_t, X_t = bootstrap(diagonal)
                    plt.plot(times, X_t, label = cnd, color =  [['black','red' ],['black','orange']][r_idx][plt_idx])
                    plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color =  [['black','red' ],['black','orange']][r_idx][plt_idx])
                   

                    plt.axhline(y = 0.5, ls = '--',color='k')
                    plt.axvline(x = 0,  ls = '--',color='k')

                    plt.ylim(ylim)
                    plt.yticks([ylim[0],1/2.0,ylim[1]])

                    plt.xticks([0,300,600,900]) 

                    PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')
              
                    plt.legend(loc = 'best')
                    handles, labels = ax.get_legend_handles_labels()

                    if r_idx: 
                        ax.legend(handles,('Neutral', 'Bounce'))
                    else:
                        ax.legend(handles,('Neutral', 'Boost'))

                    sns.despine(offset=10, trim = False)

                    # plt.title (['Target representations: T2s only', 'Distractor representations'][bidx], fontsize = 20)
                    plt.title ('Neutral vs. BoostBounce: DDD vs TDD')


        PO.stats_out(diag1, times, start = s_dec, end=e_dec,  lat1=[150, 250], lat2 =[300, 600]) 
        embed()

        # plt.suptitle('Stimulus-specific representations', fontsize = 26, y=1.08)
        
        # plt.tight_layout()
        # plt.legend(['Boost','Bounce'], loc='upper right', bbox_to_anchor=(1.04,1), ncol=3, fancybox=True, shadow=True)
        # plt.savefig(self.FolderTracker(['bdm','all','cross'], filename = 'cross-task_boost_bounce_T2_T3.pdf'))
        plt.savefig(FolderStructure.FolderTracker( filename = 'Neutral vs. BoostBounce: DDD vs TDD - IDENTITY.pdf'))


        # TARGETS

        # w = ['T3','T2'] # distractor in the order you need to load them in
        # diag1=[]
        # s_dec=[]
        # e_dec=[]

        # for bidx, B in enumerate (['T']): # 'D' 

        #     for plt_idx, cnd in enumerate(['TTTDD', 'TDTDD']):

        #         files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'T1detected', 'all'], filename = 'class_*-{}.pickle'.format(w[plt_idx])))

        #         ax = plt.subplot(1,2, 1)
        #         bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
           
        #         if cnd not in bdm[0].keys():
        #             continue

        #         X = np.stack([b[cnd]['standard'] for b in bdm]) 
        #         #Get diagonals for all subjects

        #         diagonal = np.stack([np.diag(par) for par in X])
        
        #         diag1.append(diagonal)

        #         s, e = PO.clusterPlot(diagonal, 1/2.0, p_val = 0.05, times = times, y = 0.49 + plt_idx * 0.002, color =  ['orange', 'pink'][plt_idx])
        #         s_dec.append(s)
        #         e_dec.append(e)

        #         err_t, X_t = bootstrap(diagonal)
        #         plt.plot(times, X_t, label = cnd, color =  ['orange', 'pink'][plt_idx])
        #         plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color =  ['orange', 'pink'][plt_idx])
               

        #         plt.axhline(y = 0.5, ls = '--',color='k')
        #         plt.axvline(x = 0,  ls = '--',color='k')

        #         plt.ylim(ylim)
        #         plt.yticks([ylim[0],1/2.0,ylim[1]])

        #         plt.xticks([0,300,600,900]) 

        #         PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')
          
        #         plt.legend(loc = 'best')
        #         handles, labels = ax.get_legend_handles_labels()

        #         ax.legend(handles,('Boost', 'Bounce'))

        #         sns.despine(offset=10, trim = False)


        #     PO.stats_out(diag1, times, start = s_dec, end=e_dec,  lat1=[150, 250], lat2 =[300, 600]) 
 
        # plt.savefig(FolderStructure.FolderTracker( filename = 'BoostBounce: TTT vs TDT - IDENTITY.pdf'))

    
        # GAT
        times = np.linspace(-200,900,140)
        vmin, vmax = 0.47, 0.55
        norm = MidpointNormalize(midpoint=1/2.0)

        fig = plt.figure(figsize = (12,6))

        for bidx, B in enumerate (['D']): 

            # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'boostbounce', 'all_collapsed'], filename = 'class_*-{}.pickle'.format(B)))
            
            # for plt_idx, cnd in enumerate(['boost', 'bounce']): 
            for plt_idx, cnd in enumerate(['all']):

                ax = plt.subplot(1,2,plt_idx + 1, title = 'boostbounce ', ylabel = 'Time(ms)', xlabel = 'Time (ms)')

                bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack

                X = np.stack([b[cnd]['standard'] for b in bdm])

                # Compute tresholded data
                X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
                contour = np.zeros(X_thresh.shape, dtype = bool)
                contour[X_thresh != 1/2.0] = True
               
                im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation = 'none', 
                origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
                vmin = vmin, vmax = vmax)

                # Plot contoures around significant datapoints
                plt.contour(contour, origin = 'lower',linewidths=0.2, colors = ['black'],extent=[times[0],times[-1],times[0],times[-1]])
                plt.yticks([0,300,600, 900])
                plt.xticks([0,300,600, 900]) 

                sns.despine(offset=False, trim = False)
                plt.axhline(y = 0, ls = '--',color='k')
                plt.axvline(x = 0,  ls = '--',color='k')

                plt.ylabel('Time (ms)')
                plt.xlabel('Time (ms)') 

            # add a colorbar for all figures
            cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.61]) # left, bottom, width, height (range 0 to 1)
            cbar = fig.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])                      
            cbar.set_clim([vmin,vmax])
             # cbar.set_ticks([vmin, 0, vmax])
            cbar.set_ticklabels([str(vmin),'0',str(vmax)])

            # plt.legend(loc = 'best')
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles,('Boost', 'Bounce') )

            sns.despine(offset=0, trim = False)
            # plt.tight_layout()
            plt.show()  
            plt.savefig(FolderStructure.FolderTracker( filename = 'Boostbounce_binary_TDTTvsTDDT-GAT.pdf'))

            

    # explore the post-boost and post-bounce D1 effects 

        shift = -83
        to_shift = int(np.diff([np.argmin(abs(times - t)) for t in (0,shift)]))

        f = plt.figure(figsize = (12,5))
        ylim = (0.47,0.55)
        times = np.linspace(-200,900,140)

        perm=[]


        for bidx, B in enumerate (['D']): 

            files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'boostbounce', 'all_collapsed'], filename = 'class_*-{}.pickle'.format(B)))
            
            for plt_idx, cnd in enumerate(['boost', 'bounce']): 

                ax = plt.subplot(1,2, bidx+1)

                bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
            
                if cnd not in bdm[0].keys():
                    continue
                X = np.stack([ np.roll( b[cnd]['standard'], to_shift, axis = 1)  for  b in bdm]) # shift before you stack; #shift only in test times, train times are based on localizer data
              
                diagonal = np.stack([np.diag(par) for par in X])

                perm.append(diagonal)

                PO.clusterPlot(diagonal, 1/2.0, p_val = 0.05, times = times, y = 0.48 + plt_idx * 0.002, color = [['blue', 'green' ],['pink','orange']][bidx][plt_idx])

                err_t, X_t = bootstrap(diagonal)
                plt.plot(times, X_t, label = B, color =  [['blue', 'green' ],['pink','orange']][bidx][plt_idx])
                plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = [['blue', 'green' ],['pink','orange']][bidx][plt_idx])
               

                plt.axhline(y = 0.5, ls = '--',color='k')
                plt.axvline(x = 0,  ls = '--',color='k')

                plt.ylim(ylim)
                plt.yticks([ylim[0],1/2.0,ylim[1]])

                plt.xticks([0,300,600,900]) 

                PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')
          
                plt.legend(loc = 'best')
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,('Boost', 'Bounce') )

                sns.despine(offset=10, trim = False)

                plt.title ('Distractor post boost and bounce effects' , fontsize = 20)
             
                if len(perm) ==2: 
                #     PO.clusterPlot(perm[0], perm[1], p_val = 0.05, times = times, y = 0.475 + plt_idx * 0.002, color = 'red')

                    [p_value1, T_01] = permutationTTest(perm[0], perm[1], 1000) # permutation t test
                    mask1 = np.where (p_value1 < 0.05)[0]
                    for p in mask1:
                        plt.plot(times[p], 1 * 0.475 + bidx * 0.002, 'ro')

            plt.tight_layout()
            plt.savefig(FolderStructure.FolderTracker( filename = 'Diagonal_POSTEFFECTSboostbounce.pdf'))

        
            times = np.linspace(-200,900,140)
            vmin, vmax = 0.47, 0.55
            norm = MidpointNormalize(midpoint=1/2.0)

            fig = plt.figure(figsize = (12,6))

            for bidx, B in enumerate (['D']): 

                files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'boostbounce', 'all_collapsed'], filename = 'class_*-{}.pickle'.format(B)))
                
                for plt_idx, cnd in enumerate(['boost', 'bounce']): 

                    ax = plt.subplot(1,2,plt_idx + 1, title = 'Distractors ' + cnd , ylabel = 'Time(ms)', xlabel = 'Time (ms)')

                    bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack

                    X = np.stack([ np.roll( b[cnd]['standard'], to_shift, axis = 1)  for  b in bdm])

                    # Compute tresholded data
                    X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
                    contour = np.zeros(X_thresh.shape, dtype = bool)
                    contour[X_thresh != 1/2.0] = True
                   
                    im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation = 'none', 
                    origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
                    vmin = vmin, vmax = vmax)

                    # Plot contoures around significant datapoints
                    plt.contour(contour, origin = 'lower',linewidths=0.2, colors = ['black'],extent=[times[0],times[-1],times[0],times[-1]])
                    plt.yticks([0,300,600, 900])
                    plt.xticks([0,300,600, 900]) 

                    sns.despine(offset=False, trim = False)
                    plt.axhline(y = 0, ls = '--',color='k')
                    plt.axvline(x = 0,  ls = '--',color='k')

                    plt.ylabel('Time (ms)')
                    plt.xlabel('Time (ms)') 

                # add a colorbar for all figures
                cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.61]) # left, bottom, width, height (range 0 to 1)
                cbar = fig.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])                      
                cbar.set_clim([vmin,vmax])
                 # cbar.set_ticks([vmin, 0, vmax])
                cbar.set_ticklabels([str(vmin),'0',str(vmax)])

                sns.despine(offset=0, trim = False)
                # plt.tight_layout()
                plt.show()  
                plt.savefig(FolderStructure.FolderTracker( filename = 'Distractor_POSTboostbounceGAT.pdf'))



    def stats_out(self, diag1, times, start = [100, 100], end=[800, 800], lat1 = [150, 250], lat2 = [300, 600]):

        #

        # lat1, lat2 = [150, 250], [300, 600]


        tx1=[]
        for x, l in enumerate(lat1):
            tx1.append(int(np.argmin(abs(times - l)))) # limit for the testing time
        tx2=[]
        for x, l in enumerate(lat2):
            tx2.append(int(np.argmin(abs(times - l)))) # limit for the testing time
    
        # EARLY WINDOW
        v1erl = diag1[0][:,tx1[0]:tx1[1]+1].mean(1)# condition 1; mean over that time window
        v2erl = diag1[1][:,tx1[0]:tx1[1]+1].mean(1) # condition 2

        #LATE WINDOW
        v1lat = diag1[0][:,tx2[0]:tx2[1]+1].mean(1)# condition 1; mean over that time window
        v2lat = diag1[1][:,tx2[0]:tx2[1]+1].mean(1) # condition 2


        t_early, p_early = stats.ttest_rel(v1erl, v2erl)
        t_late, p_late = stats.ttest_rel(v1lat, v2lat)

        peak1 = np.argmax(diag1[0].mean(0)) # restrict the range
        peak_value1 = diag1[0].mean(0)[peak1]
        peak_time1 = times[peak1]

        peak2 = np.argmax (diag1[1].mean(0))
        peak_value2 = diag1[1].mean(0)[peak2]
        peak_time2 = times[peak2]

        embed()
        #store this int a panda dataframe and write out to a csv file
        bay = [v1erl ,v1lat, v2erl, v2lat]
        bay_out = pd.DataFrame(bay).T
        bay_out.columns=['neutral D early', 'neutral D late', 'bounce D early', 'bounce D late']
        bay_out.to_csv('distractors_neutral_bounce_cross task identity decoding-Bayes-4JASP.csv')

        # stat_out = pd.DataFrame(index = range(1),columns=['T2 seen peak', 'T2 seen peak time', 'T2 unseen peak', 'T2 useen peak time', 't-stat early', 'p-value early', 't-stat late', 'p-value late', 'start1', 'end1', 'start2', 'end2'])
        # stat_out = pd.DataFrame(index = range(1),columns=['T1 peak', 'T1 peak time', 'D1 peak', 'D1 peak time', 't-stat early', 'p-value early', 't-stat late', 'p-value late', 'start1', 'end1', 'start2', 'end2'])
        # stat_out.loc[0] = [peak_value1, peak_time1, peak_value2, peak_time2, t_early, p_early, t_late, p_late, start[0], end[0], start[1], end[1] ]
        # stat_out.to_csv('T1 vs D1 cross task decoding.csv')


        # stat_out = pd.DataFrame(index = range(1),columns=['T1 T2 seen', 'T1 T2 seen peak time', 'T1 T2 unseen peak', 'T1 T2 unseen  peak time', 't-stat early', 'p-value early', 't-stat late', 'p-value late', 'start1', 'end1', 'start2', 'end2'])
        # stat_out.loc[0] = [peak_value1, peak_time1, peak_value2, peak_time2, t_early, p_early, t_late, p_late,s_dec[0], e_dec[0], s_dec[1], e_dec[1] ]
        # stat_out.to_csv('T1s in T2 seen vs unseen trials cross task identity decoding.csv')


    def plotLocalizer(self):

        # GAT
        times = np.linspace(-200,900,140)
        vmin, vmax = 0.45, 0.55
        norm = MidpointNormalize(midpoint=1/2.0)

        fig = plt.figure(figsize = (12,5))

        perm=[]

        for bidx, B in enumerate (['T']): 

            perm=[]

            files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'within', 'all', 'identity', 'baseline'], filename = 'class_*-identity-broad.pickle'))
            

            for plt_idx, cnd in enumerate(['digit', 'letter']): 

                ax = plt.subplot(1,2,plt_idx + 1, ylabel = 'Time(ms)', xlabel = 'Time (ms)')

                bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack

                X = np.stack([b[cnd]['standard'] for b in bdm])

                # Compute tresholded data
                X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
                contour = np.zeros(X_thresh.shape, dtype = bool)
                contour[X_thresh != 1/2.0] = True
               
                im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation='none', 
                origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
                vmin = vmin, vmax = vmax)

                # Plot contoures around significant datapoints
                plt.contour(contour, origin = 'lower',linewidths=0.2, colors = ['black'],extent=[times[0],times[-1],times[0],times[-1]])
                plt.yticks([0,300,600, 900])
                plt.xticks([0,300,600, 900]) 
                sns.despine(offset=10, trim = False)
                plt.axhline(y = 0, ls = '--',color='k')
                plt.axvline(x = 0,  ls = '--',color='k')

                plt.ylabel('Time (ms)')
                plt.xlabel('Time (ms)')

                plt.title (['Localizer: digits', 'Localizer: letters'][plt_idx], fontsize = 20)
                
                # # get time of sign. clusters
                # lim=[]
                # tr_c=[]
                # for c in range (0,len(contour)):
                #     if contour[c].any() == True: 
                #         tr_c.append(c) # train time points that have significant test points
                #         lim.append((np.where(contour[c] == True)[0] ) ) # test time

                # time_train=[]
                # time_test=[]
                # for t in range (0, len(tr_c)):
                #     time_train.append(times[tr_c[t]])
                #     time_test.append(times[lim[t]])

                # # find how long a certain classifier generalizes
                # tidx = 82 # 450ms
                # wt = tr_c.index(tidx) 
                # gen  = times[lim[wt]]


                # peak_value, peak_time = PO.decodingpeak(data= X, times = times, area = 'diagonal', peakC = True, lat1 = [100, 350], lat2 = [100, 350])
            # add a colorbar for all figures
            cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.61]) # left, bottom, width, height (range 0 to 1)
            cbar = fig.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])                      
            cbar.set_clim([vmin,vmax])
             # cbar.set_ticks([vmin, 0, vmax])
            cbar.set_ticklabels([str(vmin),'0',str(vmax)])

            # plt.legend(loc = 'best')
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles,('Boost', 'Bounce') )

            sns.despine(offset=0, trim = False)
            # plt.tight_layout()

            plt.savefig(FolderStructure.FolderTracker( filename = 'Localizer_all.pdf'))


    def taskRel(self):

        #

        vmin, vmax = 0.47, 0.65
        norm = MidpointNormalize(midpoint=1/2.0)

        f = plt.figure(figsize = (12,5)) 
        ylim = (0.47,0.65)
        times = np.linspace(-200,900,140)

        # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'within', 'all_collapsed', 'TargetvsDistractor'], filename = 'class_*-new.pickle'))
        # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'within', 'all', 'T', 'TTvsTD'], filename = 'class_*-TD.pickle'))
        # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'within', 'all', 'T', 'TDDTvsTDDD'], filename = 'class_*-TD.pickle'))
        diag1=[]
        for pidx, pos in enumerate (['TvsD_pos1', 'TTvsTD_pos2', 'TDTvsTDD_pos3', 'TDDTvsTDDD_pos4']): 

            files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'within', 'all', 'T', pos], filename = 'class_*-TD.pickle'))
        
            

            for bidx, B in enumerate (['all']): #'T', 'D'

                

                ax = plt.subplot(1,2,1)

                bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack

                X = np.stack([b[B]['standard'] for b in bdm]) 

                #Get diagonals for all subjects
                diagonal = np.stack([np.diag(par) for par in X])
                # diagonal = X

                diag1.append(diagonal.mean(0))

                PO.clusterPlot(diagonal, 1/2.0, p_val = 0.05, times = times, y = 0.48 + pidx * 0.002, color =[ 'blue', 'red','orange', 'pink'][pidx])

                err_t, X_t = bootstrap(diagonal)
                plt.plot(times, X_t, label = B, color = [ 'blue', 'red','orange', 'pink'][pidx])
                plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = [ 'blue', 'red','orange', 'pink'][pidx])
               

                plt.axhline(y = 0.5, ls = '--',color='k')
                plt.axvline(x = 0,  ls = '--',color='k')

                plt.ylim(ylim)
                plt.yticks([ylim[0],1/2.0,ylim[1]])

                plt.xticks([0,300,600,900]) 

                PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')
          
                plt.legend(loc = 'best')
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,('T vs. D on TP1', 'TT vs. TD on TP2', 'TDT vs. TDD on TP3', 'TDDT vs. TDDD on TP4'), bbox_to_anchor=(1.9,1))
                # ax.legend (bbox_to_anchor=(1.1,1.05))

                sns.despine(offset=10, trim = False)

                # plt.title (['arget representations', 'Distractor representations'][bidx], fontsize = 20)
             
                # if len(perm) ==2: 
                # #     PO.clusterPlot(perm[0], perm[1], p_val = 0.05, times = times, y = 0.475 + plt_idx * 0.002, color = 'red')

                #     [p_value1, T_01] = permutationTTest(perm[0], perm[1], 1000) # permutation t test
                #     mask1 = np.where (p_value1 < 0.05)[0]
                #     for p in mask1:
                #         plt.plot(times[p], 1 * 0.475 + bidx * 0.002, 'ro')

            # plt.suptitle('Stimulus-specific representations', fontsize = 26, y=1.08)
            
            # plt.tight_layout()
        # plt.legend(['Boost','Bounce'], loc='upper right', bbox_to_anchor=(1.04,1), ncol=3, fancybox=True, shadow=True)
        # plt.savefig(self.FolderTracker(['bdm','all','cross'], filename = 'cross-task_boost_bounce_T2_T3.pdf'))

        # plt.savefig(FolderStructure.FolderTracker( filename = 'TaskRelevance-all_positions.pdf'))
       

        vmin, vmax = 0.45, 0.55
        norm = MidpointNormalize(midpoint=1/2.0)

        fig = plt.figure(figsize = (12,5))
        ylim = (0.45,0.55)

        times = np.linspace(-200,900,140)
        for plt_idx, cnd in enumerate(['all']): 

            ax = plt.subplot(1,2,plt_idx + 1, ylabel = 'Time(ms)', xlabel = 'Time (ms)')

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
        
            X = np.stack([b[cnd]['standard'] for b in bdm])

            # Compute tresholded data
            X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
            contour = np.zeros(X_thresh.shape, dtype = bool)
            contour[X_thresh != 1/2.0] = True 
           
            im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation='none', 
            origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
            vmin = vmin, vmax = vmax)

            # Plot contoures around significant datapoints
            plt.contour(contour, origin = 'lower',linewidths=0.2, colors = ['black'],extent=[times[0],times[-1],times[0],times[-1]])
            plt.yticks([0,300,600, 900])
            plt.xticks([0,300,600, 900]) 
            sns.despine(offset=10, trim = False)
            plt.axhline(y = 0, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylabel('Time (ms)')
            plt.xlabel('Time (ms)')

            plt.title (['Targets vs Distractors'][plt_idx], fontsize = 20)

            # add a colorbar for all figures
            cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.61]) # left, bottom, width, height (range 0 to 1)
            cbar = fig.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])                      
            cbar.set_clim([vmin,vmax])
             # cbar.set_ticks([vmin, 0, vmax])
            cbar.set_ticklabels([str(vmin),'0',str(vmax)])

            # plt.legend(loc = 'best')
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles,('Boost', 'Bounce') )

            sns.despine(offset=0, trim = False)
            # plt.tight_layout()
            plt.savefig(FolderStructure.FolderTracker( filename = 'TaskRelevance-{}-GAT.pdf'.format(pos))) 



    def plotOneEvent(self):

        #

        f = plt.figure(figsize = (12,5))
        ylim = (0.45,0.55)
        times = np.linspace(-200,900,140)

        diag1=[]

        s_dec = []
        e_dec= []

        itm = 'T2'
        # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'all', 'noOrderRev', '{}s_on_T2seen-unseen_ABcond'.format(itm)], filename = 'class_*.pickle'))
        # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross',  'StimType', 'Shifted', 'T2seen-unseen', 'D1'], filename = 'class_*.pickle'))
        files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'all', '{}s_on_T2seen-unseen_ABcond'.format(itm)], filename = 'class_*.pickle'))
        

        for bidx, B in enumerate (['T2_seen', 'T2_unseen']): 
            
            ax = plt.subplot(1,2,1)

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack

            X = np.stack([b[B]['standard'] for b in bdm]) 

            #Get diagonals for all subjects
    #               diagonal = np.stack([np.diag(par) for par in X])
            diagonal = np.stack([np.diag(par) for par in X])

            diag1.append(diagonal)

            s, e = PO.clusterPlot(diagonal, 1/2.0, p_val = 0.05, times = times, y = 0.49 + bidx * 0.002, color = [ 'blue', 'red'][bidx])
            s_dec.append(s)
            e_dec.append(e)

            err_t, X_t = bootstrap(diagonal)
            plt.plot(times, X_t, label = B, color =  [ 'blue', 'red'][bidx])
            plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = [ 'blue', 'red'][bidx])
        

            plt.axhline(y = 0.5, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylim(ylim)
            plt.yticks([ylim[0],1/2.0,ylim[1]])

            plt.xticks([0,300,600,900]) 

            PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'Accuracy')

           
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,('T2_seen', 'T2_unseen') )
            plt.legend( bbox_to_anchor=(1.9, 1), loc='upper right', frameon=False)

            sns.despine(offset=10, trim = False)

            plt.title ('{}s on T2 seen versus unseen'.format(itm), fontsize = 20)

            # if len(perm) == 2: 
            # #     PO.clusterPlot(perm[0], perm[1], p_val = 0.05, times = times, y = 0.475 + plt_idx * 0.002, color = 'red')
            #     [p_value1, T_01] = permutationTTest(perm[0], perm[1], 1000) # permutation t test
            #     mask1 = np.where (p_value1 < 0.05)[0]
            #     for p in mask1:
            #         plt.plot(times[p], 1 * 0.48 + bidx * 0.002, 'ro')

        # plt.tight_layout()
        PO.stats_out(diag1, times, start = s_dec, end=e_dec, lat1=[150, 250], lat2 =[300, 600]) # computes paired-sample t-test in two diagonal time windows and wirites out a csv file
        
        plt.savefig(FolderStructure.FolderTracker( filename = '{}s_on_T2seen-unseen-ABconditions.pdf'.format(itm)))
        

        fig = plt.figure(figsize = (12,5))
        vmin, vmax = (0.45,0.55)
        times = np.linspace(-200,900,140)


        for plt_idx, cnd in enumerate(['T2_seen', 'T2_unseen']):        

            ax = plt.subplot(1,2, plt_idx  + 1, ylabel = 'Time(ms)', xlabel = 'Time (ms)')

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
        
            X = np.stack([b[cnd]['standard'] for b in bdm])

            # Compute tresholded data
            X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
            contour = np.zeros(X_thresh.shape, dtype = bool)
            contour[X_thresh != 1/2.0] = True 
           
            im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation='none', 
            origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
            vmin = vmin, vmax = vmax)

            # Plot contoures around significant datapoints
            plt.contour(contour, origin = 'lower',linewidths=0.2, colors = ['black'],extent=[times[0],times[-1],times[0],times[-1]])
            plt.yticks([0,300,600, 900])
            plt.xticks([0,300,600, 900]) 
            sns.despine(offset=10, trim = False)
            plt.axhline(y = 0, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylabel('Time (ms)')
            plt.xlabel('Time (ms)')

            plt.title (['{} on T2 AB seen'.format(itm), '{} on T2 AB unseen'.format(itm)][plt_idx], fontsize = 20)

            # add a colorbar for all figures
            cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.61]) # left, bottom, width, height (range 0 to 1)
            cbar = fig.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])                      
            cbar.set_clim([vmin,vmax])
             # cbar.set_ticks([vmin, 0, vmax])
            cbar.set_ticklabels([str(vmin),'0',str(vmax)])

            sns.despine(offset=0, trim = False)
          
            plt.savefig(FolderStructure.FolderTracker( filename = '{}s_on_T2seen-unseen-ABconditions-GAT-Accuracy-DTdecoding.pdf'.format(itm)))



    def T2seenUnseen(self):

        # Not in the manuscript, but stats are reported in the paper - WITHIN DECODING

        f = plt.figure(figsize = (12,5))
        ylim = (0.47,0.60)
        times = np.linspace(-200,900,140)

        diag1=[]
        s_dec=[]
        e_dec=[]
        files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'within', 'all_collapsed', 'T2-identity', 'AB cond'], filename = 'class_*.pickle'))

        for bidx, B in enumerate (['seen', 'unseen']): #'AB_seen', 'AB_unseen'

            ax = plt.subplot(1,2,1)

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
            
            X = np.stack([b[B]['standard'] for b in bdm]) 

            #Get diagonals for all subjects
            diagonal = np.stack([np.diag(par) for par in X])
            # diagonal = X

            diag1.append(diagonal)

            s,e = PO.clusterPlot(diagonal, 1/2.0, p_val = 0.05, times = times, y = 0.49 + bidx * 0.002, color =[ 'green', 'orange'][bidx])
            s_dec.append(s)
            e_dec.append(e)

            err_t, X_t = bootstrap(diagonal)
            plt.plot(times, X_t, label = B, color =  [ 'green', 'orange'][bidx])
            plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = [ 'green', 'orange'][bidx])
           

            plt.axhline(y = 0.5, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylim(ylim)
            plt.yticks([ylim[0],1/2.0,ylim[1]])

            plt.xticks([0,300,600,900]) 

            PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,('T2 seen', 'T2 unseen') )
            plt.legend(loc = 'best')
            sns.despine(offset=10, trim = False)


            # if len(perm) == 2: 
            # #     PO.clusterPlot(perm[0], perm[1], p_val = 0.05, times = times, y = 0.475 + plt_idx * 0.002, color = 'red')
            #     [p_value1, T_01] = permutationTTest(perm[0], perm[1], 1000) # permutation t test
            #     mask1 = np.where (p_value1 < 0.05)[0]
            #     for p in mask1:
            #         plt.plot(times[p], 1 * 0.48 + bidx * 0.002, 'ro')

        # plt.tight_layout()
        PO.stats_out(diag1, times, start = s_dec, end=e_dec, lat1 = [150, 250], lat2 = [300, 600]) # computes paired-sample t-test in two diagonal time windows and wirites out a csv file
        plt.savefig(FolderStructure.FolderTracker( filename = 'T2seen-unseen-identity-withindecoding-ABconditions.pdf'))
        


        fig = plt.figure(figsize = (12,5))
        vmin, vmax = (0.45,0.55)
        times = np.linspace(-200,900,140)


        for plt_idx, cnd in enumerate(['seen', 'unseen']):       # 'seen', 'unseen'
        # for plt_idx, cnd in enumerate(['all']):       # 'seen', 'unseen'

            ax = plt.subplot(1,2, plt_idx  + 1, ylabel = 'Training time (ms)', xlabel = 'Testing time (ms)')

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
        
            X = np.stack([b[cnd]['standard'] for b in bdm])

            # Compute tresholded data
            X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
            contour = np.zeros(X_thresh.shape, dtype = bool)
            contour[X_thresh != 1/2.0] = True 
           
            im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation='none', 
            origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
            vmin = vmin, vmax = vmax)

            # Plot contoures around significant datapoints
            plt.contour(contour, origin = 'lower',linewidths=0.2, colors = ['black'],extent=[times[0],times[-1],times[0],times[-1]])
            plt.yticks([0,300,600, 900])
            plt.xticks([0,300,600, 900]) 
            sns.despine(offset=10, trim = False)
            plt.axhline(y = 0, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylabel('Training time (ms)')
            plt.xlabel('Testing time (ms)')

            # plt.title (['T2 seen', 'T2 unseen'][plt_idx], fontsize = 20)

            # add a colorbar for all figures
            cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.61]) # left, bottom, width, height (range 0 to 1)
            cbar = fig.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])                      
            cbar.set_clim([vmin,vmax])
             # cbar.set_ticks([vmin, 0, vmax])
            cbar.set_ticklabels([str(vmin),'0',str(vmax)])

            sns.despine(offset=0, trim = False)
            # plt.tight_layout()
            plt.savefig(FolderStructure.FolderTracker( filename = 'T.pdf'))
            # plt.savefig(FolderStructure.FolderTracker( filename = 'T2seen-unseen-identity-withindecoding-GAT-ABconditions.pdf'))


    def D1T1(self):

        #

        f = plt.figure(figsize = (12,5))
        ylim = (0.47,0.55)
        times = np.linspace(-200,900,140)

        diag1=[]
        s_dec=[]
        e_dec=[]

        for bidx, B in enumerate (['T1', 'D1']): # T1, D1; 'T1', 'T2', 'T3'

            # filename = ['class_*-new.pickle', 'class_*.pickle', 'class_*.pickle'][bidx]

            files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'T1detected', 'all'], filename = 'class_*-{}.pickle'.format(B)))
        
            ax = plt.subplot(1,2,1)

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack


            X = np.stack([b['collapsed']['standard'] for b in bdm]) 

            #Get diagonals for all subjects
            diagonal = np.stack([np.diag(par) for par in X])
            # diagonal = X
        
            diag1.append(diagonal)

            s, e = PO.clusterPlot(diagonal, 1/2.0, p_val = 0.05, times = times, y = 0.49 + bidx * 0.002, color =[ 'blue', 'red', 'yellow'][bidx])
            s_dec.append(s)
            e_dec.append(e)
        
            err_t, X_t = bootstrap(diagonal)

            plt.plot(times, X_t, label = B, color =  [ 'blue', 'red', 'yellow'][bidx])
            plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = [ 'blue', 'red', 'yellow'][bidx])
           

            plt.axhline(y = 0.5, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylim(ylim)
            plt.yticks([ylim[0],1/2.0,ylim[1]])

            plt.xticks([0,300,600,900]) 

            PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')
      
            plt.legend(loc = 'best')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,('T1', 'D1') )

            sns.despine(offset=10, trim = False)

            # plt.title ('', fontsize = 20)

            # if len(perm) == 2: 
            # #     PO.clusterPlot(perm[0], perm[1], p_val = 0.05, times = times, y = 0.475 + plt_idx * 0.002, color = 'red')
            #     [p_value1, T_01] = permutationTTest(perm[0], perm[1], 1000) # permutation t test
            #     mask1 = np.where (p_value1 < 0.05)[0]
            #     for p in mask1:
            #         plt.plot(times[p], 1 * 0.48 + bidx * 0.002, 'ro')

        # plt.tight_layout()

        plt.savefig(FolderStructure.FolderTracker( filename = 'T1andD1-crosstask_all.pdf'))
        # s_dec, e_dec=np.stack(s_dec), np.stack(e_dec)
        PO.stats_out(diag1, times, start = s_dec, end=e_dec, lat1 = [150, 250], lat2 = [300, 600]) # computes paired-sample t-test in two diagonal time windows and wirites out a csv file


    def stimuli_GAT(self):

        #

        # GAT
        times = np.linspace(-200,900,140)
        vmin, vmax = 0.45, 0.55
        norm = MidpointNormalize(midpoint=1/2.0)

        fig = plt.figure(figsize = (12,5))


        for plt_idx, chans in enumerate(['all']): 

            files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'T1detected', 'all' ], filename = 'class_*-T1.pickle'))

            ax = plt.subplot(1,2,plt_idx + 1, ylabel = 'Time(ms)', xlabel = 'Time (ms)')

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
   
            X = np.stack([b['collapsed']['standard'] for b in bdm])

            # Compute tresholded data
            X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
            contour = np.zeros(X_thresh.shape, dtype = bool)
            contour[X_thresh != 1/2.0] = True 
           
            im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation='none', 
            origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
            vmin = vmin, vmax = vmax)

            # Plot contoures around significant datapoints
            plt.contour(contour, origin = 'lower',linewidths=0.2, colors = ['black'],extent=[times[0],times[-1],times[0],times[-1]])
            plt.yticks([0,300,600, 900])
            plt.xticks([0,300,600, 900]) 
            sns.despine(offset=10, trim = False)
            plt.axhline(y = 0, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylabel('Time (ms)')
            plt.xlabel('Time (ms)')

            plt.title (['T1s '+ chans][plt_idx], fontsize = 20)

            # add a colorbar for all figures
            cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.61]) # left, bottom, width, height (range 0 to 1)
            cbar = fig.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])                      
            cbar.set_clim([vmin,vmax])
             # cbar.set_ticks([vmin, 0, vmax])
            cbar.set_ticklabels([str(vmin),'0',str(vmax)])

            # get time of sign. clusters
            # lim=[]
            # tr_c=[]
            # for c in range (0,len(contour)):
            #     if contour[c].any() == True: 
            #         tr_c.append(c) # train time points that have significant test points
            #         lim.append((np.where(contour[c] == True)[0] ) ) # test time

            # time_train=[]
            # time_test=[]
            # for t in range (0, len(tr_c)):
            #     time_train.append(times[tr_c[t]])
            #     time_test.append(times[lim[t]])


            # # peak_value, peak_time = PO.decodingpeak(data= X, times = times, area = 'diagonal', peakC = True, lat1 = [100, 350], lat2 = [100, 350])

        sns.despine(offset=0, trim = False)
        # plt.tight_layout()
        plt.savefig(FolderStructure.FolderTracker( filename = 'T1_crosstask_GAT_{}.pdf'.format(chans)))


    def plotT3(self):

        # 

        diag1=[]
        s_dec=[]
        e_dec=[]

        f = plt.figure(figsize = (12,5))
        ylim = (0.47,0.55)
        times = np.linspace(-200,900,140)

        diag1=[]

        files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'cross', 'T3', 'TDTTDvsTTDTD'], filename = 'class_*.pickle'))
        
        for bidx, B in enumerate (['TDTTD', 'TTDTD']): 

            ax = plt.subplot(1,2,1)

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack

            X = np.stack([b[B]['standard'] for b in bdm]) 
  
            #Get diagonals for all subjects
    #               diagonal = np.stack([np.diag(par) for par in X])
            diagonal = np.stack([np.diag(par) for par in X])

            diag1.append(diagonal)

            s, e = PO.clusterPlot(diagonal, 1/2.0, p_val = 0.05, times = times, y = 0.49 + bidx * 0.002, color =[ 'blue', 'red', 'yellow'][bidx])
            s_dec.append(s)
            e_dec.append(e)
        
            err_t, X_t = bootstrap(diagonal)
            plt.plot(times, X_t, label = B, color =  [ 'red', 'orange'][bidx])
            plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = [ 'red', 'orange'][bidx])
           

            plt.axhline(y = 0.5, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylim(ylim)
            plt.yticks([ylim[0],1/2.0,ylim[1]])

            plt.xticks([0,300,600,900]) 

            PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')
      
            plt.legend(loc = 'best')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,('TDTTD', 'TTDTD') )

            sns.despine(offset=10, trim = False)

            plt.title ('T3 in TDTTD vs TTDTD', fontsize = 20)

        #     if len(perm) == 2: 
        #     #     PO.clusterPlot(perm[0], perm[1], p_val = 0.05, times = times, y = 0.475 + plt_idx * 0.002, color = 'red')
        #         [p_value1, T_01] = permutationTTest(perm[0], perm[1], 1000) # permutation t test
        #         mask1 = np.where (p_value1 < 0.05)[0]
        #         for p in mask1:
        #             plt.plot(times[p], 1 * 0.48 + bidx * 0.002, 'ro')

        # plt.tight_layout()
      
        PO.stats_out(diag1, times, start = s_dec, end=e_dec, lat1 = [150, 250], lat2 = [300, 600]) # computes paired-sample t-test in two diagonal time windows and wirites out a csv file
        plt.savefig(FolderStructure.FolderTracker( filename = 'T3 in TDTTDvsTTDTD-diag.pdf'))


        fig = plt.figure(figsize = (12,5))
        vmin, vmax = (0.45,0.55) 
        times = np.linspace(-200,900,140)


        for plt_idx, cnd in enumerate(['TDTTD', 'TTDTD']):        

            ax = plt.subplot(1,2, plt_idx  + 1, ylabel = 'Time(ms)', xlabel = 'Time (ms)')

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
        
            X = np.stack([b[cnd]['standard'] for b in bdm])

            # Compute tresholded data
            X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
            contour = np.zeros(X_thresh.shape, dtype = bool)
            contour[X_thresh != 1/2.0] = True 

                        # get time of sign. clusters
            lim=[]
            tr_c=[]
            for c in range (0,len(contour)):
                if contour[c].any() == True: 
                    tr_c.append(c) # train time points that have significant test points
                    lim.append((np.where(contour[c] == True)[0] ) ) # test time

            time_train=[]
            time_test=[]
            for t in range (0, len(tr_c)):
                time_train.append(times[tr_c[t]])
                time_test.append(times[lim[t]])

            # generalization times !!
            # classifiers'index; specify time in the brackets (e.g. 400ms); specify time range 
            tidx1 = int( np.argmin(abs(times - 720)) ) 
            tidx2 =int( np.argmin(abs(times - 750)) ) 

            wt1 = tr_c.index(tidx1) 
            wt2 = tr_c.index(tidx2) 
            # get the generalization time
            gen_time.append([times[lim[w]] for w in (range(wt1, wt2))])
            print(gen_time)
           
            im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation='none', 
            origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
            vmin = vmin, vmax = vmax)

            # Plot contoures around significant datapoints
            plt.contour(contour, origin = 'lower',linewidths=0.2, colors = ['black'],extent=[times[0],times[-1],times[0],times[-1]])
            plt.yticks([0,300,600, 900])
            plt.xticks([0,300,600, 900]) 
            sns.despine(offset=10, trim = False)
            plt.axhline(y = 0, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylabel('Time (ms)')
            plt.xlabel('Time (ms)')

            plt.title (['T3 in TDTTD', 'T3 in TTDTD'][plt_idx], fontsize = 20)

            # add a colorbar for all figures
            cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.61]) # left, bottom, width, height (range 0 to 1)
            cbar = fig.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])                      
            cbar.set_clim([vmin,vmax])
             # cbar.set_ticks([vmin, 0, vmax])
            cbar.set_ticklabels([str(vmin),'0',str(vmax)])

            sns.despine(offset=0, trim = False)
            # plt.tight_layout()
            plt.savefig(FolderStructure.FolderTracker( filename = 'T3 in TDTTDvsTTDTD-GAT.pdf'))


    def binarydecoding(self):

        #

        f = plt.figure(figsize = (12,5))
        ylim = (0.47,0.70)
        # times = np.linspace(-200,900,140)
        times = np.linspace(-800,900,217)

        # files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'within', 'all_collapsed', 'DDDvsTDD-neutral-bounce-binary'], filename = 'class_*.pickle'))
        files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline', 'bdm', 'within', 'all_collapsed', 'T2correct-identity', 'AB cond'], filename = 'class_*.pickle'))
        for bidx, B in enumerate (['all']): 

            ax = plt.subplot(1,2,1)

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
            
            X = np.stack([b[B]['standard'] for b in bdm]) 

            #Get diagonals for all subjects
            diagonal = np.stack([np.diag(par) for par in X])
    
            PO.clusterPlot(diagonal, 1/2.0, p_val = 0.05, times = times, y = 0.49 + bidx * 0.002, color =[ 'red'][bidx])

            err_t, X_t = bootstrap(diagonal)
            plt.plot(times, X_t, label = B, color =  [ 'red' ][bidx])
            plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = [ 'red' ][bidx])
           

            plt.axhline(y = 0.5, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')

            plt.ylim(ylim)
            plt.yticks([ylim[0],1/2.0,ylim[1]])

            plt.xticks([-600, -300, 0,300,600,900]) 

            PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')

            plt.legend(loc = 'best')
            handles, labels = ax.get_legend_handles_labels()
           
            sns.despine(offset=10, trim = False)

        plt.savefig(FolderStructure.FolderTracker( filename = 'Long_epochs_T2seen-unseen-binary.pdf'))
        

        fig = plt.figure(figsize = (12,5))
        vmin, vmax = (0.45,0.55)
        # times = np.linspace(-200,900,140)
        times = np.linspace(-800,900,217)

        for plt_idx, cnd in enumerate(['all']):       # 'seen', 'unseen'

            ax = plt.subplot(1,2, plt_idx  + 1, ylabel = 'Time(ms)', xlabel = 'Time (ms)')

            bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
        
            X = np.stack([b[cnd]['standard'] for b in bdm])

            # Compute tresholded data
            X_thresh = threshArray(X, 0.5, method = 'ttest', p_value = 0.05)
            contour = np.zeros(X_thresh.shape, dtype = bool)
            contour[X_thresh != 1/2.0] = True 
           
            im = plt.imshow(X.mean(0), norm = norm, cmap = cm.bwr, interpolation='none', 
            origin = 'lower', extent=[times[0],times[-1],times[0],times[-1]], 
            vmin = vmin, vmax = vmax)

            # Plot contoures around significant datapoints
            plt.contour(contour, origin = 'lower',linewidths=0.2, colors = ['black'],extent=[times[0],times[-1],times[0],times[-1]])
            plt.yticks([-600, -300, 0,300,600,900, 0, 300,600, 900])
            plt.xticks([-600, -300, 0,300,600,900, 0, 300,600, 900]) 
            sns.despine(offset=10, trim = False)
            plt.axhline(y = 0, ls = '--',color='k')
            plt.axvline(x = 0,  ls = '--',color='k')
            plt.ylabel('Time (ms)')
            plt.xlabel('Time (ms)')

            # add a colorbar for all figures
            cb_ax = fig.add_axes([0.91, 0.2, 0.01, 0.61]) # left, bottom, width, height (range 0 to 1)
            cbar = fig.colorbar(im, cax=cb_ax, ticks = [vmin, 0.5, vmax])                      
            cbar.set_clim([vmin,vmax])
             # cbar.set_ticks([vmin, 0, vmax])
            cbar.set_ticklabels([str(vmin),'0',str(vmax)])

            sns.despine(offset=0, trim = False)
            # plt.tight_layout()
            # plt.savefig(FolderStructure.FolderTracker( filename = 'DDDvsTDD-neutral-bounce-binary-GAT.pdf'))
            plt.savefig(FolderStructure.FolderTracker( filename ='GAT:Long_epochs_T2seen-unseen-binary.pdf'))


    def showCondTimecourse(self):

        '''
        plots targets and distractors, collapsed over all conditions; diagonal
        '''

        f = plt.figure(figsize = (12,5))
        times = np.linspace(-200,900,140)
        stim =  [['T1', 'T2'], ['D1', 'D2', 'D3'] ]
        colors = [['blue','green', 'purple'], ['red','pink','orange'] ]

        ylim = (0.47,0.55)

        shift1 = [83, 83*2, 83*3, 83*4] #how many positions to shift (for backward shift use minus sign)
        # shift1 = [83*2, 83, 83*3, 83*4]

        for bidx, B in enumerate (['T', 'D']): 
            shift=[]
            ax = plt.subplot(1,2, 1)

            for i, T in enumerate(stim[bidx]):

                files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline','bdm','cross', 'T1detected', 'all'], filename = 'class_*-{}.pickle'.format(T)))

                bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack
             
                X = np.stack([np.diag(b['TTDDD']['standard']) for b in bdm])
             
                if T is not 'T1':
                    if T == 'T2':
                        shift = shift1[0]
                    elif T == 'D1':
                        shift = shift1[1]
                    elif T == 'D2':
                        shift = shift1[2]
                    elif T == 'D3':
                        shift = shift1[3]

                    to_shift = int(np.diff([np.argmin(abs(times - t)) for t in (0,shift)]))
                    #shift only in test times, train times are based on localizer data
                    X= np.roll(X,to_shift, axis = 1)

                PO.clusterPlot(X, 1/2.0, p_val = 0.05, times = times, y = 0.48 + i * 0.002, color = colors[bidx][i])

                err_t, X_t = bootstrap(X)
                plt.plot(times, X_t, label = T, color = colors[bidx][i])
                plt.fill_between(times, X_t + err_t, X_t - err_t, alpha = 0.2, color = colors[bidx][i])
                
                plt.title('TTDDD', fontsize = 20)
             
                PO.beautifyPlot(xlabel = 'Time (ms)', ylabel = 'AUC')

                plt.ylim(ylim)
                plt.yticks([ylim[0],1/2.0,ylim[1]])

                plt.xticks([0,300, 600, 900]) 

                plt.axhline(y = 0.5, ls = '--',color='k')
                plt.axvline(x = 0,  ls = '--',color='k')
                plt.axvline(x = 83,  ls = '--',color='k')
                plt.axvline(x = 83*2,  ls = '--',color='k')
                plt.axvline(x = 83*3,  ls = '--',color='k')
                plt.axvline(x = 83*4,  ls = '--',color='k')

                plt.legend(loc = 'upper right', frameon=None)
                sns.despine(offset=10, trim = False)

        # plt.tight_layout()
        plt.show()
        plt.savefig(FolderStructure.FolderTracker(filename = 'Stimuli_per_temp_positions_TTDDD.pdf'))
 
    
    def plotANOVAcond(self):

        # Plots behavior, early and late decoding scores in conditions which were entered in 3 rmANOVAs

        file = FolderStructure.FolderTracker(extension = ['BEH' ,'results'], filename = 'plot_data_behavior_clean.csv') 
        DF = pd.read_csv(file)
        df_beh = DF.drop('subjects',axis=1)
        beh=[]
        df=[]
        #reorder behavior 
        df.append(DF.filter(like='T1'))
        df.append(DF.filter(like='T2'))
        df.append(DF.filter(like='T3'))
     

        beh = pd.concat(df, axis=1).T
        #remove variables with no data
        beh=beh.loc[(beh!=0).any(axis=1)]
        # Calculate SEM for errorbars
        yerr = beh.sem(axis = 1, skipna = False) 


        beh_cond = beh.mean(1) # mean over subjects so you get averages per TP across conditions
        beh_plot = np.zeros((5,8))
        beh_plot[0,:] = beh_cond[:8]
        beh_plot[1,:7] = beh_cond[8:15]
        beh_plot[2,4:7] = beh_cond[15:]
        BEH = pd.DataFrame (beh_plot, columns =['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD'] )

        # restructure the data for plotting
        BEH['DDTDD'][4] = BEH['DDTDD'][0]
        BEH['DDTDD'][0,1,2,3]=np.nan 

        BEH['T..DDDT'][4]=BEH['T..DDDT'][1]
        BEH['T..DDDT'][1,2,3]=np.nan   

        BEH['TDTDD'][2]=BEH['TDTDD'][1]
        BEH['TDTDD'][1,3,4]=np.nan         

        BEH['TDDTD'][3]=BEH['TDDTD'][1]
        BEH['TDDTD'][1,2,4]=np.nan 

        BEH['TTDTD'][3]=BEH['TTDTD'][2]
        BEH['TTDTD'][2,4]=np.nan 

        BEH['TDTTD'][3]=BEH['TDTTD'][2]
        BEH['TDTTD'][2]=BEH['TDTTD'][1]
        BEH['TDTTD'][1,4]=np.nan 
        BEH['TTDDD'][2,3,4] = np.nan
        BEH['TTTDD'][3,4] = np.nan


        yerrdf = np.zeros((5,8))
        yerrdf[0,:] = yerr[:8]
        yerrdf[1,:7] = yerr[8:15]
        yerrdf[2,4:7] = yerr[15:]
        rois_error_beh = pd.DataFrame (yerrdf, columns =['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD'] )
        # restructure the data for plotting
        rois_error_beh['DDTDD'][4] = rois_error_beh['DDTDD'][0]
        rois_error_beh['DDTDD'][0,1,2,3]=np.nan 

        rois_error_beh['T..DDDT'][4]=rois_error_beh['T..DDDT'][1]
        rois_error_beh['T..DDDT'][1,2,3]=np.nan   

        rois_error_beh['TDTDD'][2]=rois_error_beh['TDTDD'][1]
        rois_error_beh['TDTDD'][1,3,4]=np.nan         

        rois_error_beh['TDDTD'][3]=rois_error_beh['TDDTD'][1]
        rois_error_beh['TDDTD'][1,2,4]=np.nan 

        rois_error_beh['TTDTD'][3]=rois_error_beh['TTDTD'][2]
        rois_error_beh['TTDTD'][2,4]=np.nan 

        rois_error_beh['TDTTD'][3]=rois_error_beh['TDTTD'][2]
        rois_error_beh['TDTTD'][2]=rois_error_beh['TDTTD'][1]
        rois_error_beh['TDTTD'][1,4]=np.nan 
        rois_error_beh['TTDDD'][2,3,4] = np.nan
        rois_error_beh['TTTDD'][3,4] = np.nan


        #### get decoding scores ####

        conditions = ['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD']
        subjects =[1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35]

        lat = [[150, 250], [300, 600]]

        rois_dict={}
        rois_error_neur_dict={}

        for win, w in enumerate(['early', 'late']):
            times = np.linspace(-200,900,140)
            tx1=[]
            for x, l in enumerate(lat[win]):
                tx1.append(int(np.argmin(abs(times - l)))) 

            rois=[]
            neurSubj=[]
            rois = pd.DataFrame(index = range(5),columns=conditions) # these are subject averages
            neurSubj = pd.DataFrame(index = range(18),columns=range(32)) # 18 conditions and subjects; only targets

            for i, T in enumerate(['T1','T2','T3']):
            
                files = glob.glob(FolderStructure.FolderTracker(['pre-T1-baseline','bdm','cross','T1detected', 'all'], filename = 'class_*-{}.pickle'.format(T)))

                if T == 'D3':
                    conditions = ['T..DDDT','TTDDD','TDTDD','TDDTD','DDTDD']

                for plt_idx, cnd in enumerate(conditions): 
                    bdm = np.stack([pickle.load(open(file,'rb')) for file in files]) # open all and stack

                    if cnd not in bdm[0].keys():
                        continue   

                    X = np.stack([b[cnd]['standard'] for b in bdm])

                    diagonal = np.stack([np.diag(par) for par in X]).mean(0) # average over subjects
                    dt_range = np.mean(diagonal[range(tx1[0],tx1[1]+1)])
                    rois[cnd][i]  = dt_range # average over the entire time window

                    #Get diagonals - PER SUBJECT
                    diagonalSubj = np.stack([np.diag(par) for par in X])
                    peakRangeSubj = np.stack([s[range(tx1[0],tx1[1]+1)] for s in diagonalSubj] ).mean(1) # mean for each subject separately


                    # store the peak decoding value per subjects and per target in each condition
                    for x_idx, x in enumerate (X): # subjects # SPEARMAN
                        if T == 'T2': # set the plt_idx for indexing the data frame neur
                            neurSubj[x_idx][plt_idx+8] = peakRangeSubj[x_idx]
                        elif T == 'T3': 
                            neurSubj[x_idx][plt_idx+11] = peakRangeSubj[x_idx] # + 11 because plt_idx will be 4 when there is a third target
                        elif T == 'T1':
                            neurSubj[x_idx][plt_idx] = peakRangeSubj[x_idx]


            # restructure the data for plotting
            rois['DDTDD'][4] = rois['DDTDD'][0]
            rois['DDTDD'][0]=np.nan 

            rois['T..DDDT'][4]=rois['T..DDDT'][1]
            rois['T..DDDT'][1]=np.nan   

            rois['TDTDD'][2]=rois['TDTDD'][1]
            rois['TDTDD'][1]=np.nan         

            rois['TDDTD'][3]=rois['TDDTD'][1]
            rois['TDDTD'][1]=np.nan 

            rois['TTDTD'][3]=rois['TTDTD'][2]
            rois['TTDTD'][2]=np.nan 

            rois['TDTTD'][3]=rois['TDTTD'][2]
            rois['TDTTD'][2]=rois['TDTTD'][1]
            rois['TDTTD'][1]=np.nan 


            # Calculate SEM for errorbars
            yerr = neurSubj.sem(axis = 1, skipna = False) 
            yerrdf = np.zeros((5,8))
            yerrdf[0,:] = yerr[:8]
            yerrdf[1,:7] = yerr[8:15]
            yerrdf[2,4:7] = yerr[15:]
            rois_error_neur=[]
            rois_error_neur = pd.DataFrame (yerrdf, columns =['T..DDDT','TTDDD','TDTDD','TDDTD','TTTDD','TTDTD','TDTTD','DDTDD'] )

            # restructure the data for plotting
            rois_error_neur['DDTDD'][4] = rois_error_neur['DDTDD'][0]
            rois_error_neur['DDTDD'][0, 1,2,3]=np.nan 

            rois_error_neur['T..DDDT'][4]=rois_error_neur['T..DDDT'][1]
            rois_error_neur['T..DDDT'][1,2,3]=np.nan   

            rois_error_neur['TDTDD'][2]=rois_error_neur['TDTDD'][1]
            rois_error_neur['TDTDD'][1,3,4]=np.nan         

            rois_error_neur['TDDTD'][3]=rois_error_neur['TDDTD'][1]
            rois_error_neur['TDDTD'][1,2,4]=np.nan 

            rois_error_neur['TTDTD'][3]=rois_error_neur['TTDTD'][2]
            rois_error_neur['TTDTD'][2,4]=np.nan 

            rois_error_neur['TDTTD'][3]=rois_error_neur['TDTTD'][2]
            rois_error_neur['TDTTD'][2]=rois_error_neur['TDTTD'][1]
            rois_error_neur['TDTTD'][1,4]=np.nan 
            rois_error_neur['TTDDD'][2,3,4] = np.nan
            rois_error_neur['TTTDD'][3,4] = np.nan

            if win == 0:
                rois_dict['early'] = rois
                rois_error_neur_dict['early'] = rois_error_neur  
            else:
                rois_dict['late'] = rois
                rois_error_neur_dict['late'] = rois_error_neur  


        cond = [['TTDDD','TDTDD','TDDTD','T..DDDT'], ['TTDDD','TDTDD','TTTDD'], ['TDTDD','TDDTD','TDTTD']]
        marker = [['*','o', '*','*' ], ['*','o','>'],['o','*', '>']]
        color = [['r', 'g', 'orchid','k'], ['r', 'g', 'c'], [ 'g','orchid', 'y' ]]

        # marker = ['*', '*', 'o', '*','>', '>', '>', 'o'] 
        # color = ['k', 'r', 'g', 'orchid', 'c', 'b', 'y', 'k']

    

        for plot in ['beh','early','late']:
                    
            if plot == 'beh':
                data_mat = BEH
                error_mat = rois_error_beh
                ylabel = 'Accuracy (%)'
                ylim =(0,100)
                plt.yticks([ylim[0],20,40,60,80,ylim[1]])  
            elif plot == 'early':
                data_mat = rois_dict['early']
                error_mat = rois_error_neur_dict['early']
                ylabel = 'AUC'
                ylim = (0.49,0.53)
                plt.yticks([0.49,0.5,0.51,0.52,0.53])  
            else:
                data_mat = rois_dict['late']
                error_mat = rois_error_neur_dict['late']
                ylabel = 'AUC'
                ylim = (0.49,0.53)
                plt.yticks([ylim[0],1/2.0,ylim[1]])  
            
            
            for c in range(len(cond)):
                f = plt.figure(figsize = (12,5))
                ax = plt.subplot(1,2,1, title = 'Behavior/decoding accuracy', ylabel = ylabel, xlabel = 'Temporal positions')
                for n, cnd in enumerate (cond[c]):
                    # plt.plot(np.arange(0.25,5,1), rois[cnd], marker[i],  linewidth=3, markersize = 14)
                    plt.errorbar(np.arange(0.25,5,1), data_mat[cnd], yerr=error_mat[cnd], uplims=False, lolims=False, marker = marker[c][n], color = color[c][n], linewidth = 3, markersize = 12)
                    xlim = (-0.5, 5.5)
                    plt.xlim(xlim)
                    plt.xticks(np.arange(0.25,5,1), ['TP1', 'TP2', 'TP3', 'TP4', 'Late TP']) 
                    plt.ylim(ylim)
                    if plot is not 'beh':
                        # plt.title('Decoding', fontsize =20 )
                        plt.yticks([0.49,0.5,0.51,0.52,0.53])
                    # else:
                        # plt.title('Behavior', fontsize =20)
                sns.despine(offset=10, trim = False)
                plt.legend(bbox_to_anchor=(1.9, 1), loc='upper right', frameon=False)
                plt.tight_layout()
                plt.show()
                plt.savefig(FolderStructure.FolderTracker(filename = plot + str(c) + '.pdf'))
            

    
PO = AB_R_Plots()

# PO.plotStimTemporalOrder() #Figure 3A

# PO.showCondTimecourse () #Figure 3B 

# PO.BoostBounceNEW() #Figure 6

# PO.plotERPs(stim = 'T2') #Figure 5A

# PO.plotANOVAcond() #Figure 7

# PO.plotLocalizer() #Figure 2A

# PO.taskRel() #Figure 3C

# PO.plotOneEvent() #Figure 4

# PO.T2seenUnseen() # 

# PO.D1T1() #Figure 2D

# PO.plotT3() #Figure 6A

#PO.binarydecoding() #Figure 5B

# PO.stimuli_GAT() # Figure 2B and 2C

# PO.plotBehavior2() # Figure 1B
