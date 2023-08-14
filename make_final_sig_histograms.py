#!/usr/bin/env python
# coding: utf-8

# In[7]:

import tqdm
import sys
import os
#sys.path.insert(0, os.path.abspath('/eos/user/b/bmaier/.local/lib/python3.9/site-packages/'))
import h5py
import numpy as np
import awkward as ak
import pathlib
import pandas as pd
#print(ak.__version__)
#import mplhep as hep
import json
#plt.style.use(hep.style.CMS)
import subprocess
import case_analysis.util.sample_names as sana
#import ROOT
#from ROOT import TCanvas, TPad, TFormula, TF1, TPaveLabel, TH1F, TFile
#from ROOT import gROOT, gBenchmark


# In[8]:


from lmfit.model import load_model


# In[9]:

def get_loss_indices(branch_names):
    j1_reco = np.argwhere(branch_names=='j1RecoLoss')[0,0]
    j1_kl = np.argwhere(branch_names=='j1KlLoss')[0,0]
    j2_reco = np.argwhere(branch_names=='j2RecoLoss')[0,0]
    j2_kl = np.argwhere(branch_names=='j2KlLoss')[0,0]
    loss_dict = {'j1KlLoss':j1_kl,'j1RecoLoss':j1_reco,'j2KlLoss':j2_kl,'j2RecoLoss':j2_reco}
    return loss_dict
    
def inv_quantile_str(quantile):
    inv_quant = round((1.-quantile),2)
    return 'q{:02}'.format(int(inv_quant*100))

def write_log(dir,comments):
    with open(os.path.join(dir,'log.txt'), 'w') as f:
        f.write(f'QR attempt = {comments[0]}'+'\n')
        for l in comments[1:]:
            f.write(l+'\n')

##### Define directory paths


#signal_name = 'WpToBpT_Wp3000_Bp400_Top170_ZbtReco'
signals=sana.all_samples
signal_injections = [0.]
folds = 20
qcd_sample = 'qcdSROzData'
run_n = 141098
##### Define directory paths
case_qr_results = f'/storage/9/abal/CASE/QR_results/events/run_{run_n}'
case_qr_models = '/work/abal/CASE/QR_models'
case_reco_dir=f'/storage/9/abal/CASE/VAE_results/events/run_{run_n}'
####### FOR LOGGING PURPOSES ONLY ##########
nAttempt=9
bin_low = 1460 
bin_high = 6800
#####
for signal_name in signals:
    print(f'Analysing data with 0 pb-1 of signal {signal_name}')
    
    quantiles = ['70', '50', '30', '10', '05','01']
    all_mjj = []
    all_loss = []
    all_sel_q70 = []
    all_sel_q50 = []
    all_sel_q30 = []
    all_sel_q10 = []
    all_sel_q05 = []
    all_sel_q01 = []
    all_event_features = []
    all_event_feature_names_reversed = ['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99']

    sig_mjj = []
    sig_loss = []
    sig_sel_q70 = []
    sig_sel_q50 = []
    sig_sel_q30 = []
    sig_sel_q10 = []
    sig_sel_q05 = []
    sig_sel_q01 = []
    sig_event_features = []
    sig_event_feature_names_reversed = ['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99']

    #print(f"Fold {k}")
    qrs = []
    sig_qrs = []
    for q in quantiles:
        sig_tmpdf = pd.read_csv(f'{case_qr_models}/run_{run_n}/models_lmfit_csv/unblind_data_SR/{nAttempt}/sig_lmfit_modelresult_quant_q{q}.csv')
        sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
        sig_qrs.append(sig_p)
        #p = np.poly1d(tmpdf['par'].values.tolist()[::-1])
        #p = np.poly1d(tmpdf['par'].values.tolist())

        #print("Quantile", q)
        #print(tmpdf['par'].values.tolist()[::-1])
        #print(p(1200))
        #qrs.append(p)
        
    sig_filepath=os.path.join(case_reco_dir,signal_name.replace('Reco','_RECO'),'nominal',signal_name.replace('Reco','')+'.h5')
    with h5py.File(sig_filepath, "r") as sig_f:
        branch_names = sig_f['eventFeatureNames'][()].astype(str)
        #print(branch_names)
        features = sig_f['eventFeatures'][()]
        mask = (features[:,0]<9000)&(features[:,0]>1460.) # mjj>1460.
        features = features[mask]
        mjj = np.asarray(features[:,0])
        loss_indices = get_loss_indices(branch_names)

        loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

        for j,q in enumerate(quantiles):
            if q == '30':
                if len(sig_sel_q30) == 0:
                    sig_sel_q30 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                else:
                    sig_sel_q30 = np.concatenate((sig_sel_q30,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
            if q == '70':
                if len(sig_sel_q70) == 0:
                    sig_sel_q70 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                else:
                    sig_sel_q70 = np.concatenate((sig_sel_q70,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
            if q == '50':
                if len(sig_sel_q50) == 0:
                    sig_sel_q50 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                else:
                    sig_sel_q50 = np.concatenate((sig_sel_q50,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
            if q == '10':
                if len(sig_sel_q10) == 0:
                    sig_sel_q10 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                else:
                    sig_sel_q10 = np.concatenate((sig_sel_q10,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
            if q == '05':
                if len(sig_sel_q05) == 0:
                    sig_sel_q05 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                else:
                    sig_sel_q05 = np.concatenate((sig_sel_q05,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
            if q == '01':
                if len(sig_sel_q01) == 0:
                    sig_sel_q01 = np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)
                else:
                    sig_sel_q01 = np.concatenate((sig_sel_q01,np.expand_dims((loss > sig_qrs[j](mjj)),axis=-1)))
        if len(sig_loss) == 0:
            sig_loss = np.expand_dims(loss,axis=-1)
            sig_mjj = np.expand_dims(mjj,axis=-1)
        else:
            sig_loss = np.concatenate((sig_loss,np.expand_dims(loss,axis=-1)))
            sig_mjj = np.concatenate((sig_mjj,np.expand_dims(mjj,axis=-1)))



    sig_event_features = np.concatenate((sig_mjj,sig_loss,sig_sel_q70,sig_sel_q50,sig_sel_q30,sig_sel_q10,sig_sel_q05,sig_sel_q01),axis=-1)
    
    case_qr_datasets=f'/ceph/abal/CASE/QR_datasets/run_{run_n}/unblinded_SRData/{nAttempt}'
    pathlib.Path(case_qr_datasets).mkdir(parents=True,exist_ok=True)
    print(f'Output datasets to be used for fitting are located at {case_qr_datasets}')
    outfilename = 'bkg'
    dt = h5py.special_dtype(vlen=str)

    sig_outfilename = 'signal_%s'%(signal_name)
    sig_hf = h5py.File(os.path.join(case_qr_datasets,f'{sig_outfilename}.h5'), 'w')
    sig_hf.create_dataset('mjj', data=np.array(sig_mjj))
    sig_hf.create_dataset('loss', data=np.array(sig_loss))
    sig_hf.create_dataset('sel_q30', data=np.array(sig_sel_q70))
    sig_hf.create_dataset('sel_q50', data=np.array(sig_sel_q50))
    sig_hf.create_dataset('sel_q70', data=np.array(sig_sel_q30))
    sig_hf.create_dataset('sel_q90', data=np.array(sig_sel_q10))
    sig_hf.create_dataset('sel_q95', data=np.array(sig_sel_q05))
    sig_hf.create_dataset('sel_q99', data=np.array(sig_sel_q01))
    sig_hf.create_dataset('eventFeatures', data=np.array(sig_event_features))
    sig_hf.create_dataset('eventFeatureNames', data=np.array(sig_event_feature_names_reversed,dtype=dt))
    sig_hf.close()
        
    


