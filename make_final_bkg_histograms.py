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

nAttempt=2
signal_name = 'grav_1p5_narrowReco'
signal_injections = [0.]
folds = 1
qcd_sample = 'delphes_bkg'
run_n = 113
##### Define directory paths
case_qr_results = f'/storage/9/abal/CASE/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/{nAttempt}'
case_qr_models = f'/work/abal/CASE/QR_models/run_{run_n}/delphes_models_lmfit_csv/{nAttempt}'

####### FOR LOGGING PURPOSES ONLY ##########

bin_low = 1460 
bin_high = 6800
#####

for siginj in signal_injections:
    print(f'Analysing data with {siginj} pb-1 of signal {signal_name}$')
    
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

    data_mjj = []
    data_loss = []
    data_sel_q70 = []
    data_sel_q50 = []
    data_sel_q30 = []
    data_sel_q10 = []
    data_sel_q05 = []
    data_sel_q01 = []
    data_event_features = []
    data_event_feature_names_reversed = ['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99']

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

    inj_mjj = []
    inj_loss = []
    inj_sel_q70 = []
    inj_sel_q50 = []
    inj_sel_q30 = []
    inj_sel_q10 = []
    inj_sel_q05 = []
    inj_sel_q01 = []
    inj_event_features = []
    inj_event_feature_names_reversed = ['mJJ','loss','sel_q30','sel_q50','sel_q70','sel_q90','sel_q95','sel_q99']

    for k in tqdm.tqdm(range(folds)):
        #print(f"Fold {k}")
        qrs = []
        sig_qrs = []
        for q in quantiles:
            if siginj == 0.:
                tmpdf = pd.read_csv(f'{case_qr_models}/delphes_bkg/bkg_lmfit_modelresult_fold_{k}_quant_q{q}.csv')
                sig_tmpdf = pd.read_csv(f'{case_qr_models}/delphes_bkg/sig_lmfit_modelresult_quant_q{q}.csv')
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            else:
                # Note that this section is redundant, should never be executed for unblinding since we don't do any injection tests here. 
                tmpdf = pd.read_csv(f'{case_qr_models}/{signal_name}_{str(siginj)}/bkg+inj_lmfit_modelresult_fold_{k}_quant_q{q}.csv')
                sig_tmpdf = pd.read_csv(f'{case_qr_models}/{signal_name}_{str(siginj)}/sig_lmfit_modelresult_quant_q{q}.csv')
                sig_p = np.poly1d(sig_tmpdf['par'].values.tolist()[::-1])
                sig_qrs.append(sig_p)
            p = np.poly1d(tmpdf['par'].values.tolist()[::-1])
            #p = np.poly1d(tmpdf['par'].values.tolist())

            #print("Quantile", q)
            #print(tmpdf['par'].values.tolist()[::-1])
            #print(p(1200))
            qrs.append(p)
        
        bkg_filename=os.path.join(case_qr_results,f'{qcd_sample}_fold_{k}.h5')
        if folds==1:
            bkg_filename=os.path.join(case_qr_results,f'{qcd_sample}.h5')
        with h5py.File(bkg_filename, "r") as bkg_f:
            branch_names = bkg_f['eventFeatureNames'][()].astype(str)
            #print(branch_names)
            features = bkg_f['eventFeatures'][()]
            mask = features[:,0]<9000
            features = features[mask]
            mjj = np.asarray(features[:,0])
            loss_indices = get_loss_indices(branch_names)
            loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

            for j,q in enumerate(quantiles):
                if q == '70':
                    if len(all_sel_q70) == 0:
                        all_sel_q70 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q70 = np.concatenate((all_sel_q70,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '50':
                    if len(all_sel_q50) == 0:
                        all_sel_q50 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q50 = np.concatenate((all_sel_q50,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '30':
                    if len(all_sel_q30) == 0:
                        all_sel_q30 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q30 = np.concatenate((all_sel_q30,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '10':
                    if len(all_sel_q10) == 0:
                        all_sel_q10 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q10 = np.concatenate((all_sel_q10,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '05':
                    if len(all_sel_q05) == 0:
                        all_sel_q05 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q05 = np.concatenate((all_sel_q05,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if q == '01':
                    if len(all_sel_q01) == 0:
                        all_sel_q01 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                    else:
                        all_sel_q01 = np.concatenate((all_sel_q01,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
            if len(all_loss) == 0:
                all_loss = np.expand_dims(loss,axis=-1)
                all_mjj = np.expand_dims(mjj,axis=-1)
            else:
                all_loss = np.concatenate((all_loss,np.expand_dims(loss,axis=-1)))
                all_mjj = np.concatenate((all_mjj,np.expand_dims(mjj,axis=-1)))

        if k == 0:
            with h5py.File(f"{case_qr_results}/{signal_name}.h5", "r") as sig_f:
                branch_names = sig_f['eventFeatureNames'][()].astype(str)
                #print(branch_names)
                features = sig_f['eventFeatures'][()]
                mask = features[:,0]<9000
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

        # If signal is injected, then apply the QR to the so-called data, which is actually bkg+injected signal
        if siginj != 0.:
            data_filename=os.path.join(case_qr_results,f"delphes_bkg+inj_{signal_name}_{siginj}_fold_{k}.h5")
            if folds==1:
                data_filename=os.path.join(case_qr_results,f"delphes_bkg+inj_{signal_name}_{siginj}.h5")    
            with h5py.File(data_filename, "r") as data_f:
                branch_names = data_f['eventFeatureNames'][()].astype(str)
                #print(branch_names)
                features = data_f['eventFeatures'][()]
                mask = features[:,0]<9000
                features = features[mask]
                mjj = np.asarray(features[:,0])
                loss_indices = get_loss_indices(branch_names)
            
                loss = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

                for j,q in enumerate(quantiles):
                    if q == '30':
                        if len(data_sel_q30) == 0:
                            data_sel_q30 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q30 = np.concatenate((data_sel_q30,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '70':
                        if len(data_sel_q70) == 0:
                            data_sel_q70 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q70 = np.concatenate((data_sel_q70,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '50':
                        if len(data_sel_q50) == 0:
                            data_sel_q50 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q50 = np.concatenate((data_sel_q50,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '10':
                        if len(data_sel_q10) == 0:
                            data_sel_q10 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q10 = np.concatenate((data_sel_q10,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '05':
                        if len(data_sel_q05) == 0:
                            data_sel_q05 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q05 = np.concatenate((data_sel_q05,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                    if q == '01':
                        if len(data_sel_q01) == 0:
                            data_sel_q01 = np.expand_dims((loss > qrs[j](mjj)),axis=-1)
                        else:
                            data_sel_q01 = np.concatenate((data_sel_q01,np.expand_dims((loss > qrs[j](mjj)),axis=-1)))
                if len(data_loss) == 0:
                    data_loss = np.expand_dims(loss,axis=-1)
                    data_mjj = np.expand_dims(mjj,axis=-1)
                else:
                    data_loss = np.concatenate((data_loss,np.expand_dims(loss,axis=-1)))
                    data_mjj = np.concatenate((data_mjj,np.expand_dims(mjj,axis=-1)))


    all_event_features = np.concatenate((all_mjj,all_loss,all_sel_q70,all_sel_q50,all_sel_q30,all_sel_q10,all_sel_q05,all_sel_q01),axis=-1)
    sig_event_features = np.concatenate((sig_mjj,sig_loss,sig_sel_q70,sig_sel_q50,sig_sel_q30,sig_sel_q10,sig_sel_q05,sig_sel_q01),axis=-1)
    if siginj != 0:
        data_event_features = np.concatenate((data_mjj,data_loss,data_sel_q70,data_sel_q50,data_sel_q30,data_sel_q10,data_sel_q05,data_sel_q01),axis=-1)
        

    print(all_event_features.shape)
    
    case_qr_datasets=f'/ceph/abal/CASE/QR_datasets/run_{run_n}/unblinded_SRData/{nAttempt}'
    pathlib.Path(case_qr_datasets).mkdir(parents=True,exist_ok=True)
    print(f'Output datasets to be used for fitting are located at {case_qr_datasets}')
    outfilename = 'bkg-2'
    dt = h5py.special_dtype(vlen=str)

    if siginj != 0:
        outfilename = 'bkg_%s_%s'%(signal_name,str(siginj))

    hf = h5py.File(os.path.join(case_qr_datasets,f'{outfilename}.h5'), 'w')
    #write_log(case_qr_datasets,comments)
    log_command = f'cp -r {case_qr_models}/log.txt {case_qr_datasets}/'
    subprocess.call(log_command,shell=True)
    hf.create_dataset('mjj', data=np.array(all_mjj))
    hf.create_dataset('loss', data=np.array(all_loss))
    hf.create_dataset('sel_q30', data=np.array(all_sel_q70))
    hf.create_dataset('sel_q50', data=np.array(all_sel_q50))
    hf.create_dataset('sel_q70', data=np.array(all_sel_q30))
    hf.create_dataset('sel_q90', data=np.array(all_sel_q10))
    hf.create_dataset('sel_q95', data=np.array(all_sel_q05))
    hf.create_dataset('sel_q99', data=np.array(all_sel_q01))
    hf.create_dataset('eventFeatures', data=np.array(all_event_features))
    hf.create_dataset('eventFeatureNames', data=np.array(all_event_feature_names_reversed,dtype=dt))
    hf.close()
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

    if siginj != 0:
        data_outfilename = 'bkg+inj_%s_%s'%(signal_name,str(siginj))
        data_hf = h5py.File(os.path.join(case_qr_datasets,data_outfilename), 'w')
        data_hf.create_dataset('mjj', data=np.array(data_mjj))
        data_hf.create_dataset('loss', data=np.array(data_loss))
        data_hf.create_dataset('sel_q30', data=np.array(data_sel_q70))
        data_hf.create_dataset('sel_q50', data=np.array(data_sel_q50))
        data_hf.create_dataset('sel_q70', data=np.array(data_sel_q30))
        data_hf.create_dataset('sel_q90', data=np.array(data_sel_q10))
        data_hf.create_dataset('sel_q95', data=np.array(data_sel_q05))
        data_hf.create_dataset('sel_q99', data=np.array(data_sel_q01))
        data_hf.create_dataset('eventFeatures', data=np.array(data_event_features))
        data_hf.create_dataset('eventFeatureNames', data=data_event_feature_names_reversed)
        data_hf.close()


