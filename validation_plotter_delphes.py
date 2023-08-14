import sys
import os
import h5py
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import gridspec
import seaborn as sns;sns.set_context("paper")
import mplhep as hep;plt.style.use(hep.style.CMS)
import json
import pathlib
from recordtype import recordtype
sig = True
siginj = 0.01
signal_name = 'grav_3p5_naReco'
#methods=['ER','expectiles']
methods=['QR','quantiles']
smoothed = True
work_dir='/work/abal/CASE/CASE_as_on_github/'
srange=[1455,6500]
qcd_sample = 'delphes_bkg'
reg=250
run_n=50005  
quan=30
# If smoothed
if smoothed:
    qr_cut_data_dir = f'/ceph/abal/CASE/QR_datasets/run_{run_n}/{methods[0]}/{quan}_{methods[1]}'
    filename = f'{qr_cut_data_dir}/{qcd_sample}_smoothing_{srange[0]}_{srange[1]}.h5'
    if siginj!=0:
        filename = f'{qr_cut_data_dir}/{qcd_sample}_smoothing_{srange[0]}_{srange[1]}_inj_{siginj}_{signal_name}.h5'
        
# If not smoothed over
else:
    qr_cut_data_dir = f'/ceph/abal/CASE/QR_datasets/run_{run_n}/delphes_tests/{methods[0]}ext/{quan}_{methods[1]}'
    filename = f'{qr_cut_data_dir}/{qcd_sample}_no_smoothing.h5'
    if siginj!=0:
        filename = f'{qr_cut_data_dir}/{qcd_sample}_no_smoothing_inj_{siginj}_{signal_name}'
    
print(f'reading from {filename}')
ModelParameters = recordtype('ModelParameters','run_n,epochs,batch_sz,n_layers,n_nodes')
model_params = ModelParameters(run_n=run_n,epochs=800,batch_sz=100,n_layers=5,n_nodes=30)

#labels = ['$q=0.7$-$1.0$', '$q=0.5$-$0.7$', '$q=0.3$-$0.5$', '$q=0.1$-$0.3$', '$q=0.05$-$0.1$', '$q=0.01$-$0.05$', '$q<0.01$']

labels = ['$q=0.7$-$1.0$', '$q=0.05$-$0.1$', '$q=0.05$-$0.01$', '$q<0.01$']
#labels = ['$e=0.7$-$1.0$', '$e=0.05$-$0.1$', '$e=0.05$-$0.01$', '$e<0.01$']


# def masks(f):
#     mask_q70_100 = (f['sel_q30'][()] == 0)
#     mask_q50_70 = (f['sel_q30'][()] == 1) & (f['sel_q50'][()] == 0)
#     mask_q30_50 = (f['sel_q50'][()] == 1) & (f['sel_q70'][()] == 0)
#     mask_q10_30 = (f['sel_q70'][()] == 1) & (f['sel_q90'][()] == 0)
#     mask_q10 = (f['sel_q90'][()] == 1)
    
#     mask_q05_10 = (f['sel_q90'][()] == 1) & (f['sel_q95'][()] == 0)
#     mask_q05 = (f['sel_q95'][()] == 1)
#     mask_q01_05 = (f['sel_q95'][()] == 1) & (f['sel_q99'][()] == 0)
#     mask_q01 = (f['sel_q99'][()] == 1)
#     mask_dict={'mask_q70_100':mask_q70_100,'mask_q50_70':mask_q50_70,'mask_q30_50':mask_q30_50,'mask_q10_30':mask_q10_30,\
#                'mask_q05_10':mask_q05_10,'mask_q01_05':mask_q01_05,'mask_q01':mask_q01,'mask_q05':mask_q05,'mask_q10':mask_q10}
#     return mask_dict

def masks(f):
    
    mask_q70_100 = (f['sel_q30'][()] == 0)
    mask_q05_10 = (f['sel_q90'][()] == 1) & (f['sel_q95'][()] == 0)
    mask_q05 = (f['sel_q95'][()] == 1)
    mask_q01_05 = (f['sel_q95'][()] == 1) & (f['sel_q99'][()] == 0)
    mask_q01 = (f['sel_q99'][()] == 1)
    mask_dict={'mask_q70_100':mask_q70_100,'mask_q05_10':mask_q05_10,'mask_q01_05':mask_q01_05,'mask_q01':mask_q01}
    return mask_dict

f = h5py.File(filename, 'r')
print(f.keys())

mask_data = masks(f)


hist_q70_100 = f['mjj'][()][mask_data['mask_q70_100']]
# hist_q50_70 = f['mjj'][()][mask_data['mask_q50_70']]
# hist_q30_50 = f['mjj'][()][mask_data['mask_q30_50']]
# hist_q10_30 = f['mjj'][()][mask_data['mask_q10_30']]
hist_q05_10 = f['mjj'][()][mask_data['mask_q05_10']]
hist_q01_05 = f['mjj'][()][mask_data['mask_q01_05']]
hist_q01 = f['mjj'][()][mask_data['mask_q01']]
# hist_q05 = f['mjj'][()][mask_data['mask_q05']]
# hist_q10 = f['mjj'][()][mask_data['mask_q10']]


fig = plt.figure(figsize=(13,12))

gs = gridspec.GridSpec(2, 1, height_ratios=[10.0,3.0])
#gs.

ax = plt.subplot(gs[0])
hep.cms.label('Data', lumi=137, year='2016-18',loc=0, data=True)


ax_ratio1 = plt.subplot(gs[1])


bins = np.linspace(1400,7600,40)
#fig,ax = plt.subplots(figsize=(7,7))
ax.set_yscale('log')
ax.set_xlim(1400.,8000.)
ax_ratio1.set_xlim(1400.,8000.)
ax_ratio1.set_ylim(0.45,1.55)


DATA,_ = np.histogram(hist_q70_100,bins=bins)#,histtype='step',label=labels[0])
# DATA30,_ = np.histogram(hist_q50_70,bins=bins)#,histtype='step',label=labels[1])
# DATA50,_ = np.histogram(hist_q30_50,bins=bins)#,histtype='step',label=labels[2])
# DATA70,_ = np.histogram(hist_q10_30,bins=bins)#,histtype='step',label=labels[3])
#DATA90,_ = np.histogram(hist_q05_10,bins=bins)#,histtype='step',label=labels[4])
DATA90,_ = np.histogram(hist_q05_10,bins=bins)#,histtype='step',label=labels[4])
DATA95,_ = np.histogram(hist_q01_05,bins=bins)#,histtype='step',label=labels[4])
DATA99,_= np.histogram(hist_q01,bins=bins)#,histtype='step',label=labels[5])
    
#yvals=[DATA,DATA30,DATA50,DATA70,DATA90,DATA95,DATA99]
yvals=[DATA,DATA90,DATA95,DATA99]

ax.set_xticklabels([])
#ax_ratio1.set_xticklabels([])

colors = sns.color_palette('YlGnBu',7)[::-1]
scolors = sns.color_palette('RdPu',7)[::-1]


denominator=yvals[0]

for i in range(len(yvals)):
    ax.errorbar(bins[:-1], yvals[i], yerr = yvals[i]**0.5, marker = '.',drawstyle = 'steps-mid', color=colors[i],label=labels[i])
    
    
for i in range(1,len(yvals)):
    divarr = np.divide(yvals[i], denominator, out=np.zeros_like(denominator,dtype='float64'), where=denominator!=0)
    diverrarr = np.divide(yvals[i]**0.5, denominator, out=np.zeros_like(denominator,dtype='float64'), where=denominator!=0.)
    ax_ratio1.errorbar(bins[:-1],np.sum(denominator)/np.sum(yvals[i]) * divarr,
                yerr=np.sum(denominator)/np.sum(yvals[i])*diverrarr,fmt='o',color=colors[i])
    
#ax.text(0.36,0.79,"$\Delta\eta_{jj} \in [2.0,2.5]$",fontsize=17,transform=ax.transAxes)
ax.text(0.34,0.79,"$\Delta\eta_{jj} < 1.3$",fontsize=14,transform=ax.transAxes)
if smoothed:
    ax.text(0.34,0.73,f"Smoothing range $\in ({srange[0]},{srange[1]}) GeV$",fontsize=14,transform=ax.transAxes)

if siginj!=0:
    #ax.text(0.34,0.68,"$M_{W\prime}=3\,\\mathrm{TeV}$, $M_{B\prime}=400\,\\mathrm{GeV}$, $M_{t}=170\,\\mathrm{GeV}$",fontsize=17,transform=ax.transAxes)
    ax.text(0.34,0.68,"$M_{graviton}=3.5\,\\mathrm{TeV}$",fontsize=14,transform=ax.transAxes)
    ax.text(0.34,0.63,f"Injected signal: ${int(siginj*1000)} fb^{{-1}}$",fontsize=14,transform=ax.transAxes)

ax_ratio1.axhline(y=1.0, color='k', linestyle='--')
#ax_ratio1.set_ylabel("$q$/0.05-0.1",fontsize=21)
#ax_ratio1.set_ylabel("$q$/0.7-1.0",fontsize=21)
ax_ratio1.set_ylabel("$e$/0.7-1.0",fontsize=21)
ax_ratio1.set_xlabel("$M_{jj}$ (in GeV)")
ax.set_ylabel("No. of events")
ax.text(0.34,0.88,f"{methods[0]} applied on signal region QCD (Delphes dataset)",fontsize=14,transform=ax.transAxes,weight='bold')
# ax.text(0.05,1.01,"CMS Simulation",fontsize=26,transform=ax.transAxes,weight="bold")
# ax.text(0.76,1.01,"$26.8\,\\mathrm{fb}^{\\mathrm{-}1}$",fontsize=18,transform=ax.transAxes)
# ax.text(0.88,1.01,"$2017$",fontsize=18,transform=ax.transAxes)

ax.legend(frameon=False,fontsize=19,loc=(0.75,0.6),ncol=1)
#ax_ratio1.legend(loc='upper right')

# Create image folder if it doesn't exist
image_dir='/etpwww/web/abal/public_html/CASE/delphes_tests'
pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)
if smoothed:
    img_path = os.path.join(image_dir,f'{qcd_sample}_extended{methods[1].capitalize()}1-{quan}_smoothed{srange[0]}_{srange[1]}.png')
    if siginj!=0:
        img_path = os.path.join(image_dir,f'{qcd_sample}_extended{methods[1].capitalize()}1-{quan}_smoothed{srange[0]}_{srange[1]}_inj_{siginj}_{signal_name}.png')
        
else:
    img_path = os.path.join(image_dir,f'{qcd_sample}_extended{methods[1].capitalize()}1-{quan}.png')
    if siginj!=0:
        img_path = os.path.join(image_dir,f'{qcd_sample}_extended{methods[1].capitalize()}1-{quan}_inj_{siginj}_{signal_name}.png')
        

#img_path = os.path.join(image_dir,f'bkg_only_WITH_MIXING__NO_SMOOTHING_batchsz100_nlayers3.png')


plt.savefig(img_path)
plt.close()
f.close()