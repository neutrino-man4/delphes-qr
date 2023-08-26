import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

f2=h5py.File('/ceph/abal/CASE/QR_datasets/run_113/2/bkg+inj_grav_3p5_naReco_0.05.h5','r')
f3=h5py.File('/ceph/abal/CASE/QR_datasets/run_113/3/bkg+inj_grav_3p5_naReco_0.05.h5','r')
bins = np.linspace(1400,7600,40)

def masks(f):
    mask_q70_100 = (f['sel_q30'][()] == 0)
    mask_q50_70 = (f['sel_q30'][()] == 1) & (f['sel_q50'][()] == 0)
    mask_q30_50 = (f['sel_q50'][()] == 1) & (f['sel_q70'][()] == 0)
    mask_q10_30 = (f['sel_q70'][()] == 1) & (f['sel_q90'][()] == 0)
    mask_q10 = (f['sel_q90'][()] == 1)
    
    mask_q05_10 = (f['sel_q90'][()] == 1) & (f['sel_q95'][()] == 0)
    mask_q05 = (f['sel_q95'][()] == 1)
    mask_q01_05 = (f['sel_q95'][()] == 1) & (f['sel_q99'][()] == 0)
    mask_q01 = (f['sel_q99'][()] == 1)
    mask_dict={'mask_q70_100':mask_q70_100,'mask_q50_70':mask_q50_70,'mask_q30_50':mask_q30_50,'mask_q10_30':mask_q10_30,\
               'mask_q05_10':mask_q05_10,'mask_q01_05':mask_q01_05,'mask_q01':mask_q01,'mask_q05':mask_q05,'mask_q10':mask_q10}
    return mask_dict

for f,label in zip([f2,f3],['with jet mixing','without jet mixing']):
    mask_data=masks(f)
    hist_q70_100 = f['mjj'][()][mask_data['mask_q70_100']]
    hist_q05 = f['mjj'][()][mask_data['mask_q05']]
    hist_q01 = f['mjj'][()][mask_data['mask_q01']]
    hist_q70_100,_=np.histogram(hist_q70_100,bins=bins)
    hist_q05,_=np.histogram(hist_q05,bins=bins)
    hist_q01,_=np.histogram(hist_q01,bins=bins)
    
    chosen_hist=hist_q05

    divarr = np.divide(chosen_hist, hist_q70_100, out=np.zeros_like(hist_q70_100,dtype='float64'), where=hist_q70_100!=0)
    diverrarr = np.divide(chosen_hist**0.5, hist_q70_100, out=np.zeros_like(hist_q70_100,dtype='float64'), where=hist_q70_100!=0.)    
    plt.errorbar(bins[:-1],np.sum(hist_q70_100)/np.sum(chosen_hist)*divarr,yerr=np.sum(hist_q70_100)/np.sum(chosen_hist)*diverrarr,fmt='o',label=label)

plt.legend(loc='upper right')
plt.ylim(0.5,1.5)
plt.axhline(y=1.0,linestyle='--')
plt.xlabel('$M_{jj}$ (in GeV)')
plt.ylabel('Q95/Q30')
#plt.show()
image_dir=f'/etpwww/web/abal/public_html/CASE/delphes_tests/comparisons'

plt.savefig(os.path.join(image_dir,'Q95_ratio.png'),dpi=600)