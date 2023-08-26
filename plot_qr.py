
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py


def get_loss_indices(branch_names):
    j1_reco = np.argwhere(branch_names=='j1RecoLoss')[0,0]
    j1_kl = np.argwhere(branch_names=='j1KlLoss')[0,0]
    j2_reco = np.argwhere(branch_names=='j2RecoLoss')[0,0]
    j2_kl = np.argwhere(branch_names=='j2KlLoss')[0,0]
    loss_dict = {'j1KlLoss':j1_kl,'j1RecoLoss':j1_reco,'j2KlLoss':j2_kl,'j2RecoLoss':j2_reco}
    return loss_dict

run_n = 113
k=0
##### Define directory paths
signal_name='grav_3p5_naReco'
nAttempt=2
siginj=0.05
inj_qcd_sample = f'delphes_bkg+inj_{signal_name}_{siginj}' # File with injected signal but no mixing. QR was trained on this file with jet mixing
data_filename=f'/storage/9/abal/CASE/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/{nAttempt}/{inj_qcd_sample}.h5'
case_qr_models = f'/work/abal/CASE/QR_models/run_{run_n}/delphes_models_lmfit_csv/{nAttempt}/{signal_name}_{siginj}'
case_qr_results = f'/storage/9/abal/CASE/QR_results/events/run_{run_n}/sig_{signal_name}/xsec_0/loss_rk5_05/{nAttempt}'

with h5py.File(data_filename, "r") as bkg_f:
    branch_names = bkg_f['eventFeatureNames'][()].astype(str)
    #print(branch_names)
    features = bkg_f['eventFeatures'][()]
    mask = features[:,0]<9000
    features = features[mask]
    mjj = np.asarray(features[:,0])

tmpdf = pd.read_csv(f'{case_qr_models}/bkg+inj_lmfit_modelresult_fold_{k}_quant_q01.csv')
p = np.poly1d(tmpdf['par'].values.tolist()[::-1])

m_dis=np.arange(1200.,6800.,1.)

loss=p(m_dis)
loss_indices = get_loss_indices(branch_names)
losses = np.minimum(features[:,loss_indices['j1RecoLoss']]+0.5*features[:,loss_indices['j1KlLoss']],features[:,loss_indices['j2RecoLoss']]+0.5*features[:,loss_indices['j2KlLoss']]) 

plt.plot(m_dis,loss)

plt.scatter(mjj,losses,s=0.1)
plt.savefig('/etpwww/web/abal/public_html/test/test1.png')