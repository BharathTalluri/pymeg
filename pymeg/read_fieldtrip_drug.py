
import h5py
import numpy as np
import os
import glob

from joblib import Memory

import mne

from mne import create_info
from mne.epochs import EpochsArray


memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)


def fix_chs(rawinfo, einfo):
    ch_names = []
    for k in range(len(einfo['chs'])):
        name = einfo['chs'][k]['ch_name']
        newchan = [x for x in rawinfo['chs']
                   if name in x['ch_name']][0]
        einfo['chs'][k] = newchan
        ch_names.append(newchan['ch_name'])
    einfo['ch_names'] = ch_names
    return einfo


@memory.cache
def get_info_for_epochs(rawname):
    raw = mne.io.ctf.read_raw_ctf(rawname)
    return raw.info


def read_ft_epochs(fname, rawinfo, cachedir=os.environ['PYMEG_CACHE_DIR'],
                   trialinfo_col=-1):
    '''
    Read and cache the output of fieldtrip epochs.

    This function reads a matlab file that contains a 'data' struct with the
    following fields:

        trialinfo: matrix
            Dim is ntrials x nr_meta, the columns contain meta information
            about each trial.
        label: list of strings
            Channel names
        sampleinfo: matrix
            Dim is ntrials x 2, the first column contains the start sample
            of each epoch in the raw data.
        time: array
            Contains time points for epochs.
        trial: array
            Dim is time x channels x trials, contains the actial data

    This data is parsed into an MNE Epochs object. To correctly assign
    channel locations, types etc. the info structure from the raw data
    that generated the fieldtrip epochs is used. The channel names in
    the fieldtrip structure should still be relatable to the raw
    channel names, relatable here means that a fieldtrip channel name
    must be contained in the raw channel name.

    Args
        fname: str
            Path to .mat file to load
        rawinfo: mne info structure
            Info structure with correct channel locations etc. This
            should be obtained by reading the raw data corresponding
            to the epochs with MNE.
        cachedir: str
            Path where the epochs are saved on disk. If this is
            None the epochs are returned.
        trialinfo_col: int
            Column in trialinfo which contains trial identifier.

    Output
        Returns path to saved epochs if cachedir is not None, else
        it returns the epochs
    '''
    if cachedir is None:
        return _load_ft_epochs(fname, rawinfo, trialinfo_col=trialinfo_col)
    epochs_path = os.path.join(cachedir, fname + '-epo.fif.gz')
    if not os.path.exists(epochs_path):
        epochs = _load_ft_epochs(fname, rawinfo, trialinfo_col=trialinfo_col)
        epochs.save(epochs_path)
    return epochs_path


def _load_ft_epochs(fname, rawinfo, trialinfo_col=-1):
    # load Matlab/Fieldtrip data
    f = h5py.File(fname)
    list(f.keys())
    ft_data = f['data']
    ft_data.keys()

    trialinfo = ft_data['trialinfo']
    channels = ft_data['label']
    sampleinfo = ft_data['sampleinfo']
    time = ft_data['time']
    sfreq = np.around(1 / np.diff(time[:].ravel()), 2)
    assert(len(np.unique(sfreq)) == 1)
    n_time, n_chans, n_trial = ft_data['trial'].shape

    data = np.zeros((n_trial, n_chans, n_time))
    transposed_data = np.transpose(ft_data['trial'])
    for trial in range(n_trial):
        data[trial, :, :] = transposed_data[trial]

    data = data[:, range(n_chans), :]

    chan_names = []
    for i in range(n_chans):
        st = channels[0][i]
        obj = ft_data[st]
        chan_names.append(''.join(chr(j) for j in obj[:]))
    #ch_names = [x + '-3705' for x in chan_names]

    info = create_info(chan_names, sfreq[0])
    events = np.zeros((n_trial, 3), int)
    events[:, 2] = trialinfo[trialinfo_col]
    events[:, 0] = sampleinfo[0]

    epochs = EpochsArray(data, info, events=events, tmin = min(time)[0], verbose=False)
    epochs.info = fix_chs(rawinfo, epochs.info)
    return epochs

#subjects = {'CMC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'CPG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'CTG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'CYK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 5)],
#            'EJG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'FDC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'FLW': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'FNC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'GHT': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'HFK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'HOG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'ICD': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'JPK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'LOK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'MDC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'MJQ': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'MLC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'NDT': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'OIG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'QLW': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'RGT': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'ROW': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'RPC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'TMW': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (3, 5)],
#            'UDK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'UOC': [(1, 1), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'URG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'UXQ': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'VTQ': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
#            'XUE': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)]}

subjects = {'CMC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'CPG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'CTG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'CYK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 5)],
            'EJG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'FDC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 3), (3, 4)],
            'FLW': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'FNC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'GHT': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'HFK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'HOG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'ICD': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'JPK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'LOK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'MDC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'MJQ': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'MLC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'NDT': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'OIG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'QLW': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'RGT': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'ROW': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'RPC': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'TMW': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (3, 5)],
            'UDK': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'UOC': [(1, 1), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'URG': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'UXQ': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'VTQ': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
            'XUE': [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)]}

ftdir = '/home/pmurphy/Surprise_drug/Analysis/MEG/Preprocessed4mne/'
megdirs = ['/home/pmurphy/meg_data/surpriseD/','/mnt/homes/home024/jschipp/Surprise_Drug/Data/meg_data/']
savedir = '/home/pmurphy/Surprise_drug/Analysis/MEG/Conv2mne_induced/'

for subj, tasks in subjects.items():
    for sess, rec in tasks:
        ftname = ftdir + str(subj) + '-' + str(sess) + '_0' + str(rec) + '_preproc4mne_induced.mat';
        print(ftname)
        if subj in ['CPG','CTG','EJG','GHT','LOK','MJQ','OIG','ROW','TMW','UDK','VTQ']: # if one of Julia's
            megname = glob.glob(megdirs[1] + str(subj) + '-' + str(sess) + '*_0' + str(rec) + '.ds')
        elif subj=='JPK' and sess==1:  # naming error with meg files for this subj/sess (labelled 'JBK')
            megname = glob.glob(megdirs[0] + 'JBK' + '-' + str(sess) + '*_0' + str(rec) + '.ds')
        elif subj=='URG' and sess==1:  # naming error with meg files for this subj/sess (missing sess identifier)
            megname = glob.glob(megdirs[0] + str(subj) + '_' + '*_0' + str(rec) + '.ds')
        else:
            megname = glob.glob(megdirs[0] + str(subj) + '-' + str(sess) + '*_0' + str(rec) + '.ds')
        print(megname[0])
        
        rawinfo = get_info_for_epochs(megname[0])
        epochs = _load_ft_epochs(ftname, rawinfo)
        epochs.save(savedir + str(subj) + '-' + str(sess) + '_0' + str(rec) + '_preproc4mne_induced.mat' + '-epo.fif.gz')
        del epochs
        



