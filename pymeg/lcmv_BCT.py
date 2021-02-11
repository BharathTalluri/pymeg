import logging
import mne
import numpy as np
import os

from joblib import Memory

from os import makedirs
from os.path import join
from glob import glob

from pymeg import lcmv as pymeglcmv
from pymeg import source_reconstruction as pymegsr


memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'])
path = '/home/btalluri/confirmation_spatial/data/meg/analysis/conv2mne/'


def set_n_threads(n):
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OMP_NUM_THREADS'] = str(n)


subjects = {'Pilot03': [(1, 1), (2, 1), (3, 1), (4, 1)],
            'Pilot06': [(2, 2), (3, 1), (4, 1)]}
#            'Pilot04': [(1, 1), (2, 1), (3, 2), (4, 1)],
#            'Pilot06': [(2, 2), (3, 1), (4, 1)]}

# subjects = {'CMC': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'CPG': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'CTG': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'CYK': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,5,'r2')],
#            'EJG': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'FDC': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,2,'r1'), (2,3,'r1'), (2,4,'r2'), (2,5,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'FLW': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'FNC': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'GHT': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'HFK': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'HOG': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'ICD': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'JPK': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'LOK': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'MDC': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'MJQ': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'MLC': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'NDT': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'OIG': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'QLW': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'RGT': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'ROW': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'RPC': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'TMW': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,2,'r1'), (3,3,'r1'), (3,4,'r2'), (3,5,'r2')],
#            'UDK': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'UOC': [(1,1,'r1'), (1,3,'r1'), (1,4,'r2'), (1,5,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'URG': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'UXQ': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'VTQ': [(1,1,'r1'), (1,2,'r1'), (1,3,'r1'), (1,4,'r2'), (1,5,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')],
#            'XUE': [(1,1,'r1'), (1,2,'r1'), (1,3,'r2'), (1,4,'r2'), (2,1,'r1'), (2,2,'r1'), (2,3,'r2'), (2,4,'r2'), (3,1,'r1'), (3,2,'r1'), (3,3,'r2'), (3,4,'r2')]}


def submit():
    from pymeg import parallel
    for subject, tasks in subjects.items():
        for session, recording in tasks:
            for signal in ['LF', 'HF']:  # for signal in ['BB', 'HF', 'LF']:
                parallel.pmap(
                    extract, [(subject, session, recording, signal)],
                    walltime='15:00:00', memory=50, nodes=1, tasks=5,
                    name='sr' + str(subject) + '_' + str(session) + str(recording), ssh_to='node028', env='py36')


def lcmvfilename(subject, session, signal, recording, chunk=None):
    try:
        makedirs(path)
    except:
        pass
    if chunk is None:
        filename = '%s-SESS%i-%i-%s-lcmv.hdf' % (
            subject, session, recording, signal)
    else:
        filename = '%s-SESS%i-%i-%s-chunk%i-lcmv.hdf' % (
            subject, session, recording, signal, chunk)
    return join(path, filename)


def get_stim_epoch(subject, session, recording):
    from pymeg import preprocessing as pymegprepr
    globstring = '/home/btalluri/confirmation_spatial/data/meg/analysis/conv2mne/%s-%i_0%i_*fif.gz' % (subject, session, recording)
    filenames = glob(globstring)[0]
    epochs = mne.read_epochs(filenames)
    # epochs.times = epochs.times - 1  # PM: this was somehow necessary for initial pipeline, but *NOT* for induced
    epochs = epochs.pick_channels([x for x in epochs.ch_names if x.startswith('M')])
    id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    min_time, max_time = epochs.times.min() + 0.75, epochs.times.max() - 0.5
    data_cov = pymeglcmv.get_cov(epochs, tmin=min_time, tmax=max_time)
    return data_cov, epochs, filenames


def extract(subject, session, recording, signal_type='BB', BEM='three_layer', debug=False, chunks=100, njobs=4):
    mne.set_log_level('WARNING')
    pymeglcmv.logging.getLogger().setLevel(logging.INFO)
    set_n_threads(1)

    logging.info('Reading stimulus data')
    data_cov, epochs, epochs_filename = get_stim_epoch(subject, session, recording)

    megdirs = '/home/btalluri/confirmation_spatial/data/meg/raw/'

    if subject == 'Pilot03' and session == 1:  # naming error with meg files for this subj/sess (labelled 'JBK')
        raw_filename = glob(megdirs + str(subject) + '_' + '*_0' + str(recording) + '.ds')
    else:
        raw_filename = glob(megdirs + str(subject) + '-' + str(session) + '*_0' + str(recording) + '.ds')

    assert len(raw_filename) == 1
    raw_filename = raw_filename[0]

    trans_filename = glob('/home/btalluri/confirmation_spatial/data/mri/trans_mats/%s-%i_0%i-trans.fif' % (subject, session, recording))[0]
    logging.info('Setting up source space and forward model')

    conductivity = (0.3, 0.006, 0.3)  # otherwise, use the default settings

    forward, bem, source = pymegsr.get_leadfield(subject, raw_filename, epochs_filename, trans_filename, bem_sub_path='bem_ft', conductivity=conductivity)
    labels = pymegsr.get_labels(subject)

    labels = pymegsr.labels_exclude(labels,
                                    exclude_filters=['wang2015atlas.IPS4',
                                                     'wang2015atlas.IPS5',
                                                     'wang2015atlas.SPL',
                                                     'JWDG_lat_Unknown'])
    labels = pymegsr.labels_remove_overlap(labels, priority_filters=['wang', 'JWDG'],)

    # Now chunk Reconstruction into blocks of ~100 trials to save Memory
    fois_h = np.arange(36, 162, 4)
    fois_l = np.arange(1, 36, 1)
    tfr_params = {
        'HF': {'foi': fois_h, 'cycles': fois_h * 0.25, 'time_bandwidth': 6 + 1,
               'n_jobs': njobs, 'est_val': fois_h, 'est_key': 'HF', 'sf': 400,
               'decim': 20},
        'LF': {'foi': fois_l, 'cycles': fois_l * 0.4, 'time_bandwidth': 1 + 1,
               'n_jobs': njobs, 'est_val': fois_l, 'est_key': 'LF', 'sf': 400,
               'decim': 20}
    }

    events = epochs.events[:, 2]
    filters = pymeglcmv.setup_filters(epochs.info, forward, data_cov, None, labels)

    set_n_threads(1)

    for i in range(0, len(events), chunks):
        filename = lcmvfilename(subject, session, signal_type, recording, chunk=i)
        if os.path.isfile(filename):
            continue
        if signal_type == 'BB':
            logging.info('Starting reconstruction of BB signal')
            M = pymeglcmv.reconstruct_broadband(
                filters, epochs.info, epochs._data[i:i + chunks],
                events[i:i + chunks],
                epochs.times, njobs=1)
        else:
            logging.info('Starting reconstruction of TFR signal')
            M = pymeglcmv.reconstruct_tfr(
                filters, epochs.info, epochs._data[i:i + chunks],
                events[i:i + chunks], epochs.times,
                est_args=tfr_params[signal_type],
                njobs=4)
        M.to_hdf(filename, 'epochs')
    set_n_threads(njobs)
