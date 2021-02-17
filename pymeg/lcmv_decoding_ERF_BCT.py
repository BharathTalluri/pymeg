# Some imports:
import logging
import mne
import numpy as np
import pandas as pd
import os
import scipy.io as sio
import time
from joblib import Memory  # Provides caching of results
from glob import glob

from pymeg import lcmv as pymeglcmv
from pymeg import source_reconstruction as pymegsr
from pymeg import decoding_ERF

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import linear_model

from pymeg.lcmv_BCT import get_stim_epoch
from pymeg.source_reconstruction import get_leadfield  # NB: source_reconstruction also has get_trans_epoch, get_ctf_trans - not sure which we want, NW has get_trans

# Setup some paths:
memory = Memory(cachedir='/mnt/homes/home024/btalluri/tmp/')
subjects_dir = "/home/btalluri/confirmation_spatial/data/mri/fs_converted"  # freesurfer subject dirs
trans_dir = "/home/btalluri/confirmation_spatial/data/mri/trans_mats"  # transofrmation matrices

# We need to restrict the number of threads that we use for the cluster


def set_n_threads(n):
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


# make dict of subjects/sessions/recordings
subjects = {'Pilot03': [(1, 1), (2, 1), (3, 1), (4, 1)],
            'Pilot04': [(1, 1), (2, 1), (3, 2), (4, 1)],
            'Pilot06': [(2, 2), (3, 1), (4, 1)]}
subjects_attn = {'Pilot03': [(1, 1), (2, 1)],
                 'Pilot04': [(1, 1), (2, 1)],
                 'Pilot06': [(2, 2)]}
subjects_choice = {'Pilot03': [(3, 1), (4, 1)],
                   'Pilot04': [(3, 2), (4, 1)],
                   'Pilot06': [(3, 1), (4, 1)]}
conditions = ['all', 'attn', 'choice']

# typical processing demands (in # cores) per subject
mem_demand = {'Pilot03': 5, 'Pilot04': 5, 'Pilot06': 5}

# typical processing demands (in # vertices) per area (not used but useful reference fo setting ntasks at submit)
mem_area = {'vfcPrimary': 3650, 'vfcEarly': 9610, 'vfcV3ab': 3280,
            'vfcIPS01': 4200, 'vfcIPS23': 2200, 'JWG_aIPS': 870,
            'JWG_IPS_PCeS': 4600, 'JWG_M1': 2900, 'HCPMMP1_premotor': 9900}

# Mapping areas to labels
areas_to_labels = {
    "vfcPrimary": [
        u"lh.wang2015atlas.V1d-lh", u"rh.wang2015atlas.V1d-rh",
        u"lh.wang2015atlas.V1v-lh", u"rh.wang2015atlas.V1v-rh",
    ],
    "vfcEarly": [
        u"lh.wang2015atlas.V2d-lh", u"rh.wang2015atlas.V2d-rh",
        u"lh.wang2015atlas.V2v-lh", u"rh.wang2015atlas.V2v-rh",
        u"lh.wang2015atlas.V3d-lh", u"rh.wang2015atlas.V3d-rh",
        u"lh.wang2015atlas.V3v-lh", u"rh.wang2015atlas.V3v-rh",
        u"lh.wang2015atlas.hV4-lh", u"rh.wang2015atlas.hV4-rh",
    ],
    "vfcVO": [
        u"lh.wang2015atlas.VO1-lh", u"rh.wang2015atlas.VO1-rh",
        u"lh.wang2015atlas.VO2-lh", u"rh.wang2015atlas.VO2-rh",
    ],
    "vfcPHC": [
        u"lh.wang2015atlas.PHC1-lh", u"rh.wang2015atlas.PHC1-rh",
        u"lh.wang2015atlas.PHC2-lh", u"rh.wang2015atlas.PHC2-rh",
    ],
    "vfcV3ab": [
        u"lh.wang2015atlas.V3A-lh", u"rh.wang2015atlas.V3A-rh",
        u"lh.wang2015atlas.V3B-lh", u"rh.wang2015atlas.V3B-rh",
    ],
    # "vfcTO": [
    #     u"lh.wang2015atlas.TO1-lh", u"rh.wang2015atlas.TO1-rh",
    #     u"lh.wang2015atlas.TO2-lh", u"rh.wang2015atlas.TO2-rh",
    # ],
    "vfcLO": [
        u"lh.wang2015atlas.LO1-lh", u"rh.wang2015atlas.LO1-rh",
        u"lh.wang2015atlas.LO2-lh", u"rh.wang2015atlas.LO2-rh",
    ],
    "vfcIPS01": [
        u"lh.wang2015atlas.IPS0-lh", u"rh.wang2015atlas.IPS0-rh",
        u"lh.wang2015atlas.IPS1-lh", u"rh.wang2015atlas.IPS1-rh",
    ],
    "vfcIPS23": [
        u"lh.wang2015atlas.IPS2-lh", u"rh.wang2015atlas.IPS2-rh",
        u"lh.wang2015atlas.IPS3-lh", u"rh.wang2015atlas.IPS3-rh",
    ],
    # "JWG_aIPS": ["lh.JWDG.lr_aIPS1-lh", "rh.JWDG.lr_aIPS1-rh", ],
    # "JWG_IPS_PCeS": ["lh.JWDG.lr_IPS_PCes-lh", "rh.JWDG.lr_IPS_PCes-rh", ],
    # "JWG_M1": ["lh.JWDG.lr_M1-lh", "rh.JWDG.lr_M1-rh", ],
    # "HCPMMP1_premotor": (["L_{}_ROI-lh".format(area) for area in ["55b", "6d", "6a", "FEF", "6v", "6r", "PEF"]] + ["R_{}_ROI-rh".format(area) for area in ["55b", "6d", "6a", "FEF", "6v", "6r", "PEF"]]),
}


# Submit to cluster. One job per ROI and subject
def submit_allCond(only_glasser=False):
    from pymeg import parallel
    # from itertools import product

    for area in list(areas_to_labels.keys()):
        for subject in subjects.keys():
            print("Submitting %s -> %s" % (subject, area))
            parallel.pmap(
                decode,
                [(subject, area, subjects[subject], 0)],  # added session/recording info as input
                walltime="80:00:00",
                memory=mem_demand[subject] * 10 - 10,
                nodes=1,
                tasks=mem_demand[subject] - 1,
                env="py36",
                name="decode_erf_" + area + subject,
            )
            time.sleep(1)


def submit_attn(only_glasser=False):
    from pymeg import parallel
    # from itertools import product

    for area in list(areas_to_labels.keys()):
        for subject in subjects_attn.keys():
            print("Submitting %s -> %s" % (subject, area))
            parallel.pmap(
                decode,
                [(subject, area, subjects_attn[subject], 1)],  # added session/recording info as input
                walltime="200:00:00",
                memory=mem_demand[subject] * 10 - 10,
                nodes=1,
                tasks=mem_demand[subject] - 1,
                env="py36",
                name="decode_erf_" + area + subject,
            )
            time.sleep(1)


def submit_choice(only_glasser=False):
    from pymeg import parallel
    # from itertools import product

    for area in list(areas_to_labels.keys()):
        for subject in subjects_choice.keys():
            print("Submitting %s -> %s" % (subject, area))
            parallel.pmap(
                decode,
                [(subject, area, subjects_choice[subject], 2)],  # added session/recording info as input
                walltime="200:00:00",
                memory=mem_demand[subject] * 10 - 10,
                nodes=1,
                tasks=mem_demand[subject] - 1,
                env="py36",
                name="decode_erf_" + area + subject,
            )
            time.sleep(1)


# This function returns labels for one subject
def get_labels(subject, only_glasser):
    if not only_glasser:
        labels = pymegsr.get_labels(
            subject=subject,
            filters=["*wang*.label", "*JWDG*.label"],
            annotations=["HCPMMP1"],
            sdir=subjects_dir,
        )
        labels = pymegsr.labels_exclude(
            labels=labels,
            exclude_filters=[
                "wang2015atlas.IPS4",
                "wang2015atlas.IPS5",
                "wang2015atlas.SPL",
                "JWDG_lat_Unknown",
            ],
        )
        labels = pymegsr.labels_remove_overlap(
            labels=labels, priority_filters=["wang", "JWDG"]
        )
    else:
        labels = pymegsr.get_labels(
            subject=subject,
            filters=["select_nothing"],
            annotations=["HCPMMP1"],
            sdir=subjects_dir,
        )
    return labels


# @memory.cache
def decode(subject, area, sessinfo, condition_idx, epoch_type="stimulus", only_glasser=False, BEM="three_layer", debug=False, target="response"):
    # Only show warning and above:
    mne.set_log_level("WARNING")
    pymeglcmv.logging.getLogger().setLevel(logging.INFO)

    set_n_threads(1)
    condition = conditions[condition_idx]
    # Get all labels for this subject
    if subject == 'Pilot04':
        labels = get_labels('fsaverage', only_glasser)
    else:
        labels = get_labels(subject, only_glasser)
    # And keep only the ones that we are interested in (area parameter of this function) -> needs to be in areas_to_labels dict defined above
    labels = [x for x in labels if any([cl for cl in areas_to_labels[area] if cl in x.name])]
    print(labels)

    if len(labels) < 1:
        raise RuntimeError('Expecting at least two labels')

    # Turn individual labels into one big label that contains all vertices that
    # belong to one ROI
    label = labels.pop()
    for l in labels:
        label += l

    print('Selecting this label for area %s:' % area, label)

    # ATTEMPT TAKING SESS/REC INFO & EXISTING FUNCTIONS INTO ACCOUNT
    data = []
    fwds = []
    bems = []
    sources = []
    sessions = []
    low_pass_fs = None   # low-pass filter cutoff; set to None is no filter required
    for sess, rec in sessinfo:
        logging.info("Reading data for %s, sess %i, rec %i " % (subject, sess, rec))
        data_cov, epoch, epoch_filename = get_stim_epoch(subject, sess, rec, lopass=low_pass_fs)
        data.append((data_cov, epoch))  # N.B. data may in fact need to be output currently called 'epochs'

        logging.info("Setting up source space and forward model")
        megdirs = '/home/btalluri/confirmation_spatial/data/meg/raw/'
        if subject == 'Pilot03' and sess == 1:  # naming error with meg files for this subj/sess (labelled 'JBK')
            raw_filename = glob(megdirs + str(subject) + '_' + '*_0' + str(rec) + '.ds')
        else:
            raw_filename = glob(megdirs + str(subject) + '-' + str(sess) + '*_0' + str(rec) + '.ds')

        assert len(raw_filename) == 1
        raw_filename = raw_filename[0]
        trans_filename = glob(trans_dir + '/' + str(subject) + '-' + str(sess) + '_0' + str(rec) + '-trans.fif')[0]
        if subject == 'Pilot04':
            forward, bem, source = get_leadfield('fsaverage', raw_filename, epoch_filename, trans_filename, bem_sub_path='bem_ft')
        else:
            forward, bem, source = get_leadfield(subject, raw_filename, epoch_filename, trans_filename, bem_sub_path='bem_ft')
        fwds.append(forward)
        sessions.append(sess)
        # bems.append(bem)
        # sources.append(source)

    # pull trial IDs
    events = [d[1].events[:, 2] for d in data]
    events = np.hstack(events)

    # Compute LCMV filters for each session
    filters = []
    for (data_cov, epochs), forward in zip(data, fwds):   # PM: N.B. data_cov may not be part of data as currently configured
        filters.append(pymeglcmv.setup_filters(epochs.info, forward, data_cov, None, [label]))
    set_n_threads(1)

    # specify decoding settings
    clf = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("PCA", PCA(n_components=0.95, svd_solver='full')),
            ("RidgeReg", linear_model.Ridge(alpha=10)),
        ]
    )

    # specify sample onsets and window for decoding
    smpon = np.array([0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 2.45, 2.60, 2.75, 2.90, 3.05, 3.20])   # vector of sample onsets (s)
    smpwin = [-0.10001, 1.00001]   # window for decoding, relative to sample onset (s) - going marginally outside desired bnds important to catch all times
    basewin = [-0.050, 0]   # window relative to sample onset for sample-specific baseline (set to None if not required)
    subsmp = 200    # frequency (Hz) at which to subsample (set to None if no subsampling required - original fs = 400 Hz); NB: should be factor of 400

    # load to-be-decoded variables & check that trials are appropiately aligned
    matname = ('/home/btalluri/confirmation_spatial/data/meg/analysis/preprocessed4mne/BehavPupil/%s_4decode.mat' % (subject))
    mat = sio.loadmat(matname)
    sess_idx = []
    for sess in sessions:
        sess_idx.append(np.where(mat['sess'] == sess)[0])
    sess_idx = np.concatenate(sess_idx)
    mat_events = np.int64(np.concatenate(mat["tIDs"]))  # convert matlab events to same type as python events
    assert np.array_equal(events, mat_events[sess_idx])
    # Perform source reconstruction, using for each session the appropriate filter
    # Iterates over sample positions to mitigate memory demands
    all_smp = []  # inialize DataFrame containing all decoding results, across sample positions/variables
    for smp in range(len(smpon)):
        fname = "/home/btalluri/confirmation_spatial/data/meg/analysis/conv2mne/decodeERF_nofilt/%s_%s_%s_cond-%s_finegrainERF.hdf" % (subject, area, str(smp + 1), condition)
        # the try: except: block implements caching, if output is already there, don't do it again.
        try:
            all_s = pd.read_hdf(fname)
        except FileNotFoundError:
            # perform source reconstruction of ERF data
            erfdata, events, times = decoding_ERF.get_lcmv([d[1].copy().crop(smpon[smp] + smpwin[0], smpon[smp] + smpwin[1]) for d in data], filters, njobs=6)    # erfdata is ntrials*nvertices*ntimes
            times = times - smpon[smp]  # making times vector relative to sample onset

            # sample-specific baseline subtraction
            if basewin is not None:
                id_time = (basewin[0] <= times) & (times <= basewin[1])
                means = erfdata[:, :, id_time].mean(-1)
                erfdata -= means[:, :, np.newaxis]

            # subsample
            if subsmp is not None:
                erfdata = erfdata[:, :, 0::round(400 / subsmp)]
                times = times[0::round(400 / subsmp)]

            # loop through variables to be decoded
            all_s = []
            for hemi in [0, 180]:
                hemi_vals = mat["ref_angles"]
                sess_hemi = hemi_vals[sess_idx]
                hemi_idx = np.where(sess_hemi == hemi)[0]
                for target in ["samples"]:
                    # pull variable to be decoded
                    target_vals = mat[target]   # target_vals will be a numpy ndarray, ntrials*nsamples
                    sess_targets = target_vals[sess_idx, :]
                    # perform decoding
                    dcd = decoding_ERF.Decoder(sess_targets[hemi_idx, smp], ("RidgeReg", clf))
                    # erfdata is ntrials*nvertices*ntimes
                    hemi_erfdata = erfdata[hemi_idx, :, :]
                    hemi_events = events[hemi_idx]
                    k = dcd.classify(
                        hemi_erfdata, times, hemi_events, area,   # feeding in times aligned to smp onset
                        average_vertices=False                    # NB, set average_vertices to False if want to preserve vertices as separate features
                    )
                    k.loc[:, "target"] = target  # include target_val label in dataframe
                    k.loc[:, "hemi"] = hemi  # include the hemifield information
                    all_s.append(k)   # append decoding results for this target_val combo

            all_s = pd.concat(all_s)         # concatenate all target_vals
            all_s.loc[:, 'ROI'] = area       # and include ROI/sample position labels
            all_s.loc[:, "sample"] = str(smp + 1)
            all_s.to_hdf(fname, "df")  # save once all target_vals have been iterated over

        all_smp.append(all_s)  # append decoding results for this sample position

    all_smp = pd.concat(all_smp)  # concatenate all sample positions
    all_smp.to_csv("/home/btalluri/confirmation_spatial/data/meg/analysis/conv2mne/decodeERF_nofilt/%s_%s_allSmp_cond-%s_finegrainERF.csv" % (subject, area, condition))

    return all_smp


# @memory.cache
def decode_consistency(subject, area, sessinfo, condition_idx, epoch_type="stimulus", only_glasser=False, BEM="three_layer", debug=False, target="response"):
    # Only show warning and above:
    mne.set_log_level("WARNING")
    pymeglcmv.logging.getLogger().setLevel(logging.INFO)

    set_n_threads(1)
    condition = conditions[condition_idx]
    # Get all labels for this subject
    if subject == 'Pilot04':
        labels = get_labels('fsaverage', only_glasser)
    else:
        labels = get_labels(subject, only_glasser)
    # And keep only the ones that we are interested in (area parameter of this function) -> needs to be in areas_to_labels dict defined above
    labels = [x for x in labels if any([cl for cl in areas_to_labels[area] if cl in x.name])]
    print(labels)

    if len(labels) < 1:
        raise RuntimeError('Expecting at least two labels')

    # Turn individual labels into one big label that contains all vertices that
    # belong to one ROI
    label = labels.pop()
    for l in labels:
        label += l

    print('Selecting this label for area %s:' % area, label)

    # ATTEMPT TAKING SESS/REC INFO & EXISTING FUNCTIONS INTO ACCOUNT
    data = []
    fwds = []
    bems = []
    sources = []
    sessions = []
    low_pass_fs = None   # low-pass filter cutoff; set to None is no filter required
    for sess, rec in sessinfo:
        logging.info("Reading data for %s, sess %i, rec %i " % (subject, sess, rec))
        data_cov, epoch, epoch_filename = get_stim_epoch(subject, sess, rec, lopass=low_pass_fs)
        data.append((data_cov, epoch))  # N.B. data may in fact need to be output currently called 'epochs'

        logging.info("Setting up source space and forward model")
        megdirs = '/home/btalluri/confirmation_spatial/data/meg/raw/'
        if subject == 'Pilot03' and sess == 1:  # naming error with meg files for this subj/sess (labelled 'JBK')
            raw_filename = glob(megdirs + str(subject) + '_' + '*_0' + str(rec) + '.ds')
        else:
            raw_filename = glob(megdirs + str(subject) + '-' + str(sess) + '*_0' + str(rec) + '.ds')

        assert len(raw_filename) == 1
        raw_filename = raw_filename[0]
        trans_filename = glob(trans_dir + '/' + str(subject) + '-' + str(sess) + '_0' + str(rec) + '-trans.fif')[0]
        if subject == 'Pilot04':
            forward, bem, source = get_leadfield('fsaverage', raw_filename, epoch_filename, trans_filename, bem_sub_path='bem_ft')
        else:
            forward, bem, source = get_leadfield(subject, raw_filename, epoch_filename, trans_filename, bem_sub_path='bem_ft')
        fwds.append(forward)
        sessions.append(sess)
        # bems.append(bem)
        # sources.append(source)

    # pull trial IDs
    events = [d[1].events[:, 2] for d in data]
    events = np.hstack(events)

    # Compute LCMV filters for each session
    filters = []
    for (data_cov, epochs), forward in zip(data, fwds):   # PM: N.B. data_cov may not be part of data as currently configured
        filters.append(pymeglcmv.setup_filters(epochs.info, forward, data_cov, None, [label]))
    set_n_threads(1)

    # specify decoding settings
    clf = Pipeline(
        [
            ("Scaling", StandardScaler()),
            ("PCA", PCA(n_components=0.95, svd_solver='full')),
            ("RidgeReg", linear_model.Ridge(alpha=10)),
        ]
    )

    # specify sample onsets and window for decoding
    smpon = np.array([2.45, 2.60, 2.75, 2.90, 3.05, 3.20])   # vector of sample onsets (s)
    smpwin = [-0.10001, 1.00001]   # window for decoding, relative to sample onset (s) - going marginally outside desired bnds important to catch all times
    basewin = [-0.050, 0]   # window relative to sample onset for sample-specific baseline (set to None if not required)
    subsmp = 200    # frequency (Hz) at which to subsample (set to None if no subsampling required - original fs = 400 Hz); NB: should be factor of 400

    # load to-be-decoded variables & check that trials are appropiately aligned
    matname = ('/home/btalluri/confirmation_spatial/data/meg/analysis/preprocessed4mne/BehavPupil/%s_4decode.mat' % (subject))
    mat = sio.loadmat(matname)
    sess_idx = []
    for sess in sessions:
        sess_idx.append(np.where(mat['sess'] == sess)[0])
    sess_idx = np.concatenate(sess_idx)
    mat_events = np.int64(np.concatenate(mat["tIDs"]))  # convert matlab events to same type as python events
    assert np.array_equal(events, mat_events[sess_idx])
    # Perform source reconstruction, using for each session the appropriate filter
    # Iterates over sample positions to mitigate memory demands
    all_smp = []  # inialize DataFrame containing all decoding results, across sample positions/variables
    for smp in range(len(smpon)):
        fname = "/home/btalluri/confirmation_spatial/data/meg/analysis/conv2mne/decodeERF_nofilt/%s_%s_%s_consistency_cond-%s_finegrainERF.hdf" % (subject, area, str(smp + 1), condition)
        # the try: except: block implements caching, if output is already there, don't do it again.
        try:
            all_s = pd.read_hdf(fname)
        except FileNotFoundError:
            # perform source reconstruction of ERF data
            erfdata, events, times = decoding_ERF.get_lcmv([d[1].copy().crop(smpon[smp] + smpwin[0], smpon[smp] + smpwin[1]) for d in data], filters, njobs=6)    # erfdata is ntrials*nvertices*ntimes
            times = times - smpon[smp]  # making times vector relative to sample onset

            # sample-specific baseline subtraction
            if basewin is not None:
                id_time = (basewin[0] <= times) & (times <= basewin[1])
                means = erfdata[:, :, id_time].mean(-1)
                erfdata -= means[:, :, np.newaxis]

            # subsample
            if subsmp is not None:
                erfdata = erfdata[:, :, 0::round(400 / subsmp)]
                times = times[0::round(400 / subsmp)]

            # loop through variables to be decoded
            all_s = []
            for hemi in [0, 180]:
                hemi_vals = mat["ref_angles"]
                sess_hemi = hemi_vals[sess_idx]
                hemi_idx = np.where(sess_hemi == hemi)[0]
                for target in ["consistent_samples", "inconsistent_samples"]:
                    # pull variable to be decoded
                    target_vals = mat[target]   # target_vals will be a numpy ndarray, ntrials*nsamples
                    sess_targets = target_vals[sess_idx, :]
                    # get indices for valid samples
                    valid_target_idx = np.where(np.isfinite(sess_targets[:, smp + 6]))[0]
                    # get trial indices for decoding
                    decoding_trial_idx = np.intersect1d(hemi_idx, valid_target_idx)
                    # perform decoding
                    dcd = decoding_ERF.Decoder(sess_targets[decoding_trial_idx, smp + 6], ("RidgeReg", clf))
                    # erfdata is ntrials*nvertices*ntimes
                    hemi_erfdata = erfdata[decoding_trial_idx, :, :]
                    hemi_events = events[decoding_trial_idx]
                    k = dcd.classify(
                        hemi_erfdata, times, hemi_events, area,   # feeding in times aligned to smp onset
                        average_vertices=False                    # NB, set average_vertices to False if want to preserve vertices as separate features
                    )
                    k.loc[:, "target"] = target  # include target_val label in dataframe
                    k.loc[:, "hemi"] = hemi  # include the hemifield information
                    all_s.append(k)   # append decoding results for this target_val combo

            all_s = pd.concat(all_s)         # concatenate all target_vals
            all_s.loc[:, 'ROI'] = area       # and include ROI/sample position labels
            all_s.loc[:, "sample"] = str(smp + 7)
            all_s.to_hdf(fname, "df")  # save once all target_vals have been iterated over

        all_smp.append(all_s)  # append decoding results for this sample position

    all_smp = pd.concat(all_smp)  # concatenate all sample positions
    all_smp.to_csv("/home/btalluri/confirmation_spatial/data/meg/analysis/conv2mne/decodeERF_nofilt/%s_%s_allSmp_consistency_cond-%s_finegrainERF.csv" % (subject, area, condition))

    return all_smp

# # code for reading already created individual sample files to csv
# subject = "ECB"
# area = area = "vfcPrimary"
# for alphaRR in [1]:
#     all_smp = []
#     for smp in [0]:
#         fname = "/home/pmurphy/Surprise_accumulation/Analysis/MEG/Conv2mne/decode/%s_%s_%s_%s_finegrainTF_nophase.hdf" % (subject, area, str(smp+1), str(alphaRR))
#         all_s = pd.read_hdf(fname)
#         all_smp.append(all_s)
#     all_smp = pd.concat(all_smp)  # concatenate all sample positions
#     all_smp.to_csv("/home/pmurphy/Surprise_accumulation/Analysis/MEG/Conv2mne/decode/%s_%s_%s_full_finegrainTF_nophase.csv" % (subject, area, str(alphaRR)))
