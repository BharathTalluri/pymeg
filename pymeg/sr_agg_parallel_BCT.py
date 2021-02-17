
subjects = {'Pilot04': 4}
#            'Pilot06': 4}


def submit_aggregates(cluster='uke'):
    from pymeg import parallel
    for subject, final_sess in subjects.items():
        if subject == 'Pilot06':
            first_sess = 2
        else:
            first_sess = 1
        for sessnum in range(first_sess, final_sess + 1):
            for datatype in ['F']:  # for datatype in ['F','BB']:
                parallel.pmap(aggregate, [(subject, sessnum, datatype)],
                              name='agg' + str(sessnum) + '-' + str(subject) + datatype,
                              tasks=6, memory=60, nodes=1, walltime='30:00:00', ssh_to='node028', env='py36')  # 3,30 for G, 5,50 for F,BB


def aggregate(subject, session, datatype):
    from pymeg import aggregate_sr as asr
    from os.path import join
    data = ('/home/btalluri/confirmation_spatial/data/meg/analysis/conv2mne/%s-SESS%i-*%s*-lcmv.hdf' % (subject, session, datatype))

    if datatype == 'F':    # time-frequency
        agg = asr.aggregate_files(data, data, (-0.4, -0.2), to_decibels=True)
    elif datatype == 'BB':    # broadband
        agg = asr.aggregate_files(data, data, (-0.2, 0), to_decibels=False)
    elif datatype == 'G':    # gamma (optimized)
        agg = asr.aggregate_files(data, data, (-0.1, -0.05), to_decibels=True)

    filename = join('/home/btalluri/confirmation_spatial/data/meg/analysis/conv2mne/agg/',
                    'S%s_SESS%i_%s_agg.hdf' % (subject, session, datatype))
    asr.agg2hdf(agg, filename)


if __name__ == "__main__":
    submit_aggregates()
