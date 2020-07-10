# Copyright (c) 2020 Herman Tarasau
# # Pre-processing
# coding: utf-8


import mne
import numpy as np
from scipy.signal import stft
import pickle
from pathlib import Path

from utils import IAF, ERD, CONFIG, electrodes, WAVES, remove_outliers

load_dir = Path(__file__).parent.parent / "data"
obg_dir = Path(__file__).parent.parent / "obj_dumps"

curr_dir = load_dir / "subject1" / "music"
cal_name = "NabilaHosny.Music.EyesClosed.edf"
exp_name = "NabilaHosny.Music.FirstTask.edf"

print(cal_name)
raw_cal = mne.io.read_raw_edf(cal_name, preload=True)
raw_exp = mne.io.read_raw_edf(exp_name, preload=True)
sfreq = raw_exp.info['sfreq']

        # Data from specific channels
eyes = raw_cal.copy().pick_channels(ch_names=electrodes["central"])
experiment = raw_exp.copy().pick_channels(ch_names=electrodes["central"])

        # Filtering AC line noise with notch filter

eyes_filtered_data = mne.filter.notch_filter(x=eyes.get_data(), Fs=sfreq, freqs=[50, 100])
experiment_filtered_data = mne.filter.notch_filter(x=experiment.get_data(), Fs=sfreq, freqs=[50, 100])
        # eyes_filtered_data = eyes.get_data()
        # experiment_filtered_data = experiment.get_data()
        # Preparing data for plotting
eyes_filtered = mne.io.RawArray(data=eyes_filtered_data,
                                info=mne.create_info(ch_names=electrodes["central"], sfreq=sfreq))
experiment_filtered = mne.io.RawArray(data=experiment_filtered_data,
                                info=mne.create_info(ch_names=electrodes["central"], sfreq=sfreq))

IAF_p = IAF(CONFIG["subjects"]["subject1"])

        # Getting L1A, L2A, UA, Theta waves from eyes closed using FIR filtering. Also we take mean signal from all
        # channels
eyes_sub_bands = {
    'L1A': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 4,
                                    h_freq=IAF_p - 2, sfreq=sfreq, method="fir"),
    'L2A': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 2,
                                    h_freq=IAF_p, sfreq=sfreq, method="fir"),
    'UA': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p,
                                    h_freq=IAF_p + 2, sfreq=sfreq, method="fir"),
    'Th': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0), l_freq=IAF_p - 6,
                                    h_freq=IAF_p - 4, sfreq=sfreq, method="fir"),
    'Beta': mne.filter.filter_data(data=np.mean(eyes_filtered.get_data(), axis=0),
                                    l_freq=IAF_p + 2,
                                    h_freq=30, sfreq=sfreq, method="fir")}

        # Getting L1A, L2A, UA, Theta waves from experiment data using FIR filtering. Also we take mean signal from all
        # channels
experiment_sub_bands = {'L1A': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                        l_freq=IAF_p - 4, h_freq=IAF_p - 2, sfreq=sfreq,
                                                         method="fir"),
                        'L2A': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                        l_freq=IAF_p - 2, h_freq=IAF_p, sfreq=sfreq,
                                                        method="fir"),
                        'UA': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                        l_freq=IAF_p,
                                                        h_freq=IAF_p + 2, sfreq=sfreq, method="fir"),
                        'Th': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                        l_freq=IAF_p - 6, h_freq=IAF_p - 4, sfreq=sfreq,
                                                        method="fir"),
                        'Beta': mne.filter.filter_data(data=np.mean(experiment_filtered.get_data(), axis=0),
                                                        l_freq=IAF_p + 2, h_freq=30, sfreq=sfreq,
                                                        method="fir")}

energies_EyeClosed = {'L1A': np.mean(eyes_sub_bands['L1A']),'L2A': np.mean(eyes_sub_bands['L2A']), 'UA':np.mean(eyes_sub_bands['UA']), 'Th':np.mean(eyes_sub_bands['Th']), 'Beta': np.mean(eyes_sub_bands['Beta']) }
print(energies_EyeClosed)
energies_Exp = {'L1A': np.mean(experiment_sub_bands['L1A']),'L2A': np.mean(experiment_sub_bands['L2A']), 'UA':np.mean(experiment_sub_bands['UA']), 'Th':np.mean(experiment_sub_bands['Th']), 'Beta': np.mean(experiment_sub_bands['Beta'])}
print(energies_Exp)

