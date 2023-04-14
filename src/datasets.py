import torch
from torch.utils.data import Dataset
from helper_code import *
import numpy as np, os, sys
from tqdm import tqdm
import librosa

class patientDataset(Dataset):
    def __init__(self, data_folder):
        
        self.no_of_segments = 4
        
        self.data_folder  = data_folder
        self.patient_ids  = find_data_folders(data_folder)
        self.num_patients = len(self.patient_ids)
        
    def getMetadata(self, idx):
        # Load data.
        patient_id = self.patient_ids[idx]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(self.data_folder, patient_id)
        
        # Define file location.
        patient_metadata_file = os.path.join(self.data_folder, patient_id, patient_id + '.txt')
        recording_metadata_file = os.path.join(self.data_folder, patient_id, patient_id + '.tsv')

        # Load non-recording data.
        patient_metadata = load_text_file(patient_metadata_file)
        recording_metadata = load_text_file(recording_metadata_file)
        
        return patient_metadata, recording_metadata
        
    def __getitem__(self, index):
        
        patient_metadata, recording_metadata = self.getMetadata(index)
        
        # Load recordings.
        recording_ids = list(get_recording_ids(recording_metadata))
        
        recording_locations = []
        for recording_id in reversed(recording_ids):
            if recording_id != 'nan':
                recording_location = os.path.join(self.data_folder, self.patient_ids[index], recording_id)
                recording_locations.append(recording_location)
            
            if len(recording_locations) >= self.no_of_segments:
                break
        
        outcomes = self.no_of_segments*[get_outcome(patient_metadata) ]
        
        return recording_locations, outcomes
        
    def __len__(self):
        return self.num_patients
        
        
class eegDataset(Dataset):
    def __init__(self, patientDataset, idxs):
        
        self.channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 
                         'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
                         'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
        
        self.l_cutoff = 0.1
        self.h_cutoff = 30
        self.order = 5
        
        self.final_sr = 2*self.h_cutoff
        
        self.st_time = 5
        self.end_time = 295
        self.hop_time = 10
        self.segment_length = 20
        
        self.recordings = []
        self.outcomes   = []
        
        for idx in tqdm(idxs):
            self.addPatient(patientDataset, idx)
        
    def bandpassFilter(self, signal, sampling_rate):
        nyq = 0.5 * sampling_rate
        normal_l_cutoff = self.l_cutoff / nyq
        normal_h_cutoff = self.h_cutoff / nyq

        # get the filter coefficients
        b, a = sp.signal.butter( self.order, [normal_l_cutoff, normal_h_cutoff], btype='bp', analog=False)
        filtered_signal = sp.signal.filtfilt(b, a, signal)
        return filtered_signal
        
    def addRecording(self, recording_data, sr, outcome):
        filtered_data  = self.bandpassFilter(recording_data, sr)
        resampled_data = librosa.resample(y=filtered_data, orig_sr=sr, target_sr=self.final_sr)
        
        for st_time in range(self.st_time, self.end_time, self.hop_time):
            end_time = st_time+self.segment_length
            if end_time > self.end_time:
                continue
                
            st_mkr, end_mkr = st_time*self.final_sr, end_time*self.final_sr
            recording_segment = resampled_data[:, st_mkr:end_mkr]
            mx_arr = np.max(np.abs(recording_segment) , axis=1)
            mx_arr[mx_arr==0] = 1
#             mx_arr = np.where(mx_arr == 0, 1, mx_arr)
            recording_segment = (recording_segment.T/mx_arr ).T
            
            self.recordings.append(recording_segment)
            self.outcomes.append(outcome)
        
    def addPatient(self, patientDataset, idx):
        recording_locations, outcomes = patientDataset[idx]
        
        for i in range(len(recording_locations)):
            recording_location, outcome = recording_locations[i], outcomes[i]
            recording_data, sampling_frequency, channels = load_recording(recording_location)
            recording_data = reorder_recording_channels(recording_data, channels, self.channels)
            self.addRecording(recording_data, sampling_frequency, outcome)
        
    def __getitem__(self, index):
        return self.recordings[index], self.outcomes[index]
        
    def __len__(self):
        return len(self.recordings)