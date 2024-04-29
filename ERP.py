import mne 
from mne import events_from_annotations
import numpy as np
import matplotlib.pyplot as plt

ss01_raw_path = '/Users/user/Desktop/SPHSC525/EEG/SS01/SS01-SR-02082016-cnt.cnt'
ss01_raw = mne.io.read_raw_cnt(ss01_raw_path, preload = True)
# Convert annotations to events
events, event_dict = events_from_annotations(ss01_raw)
#events = mne.find_events(ss01_raw, stim_channel="STI 014"); weird, they don't have STI 014 channel

# Plot events on raw data to visualize event markers
mne.viz.plot_events(events, sfreq=ss01_raw.info['sfreq']) #it's weird

# Print event dictionary (mapping of event names to event codes)
print(event_dict)

print(events[:30])  # it's weird; it's not properly coded

baseline = (None, 0)  # Baseline correction period (from start of epoch to 0 seconds)
epochs = mne.Epochs(ss01_raw, events, event_id={'SentenceOnset': 1}, tmin=-1.0, tmax=1.5, baseline= baseline, preload=True)

# Apply independent component analysis (ICA) to remove artifacts
ica = mne.preprocessing.ICA(n_components=20, random_state=42)
ica.fit(epochs)

# Plot ICA components
ica.plot_components()

# Manually select components for exclusion based on artifact patterns
# Replace 'ica.exclude' with indices of components to exclude (e.g., [0, 1, 2])
ica.exclude = [0, 1, 2]
epochs_cleaned = epochs.copy().apply_ica(ica)

# Re-reference epochs to the average of the mastoids (M1, M2)
epochs_cleaned.set_eeg_reference('average', projection=True)

# Apply baseline correction (using the 200 ms before the time-locking point as baseline)
epochs_cleaned.apply_baseline(baseline=(-0.2, 0), mode='mean')

# Apply low-pass filter to the dataset
epochs_cleaned.filter(l_freq=None, h_freq=35, method='iir', verbose=True)

# Remove epochs with remaining artifacts
reject_criteria = dict(eeg=1000e-6, reject=dict(eeg=4))  # Artifact rejection criteria
epochs_cleaned.drop_bad(reject=reject_criteria)

# Average epochs within sentence types for each participant
epochs_average = epochs_cleaned.average()

# Plot grand average ERPs across participants
epochs_average.plot_joint()

# Save preprocessed epochs and average data
epochs_cleaned.save('/Users/user/Desktop/SPHSC525/EEG/SS01/preprocessed_epochs.fif')  # Save preprocessed epochs
epochs_average.save('/Users/user/Desktop/SPHSC525/EEG/SS01/average_epochs.fif')        # Save average epochs

# Load preprocessed epochs data for one participant (replace 'path_to_epochs_file.fif' with your file path)
epochs = mne.read_epochs('path_to_epochs_file.fif')

# Select data for Cz electrode
electrode = 'Cz'
epochs_cz = epochs.copy().pick_channels([electrode])

# Plot ERP for syntactically-acceptable sentences (condition: 1)
condition_acceptable = 'syntactically-acceptable'
epochs_acceptable = epochs_cz[condition_acceptable]
erp_acceptable = epochs_acceptable.average()

# Plot ERP for sentences with syntax errors (condition: 2)
condition_error = 'syntax-error'
epochs_error = epochs_cz[condition_error]
erp_error = epochs_error.average()

# Plot individual ERPs for one participant at Cz electrode
plt.figure(figsize=(10, 6))
plt.plot(erp_acceptable.times, erp_acceptable.data[0], label='Syntactically Acceptable', color='blue')
plt.plot(erp_error.times, erp_error.data[0], label='Syntax Error', color='red')
plt.axvline(x=0, color='k', linestyle='--')  # Mark onset of error at time 0
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.title(f'ERP at {electrode} Electrode for Participant')
plt.legend()
plt.show()

# Plot average ERPs for syntactically-acceptable sentences vs sentences with syntax errors
plt.figure(figsize=(10, 6))
plt.plot(erp_acceptable.times, erp_acceptable.data[0], label='Syntactically Acceptable', color='blue')
plt.plot(erp_error.times, erp_error.data[0], label='Syntax Error', color='red')
plt.axvline(x=0, color='k', linestyle='--')  # Mark onset of error at time 0
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.title(f'Average ERPs at {electrode} Electrode')
plt.legend()
plt.show()