import mne
from mne import events_from_annotations
import numpy as np
import matplotlib.pyplot as plt
import glob

ss01_raw_path = '/Users/user/Desktop/SPHSC525/EEG/SS01/SS01-SR-02082016-cnt.cnt'
ss01_raw = mne.io.read_raw_cnt(ss01_raw_path, preload = True)
# Convert annotations to events
events, event_dict = events_from_annotations(ss01_raw)
#events = mne.find_events(ss01_raw, stim_channel="STI 014")

print(events)  # show the first 5

# Plot events on raw data to visualize event markers
mne.viz.plot_events(events, sfreq=ss01_raw.info['sfreq'])

# Print event dictionary (mapping of event names to event codes)
print(event_dict)

print(events[:30])  # show the first 5


# Manually define event IDs for congruent and incongruent trials
event_id = {'203': 1, '207': 2}

# Check if the specified event IDs are present in the event_dict
if all(key in event_dict for key in event_id.keys()):
    # Create epochs around congruent and incongruent events with baseline correction
    epochs_congruent = mne.Epochs(ss01_raw, events, event_id=event_id['203'], tmin=-1.0, tmax=2.5, baseline=(-1.0, 0), preload=True)
    epochs_incongruent = mne.Epochs(ss01_raw, events, event_id=event_id['207'], tmin=-1.0, tmax=2.5, baseline=(-1.0, 0), preload=True)

    # Get the Cz channel index
    cz_index = epochs_congruent.info['ch_names'].index('Cz')

    # Get the average ERP for congruent and incongruent trials at Cz
    erp_congruent = epochs_congruent.average(picks=[cz_index]).data[0] * 1e6  # Convert to microvolts
    erp_incongruent = epochs_incongruent.average(picks=[cz_index]).data[0] * 1e6  # Convert to microvolts

    # Create time array for x-axis
    times = epochs_congruent.times

    # Plot the ERP
    plt.figure(figsize=(8, 6))
    plt.plot(times, erp_congruent, label='Sentences without errors')
    plt.plot(times, erp_incongruent, label='Sentences with errors')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title('ERP at Cz for Participant ss01')
    plt.axvline(x=0, color='black', linestyle='--', label='Error onset')
    plt.ylim(np.min([erp_congruent, erp_incongruent]), np.max([erp_congruent, erp_incongruent]))  # Adjust y-axis limits based on data
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Specified event IDs not found in the event dictionary.")
    


#group average
participant_dirs = glob.glob('/Users/user/Desktop/SPHSC525/EEG/SS*')
event_id = {'203': 1, '207': 2}

erp_congruent_all = []
erp_incongruent_all = []

for participant_dir in participant_dirs:
    cnt_file = glob.glob(f'{participant_dir}/*-cnt.cnt')[0]
    raw = mne.io.read_raw_cnt(cnt_file, preload=True)

    # Convert annotations to events
    events, event_dict = events_from_annotations(raw)

    # Check if the specified event IDs are present in the event_dict
    if all(key in event_dict for key in event_id.keys()):
        # Create epochs around congruent and incongruent events with baseline correction
        epochs_congruent = mne.Epochs(raw, events, event_id=event_id['203'], tmin=-1.0, tmax=2.5, baseline=(-1.0, 0), preload=True)
        epochs_incongruent = mne.Epochs(raw, events, event_id=event_id['207'], tmin=-1.0, tmax=2.5, baseline=(-1.0, 0), preload=True)

        # Get the Cz channel index
        cz_index = epochs_congruent.info['ch_names'].index('Cz')

        # Get the average ERP for congruent and incongruent trials at Cz
        erp_congruent = epochs_congruent.average(picks=[cz_index]).data[0] * 1e6  # Convert to microvolts
        erp_incongruent = epochs_incongruent.average(picks=[cz_index]).data[0] * 1e6  # Convert to microvolts

        erp_congruent_all.append(erp_congruent)
        erp_incongruent_all.append(erp_incongruent)

    else:
        print(f"Specified event IDs not found in the event dictionary for participant {participant_dir}.")

# Create time array for x-axis
times = epochs_congruent.times

# Calculate the average ERP across all participants
erp_congruent_avg = np.mean(erp_congruent_all, axis=0)
erp_incongruent_avg = np.mean(erp_incongruent_all, axis=0)

# Plot the average ERP
plt.figure(figsize=(8, 6))
plt.plot(times, erp_congruent_avg, label='Sentences without errors')
plt.plot(times, erp_incongruent_avg, label='Sentences with errors')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Average ERP at Cz for All Participants')
plt.axvline(x=0, color='black', linestyle='--', label='Error onset')
plt.ylim(np.min([erp_congruent_avg, erp_incongruent_avg]), np.max([erp_congruent_avg, erp_incongruent_avg]))  # Adjust y-axis limits based on data
plt.legend()
plt.grid()
plt.show()




