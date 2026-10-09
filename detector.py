import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#1 Load audio file
file_name = "SOCAL32E_sitE_090420_000345.x.flac"
audio_data, sample_rate = sf.read(file_name, frames=10000000)

#2 Apply the High-Pass Filter
cutiff_freq = 20000 # Cut out everything below 20 kHz
nyquist = sample_rate / 2
normalized_cutoff = cutiff_freq / nyquist

# Build and apply the mathematical filter
b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
filtered_audio = signal.filtfilt(b, a, audio_data)

#2 Print out the basic statistics
print(f"Sample rate: {sample_rate} Hz ")
print(f"Total audio samples: {len(audio_data)}")


#3 Draw the Spectrogram
plt.figure(figsize=(10, 4))
plt.specgram(filtered_audio, Fs=sample_rate, cmap='viridis', vmin=-150, vmax=-80)
plt.title("Ocean Acoustic Recording Spectrogram")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [seconds]")
plt.colorbar(label="Intensity [dB]")

#3 Calculate the energy of the signal
signal_energy = filtered_audio ** 2
#4 Calculate the detection threshold
average_energy = np.mean(signal_energy)
std_dev = np.std(signal_energy)
# We will define a spike as anything that is at least 100 standard devations louder than the average
threshold = average_energy + (80 * std_dev)
print(f"Detection threshold: {threshold}")

#5 Find the clicks
# Search the array for values greater than our threshold
click_indices = np.where(signal_energy > threshold)[0]

print(f"Number of data points above threshold: {len(click_indices)}")

#6 Group the points into distinct click events
# Find the distance between consecutive flagged points
gaps = np.diff(click_indices)

# If the gap is larger than 100 data points (0.5ms), count it as a new event boundary
new_event_boundaries = np.where(gaps > 100)[0]

#The total number of events is the number of boundaries + 1 (for the first event)
total_events = len(new_event_boundaries) + 1
print(f"Total distinct acoustic events detected: {total_events}")

#7 Convert click locations to seconds and plot them
# Get the exact sample number for the start of each click event
click_sample_numbers = click_indices[new_event_boundaries]
# Divide by sample rate to get the time in seconds
click_times_seconds = click_sample_numbers / sample_rate
# Draw a red vertical line for each click on the spectrogram
for t in click_times_seconds:
    plt.axvline(x=t, color='red', alpha=0.3, linewidth=1)



plt.show() # This commands Python to open the image window