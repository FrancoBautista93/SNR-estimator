import numpy as np
import librosa
import csv
import os
import glob

################################ Load Signals ################################
filenames=[]
SNRs=[]

for filename in glob.glob(os.path.join('audios_folder','*.wav')):
    filenames.append(os.path.basename(filename))

for i in filenames:
    signal, sr = librosa.load('audios_folder'+i, sr=None)
    total_time=librosa.get_duration(signal, sr=sr)

############################## VAD's timestamps ##############################

    segments = open('model_folder'+ os.path.splitext(i)[0] + '.txt','r')
    linessegments=segments.readlines()
    start_time=[]
    end_time=[]
    n_start_time=[]
    n_end_time=[]

    for x in linessegments:
        start_time.append(x.split(' ')[2])
        end_time.append(x.split()[3])
    segments.close()

    start_time=[float(i) for i in start_time]
    end_time=[float(i) for i in end_time]

    voice_regions_timestamps=[start_time, end_time]

# No voice regions

    if start_time[0]==0.0:
        n_start_time.append(end_time[0])
        for i in end_time[1:]:
            n_start_time.append(i)
    else:
        n_start_time.append(0.0)
        for i in end_time:
            n_start_time.append(i)

    if end_time[-1]==total_time:
        for i in start_time[:-1]:
            n_end_time.append(i)
        n_end_time.append(start_time[-1])
    else:
        for i in start_time:
            n_end_time.append(i)
        n_end_time.append(total_time)

    if start_time[0]==0.0:
        n_end_time.pop(0)

    if end_time[-1]==total_time:
        n_start_time.pop(-1)
    
    no_voice_regions_timestamps=[n_start_time,n_end_time]

############################# Energy calculation #############################

    energy=np.abs(signal**2)

    samples_voice=[]
    samples_no_voice=[]

    for i in range(len(start_time)):   
        samples_voice.append(librosa.time_to_samples(np.array([start_time[i],end_time[i]]),sr=sr))
    
    for i in range(len(n_start_time)):   
        samples_no_voice.append(librosa.time_to_samples(np.array([n_start_time[i],n_end_time[i]]),sr=sr))

    for i in range(len(samples_no_voice)):
        samples_no_voice[i][0]=samples_no_voice[i][0]+1

    for i in range(len(samples_no_voice)):
       samples_no_voice[i][-1]=samples_no_voice[i][-1]-1
    
    if start_time[0]==0.0:
        samples_voice[0][0]=0
    
    if end_time[-1]==total_time:
        samples_voice[-1][-1]=(total_time*sr)
 
    voice=[]
    no_voice=[]
 
    for i in samples_voice:
        voice.append(energy[i[0]:i[1]])

    voice=np.hstack(voice)

    for i in samples_no_voice:
        no_voice.append(energy[i[0]:i[1]])
    
    no_voice=np.hstack(no_voice)

############################### SNR estimation ###############################

    voice_mean=np.mean(voice)
    no_voice_mean=np.mean(no_voice)

    SNR = 10*np.log10(abs(voice_mean-no_voice_mean)/no_voice_mean)
    SNRs.append(round(SNR,2))

headers=['Filename','SNR (dB)']

with open('snr.csv', 'w', newline="") as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(headers)
    writer.writerows(zip(filenames, SNRs))
