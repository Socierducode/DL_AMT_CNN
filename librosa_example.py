# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:19:13 2017

@author: wyc

Audio/CQT/midi parser and viz

"""

import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import re
#import midi_to_mat as mm
import pretty_midi

audio_path='/home/wyc/Desktop/CNN_AMT/data/train/SequenceTest_POLY_0010.wav'
# Set the parameters
RangeMIDInotes=[21,108]
sr=44100
bins_per_octave=36
n_octave=7

###################
#   Audio Load    #
###################
def audio_load(audio_path,plot=False):
	y, _ = lb.load(audio_path, sr=sr)
	if plot:
		plt.figure()
		lb.display.waveplot(y,sr=sr)
		plt.title('WaveForm of the example audio')
	return y
###################
#   STFT module   #
###################    
def stft_module(y,plot=False):
	stft_spectrum=lb.stft(y, n_fft=1024,hop_length=512,center=True, dtype=np.complex64)
	stft=np.abs(stft_spectrum)   #compute the amplitude
	if plot: #For testing
		plt.figure()
		print stft.shape  # =(1 + n_fft/2, t), t=431 if hop_length=512
		plt.subplot(211)
		# plt.imshow(stft)
		# plt.colorbar(format='%+2.0f dB')
		plt.plot(stft[:, 100])  # Plot a single frame
		plt.title('100th frame')
		plt.subplot(212)
		# Note:lb display must be just put after the other plots
		lb.display.specshow(lb.amplitude_to_db(stft), sr=sr, fmin=lb.midi_to_hz(RangeMIDInotes[0]), x_axis='time', y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		plt.title('STFT_Linear-frequency power spectrogram')
	return stft

##################
#   CQT module   #
##################

#Reference code is located in librosa/core/constantq/def cqt
#Reference article: "Constant-Q transform toolbox for music processing".
#Real= True result is slightly different from the abs(a) of Real=false result: e-17
# Freq axis: By default sr= 22050, bins_per_octave=12,n_bins=84 (max=112/per octave),fmin=C1(not good for a 88-key piano) 
# Time resolution: 5s/431frames=0.0116: hop_length=512 (by default)

# delta f_min=27.5*(2**(1.0/60))-27.5=0.3195346083028703
# Beneto's code: fmin=27.5(A0), fmax=rs/2, we can't define a fmax here so we choose the max n_bins
# Trick: Beneto's CQT spectrogram method, reduce the high-frequency deform of low-pass filter from the raw audio
def cqt_module(y,sr=sr,bins_per_octave=bins_per_octave,n_octave=n_octave,range=RangeMIDInotes,plot=False):
	CQT_spectrum=lb.cqt(y,sr=sr,bins_per_octave=bins_per_octave,n_bins=n_octave*bins_per_octave,fmin = lb.midi_to_hz(range[0]),real=False)
	CQT = np.abs(CQT_spectrum)
	if plot:
		CQT.shape
		CQT_energy=np.sum(CQT,axis=0)
		plt.figure()
		plt.subplot(311)
		plt.plot(CQT[:, 300])
		plt.title('300th frame')
		plt.subplot(312)
		plt.plot(CQT_energy)
		plt.title('Energy')
		plt.subplot(313)
		lb.display.specshow(lb.logamplitude(CQT_spectrum[0:500, :] ** 2, ref_power=np.max),
						sr=sr, bins_per_octave=bins_per_octave, fmin=lb.midi_to_hz(range[0]), x_axis='time', y_axis='cqt_note')
		#plt.colorbar(format='%+2.0f dB')
		plt.title('Constant-Q power spectrum')
		plt.tight_layout()
	return CQT

# # Abs value viz
# lb.display.specshow(CQT,sr=sr,bins_per_octave=36,fmin=lb.note_to_hz('A0'), x_axis='time', y_axis='cqt_note')
# plt.colorbar(format='%+2.0f')
# plt.title('Constant-Q spectrum')
# plt.tight_layout()

# SNR viz: ref to the max power(in CQT ref code)


#Output definition(y)
##############################
#   MIDI processing module   #
##############################

#Corresponding midi_path to the
def midi_module(audio_path,y,CQT,sr,plot=True):
	midi_path=re.sub(r'.wav', '.mid', audio_path)
	n_frames=CQT.shape[1]
# Output definition(y)
	#Ground_truth_mat=mm.midi2mat(midi_path,len(y),n_frames,sr)[0]
	midi_data = pretty_midi.PrettyMIDI(midi_path)
	pianoRoll = midi_data.instruments[0].get_piano_roll(fs=CQT.shape[1] * 44100. / len(y))
	Ground_truth_mat = (pianoRoll[RangeMIDInotes[0]:RangeMIDInotes[1] + 1, :CQT.shape[1]] > 0)
	if plot:
		plt.figure()
		plt.subplot(211)
		lb.display.specshow(Ground_truth_mat,sr=sr,bins_per_octave=12,fmin=lb.note_to_hz('A0'), x_axis='time', y_axis='cqt_note')

# Label distribution in the sequence
		plt.subplot(212)
		n_pitch_frame=np.sum(Ground_truth_mat,axis=1)
		plt.bar(range(RangeMIDInotes[0],RangeMIDInotes[1]+1),n_pitch_frame/np.sum(n_pitch_frame).astype(np.float))
		plt.xticks(range(RangeMIDInotes[0],RangeMIDInotes[1]+1,12),
           	lb.midi_to_note(range(RangeMIDInotes[0],RangeMIDInotes[1]+1,12)))
		plt.xlabel('Midi note')
		plt.ylabel('Note probability')
	return Ground_truth_mat #(88, 10979)


if __name__=='__main__':
	y=audio_load(audio_path,plot=True)
	CQT=cqt_module(y, sr=sr, bins_per_octave=bins_per_octave, n_octave=n_octave, range=RangeMIDInotes, plot=True)
	pianoroll=midi_module(audio_path,y,CQT,sr,plot=True)




#######################
#Hand-made code test ##
#######################
# import pretty_midi
# import librosa as lb
# 	audio_path='/home/wyc/Desktop/data/train/MAPS_MUS-alb_esp2_AkPnCGdD.wav'
# 	midi_path='/home/wyc/Desktop/data/train/MAPS_MUS-alb_esp2_AkPnCGdD.mid'
# 	midi_data = pretty_midi.PrettyMIDI(midi_path)
# 	pianoRoll = midi_data.instruments[0].get_piano_roll(fs=CQT.shape[1]*44100./len(y))  #[128,10806]
# 	plt.figure()
# 	plt.subplot(311)
# 	lb.display.specshow(Ground_truth_mat,sr=sr,bins_per_octave=12,fmin=lb.note_to_hz('A0'), x_axis='time', y_axis='cqt_note')
# 	plt.subplot(312)
# 	lb.display.specshow((pianoRoll[RangeMIDInotes[0]:RangeMIDInotes[1]+1,:Ground_truth_mat.shape[1]]>0), sr=44100, bins_per_octave=12, fmin=lb.note_to_hz('A0'), x_axis='time',
# 					y_axis='cqt_note')
# 	plt.subplot(313)
# 	lb.display.specshow(((pianoRoll[RangeMIDInotes[0]:RangeMIDInotes[1]+1,:Ground_truth_mat.shape[1]]>0)==Ground_truth_mat), sr=44100, bins_per_octave=12, fmin=lb.note_to_hz('A0'), x_axis='time',
# 					y_axis='cqt_note')