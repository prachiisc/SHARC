##################################################################################################################
# A script for VAD benchmarking 
################
# VBx VAD      #
################
# TDNN DIHARD3 #
################
# Silero VAD   #
################
# WEB RTC VAD  #
################
# Pyannote VAD #
################
##################################################################################################################
import torch
from pyannote.audio.pipelines import VoiceActivityDetection
import math
import logging
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import os
import argparse 
import subprocess
import pickle
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pyannote.audio import Model
from pdb import set_trace as bp

import warnings
warnings.filterwarnings("ignore")

# . utils/parse_options.sh

# pyannote_pretrained_model= sys.argv[1] #vad_benchmarking/VAD_model/pytorch_model.bin

def silero_vad(args,input_wav):
	model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
		                      model='silero_vad',
		                      force_reload=True)
	(get_speech_timestamps,save_audio,read_audio,VADIterator,collect_chunks) = utils
	#print(utils)
	sampling_rate = 16000 
	wav = read_audio(input_wav, sampling_rate=sampling_rate)
	nfile=input_wav.split("/")[-1].split(".")[0]	
	#print(input_wav.split("/")[-1].split(".")[0])
	speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
	#print("".join(speech_timestamps))
	# with open("silero_vad"+nfile+".pkl",'wb') as f:	
	# 	pickle.dump(speech_timestamps,f)
	with open(f"{args.outputpath}/{nfile}.pkl",'wb') as f:
		pickle.dump(speech_timestamps,f)

def VBxVAD(input_wav):
	#print(input_wav.split("/")[-1].split(".")[0])
	subprocess.run(["sh","/home1/somil/VBx-VAD/VB_VAD.sh",input_wav])
		
def pyaanote_vad(args,input_wav,dataset,HYPER_PARAMETERS=None):
	nfile=input_wav.split("/")[-1].split(".")[0]

	model = Model.from_pretrained(args.pyannote_pretrained_model)
	# model = model.to('cuda:0')
	pipeline = VoiceActivityDetection(segmentation=model)
	if HYPER_PARAMETERS is None:
		if "ami" in args.dataset:
			HYPER_PARAMETERS = {"min_duration_off": 0.0979,
			"min_duration_on": 0.0554,
			"offset": 0.4807,
			"onset": 0.8104 }
		else:
			HYPER_PARAMETERS = {
			# onset/offset activation thresholds
			"onset": 0.5, "offset": 0.5,
			# remove speech regions shorter than that many seconds.
			"min_duration_on": 0.0,
			# fill non-speech regions shorter than that many seconds.
			"min_duration_off": 0.0
			}
	# pipeline = pipeline.to(0)
	pipeline.instantiate(HYPER_PARAMETERS)
	
	vad = pipeline(input_wav)

	onset=HYPER_PARAMETERS['onset']
	offset = HYPER_PARAMETERS['offset']
	min_duration_on = HYPER_PARAMETERS["min_duration_on"]
	min_duration_off = HYPER_PARAMETERS["min_duration_off"]

	with open(f"{args.outputpath}/{nfile}.pkl",'wb') as f:
		pickle.dump(vad,f)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-audio', type=str, help="Input audio file")
	parser.add_argument('--in-VAD', type=str, help="Input vad TYPE: 1. VBx_VAD | 2. silero_VAD | 3. Pyannote_VAD ")
	parser.add_argument('--dataset',type=str,help="DISPLACE | AMI | Voxconverse",default="DISPLACE")
	parser.add_argument('--onset',type=float)
	parser.add_argument('--offset',type=float)
	parser.add_argument('--min_duration_on',type=float)
	parser.add_argument('--min_duration_off',type=float)
	parser.add_argument('--tuning',action='store_true') # if tuning is required otherwise use default 
	parser.add_argument('--outputpath', type=str, help="output pickle file")
	parser.add_argument('--pyannote_pretrained_model', type=str, help="pyannote pretrained model")
	args = parser.parse_args()		
	wavscp=args.in_audio
	vad_type=args.in_VAD
	
	if vad_type=="VBx_VAD":
		VBxVAD(input_wav)
	elif vad_type=="silero_VAD":
		readscp = np.genfromtxt(wavscp,dtype=str)
		
		if len(readscp.shape) == 1: 
			m = readscp.shape[0]
			column = min(2,m-1)
			wavfiles= readscp.reshape(-1,m)[:,column]
		else:
			column = min(2,readscp.shape[1]-1)
			wavfiles= readscp[:,column]

		for input_wav in wavfiles:
			nfile=input_wav.split("/")[-1].split(".")[0]
			if not os.path.exists(f"{args.outputpath}/{nfile}.pkl"):
				silero_vad(args,input_wav)
	elif vad_type=="Pyannote_VAD":
		if args.tuning:
			HYPER_PARAMETERS = {
				"onset": args.onset, "offset": args.offset,
				"min_duration_on": args.min_duration_on,
				"min_duration_off": args.min_duration_off
			}
		else:
			HYPER_PARAMETERS = None
		
		readscp = np.genfromtxt(wavscp,dtype=str)
		
		if len(readscp.shape) == 1: 
			m = readscp.shape[0]
			column = min(2,m-1)
			wavfiles= readscp.reshape(-1,m)[:,column]
		else:
			column = min(2,readscp.shape[1]-1)
			wavfiles= readscp[:,column]
		
		for input_wav in wavfiles:
			nfile=input_wav.split("/")[-1].split(".")[0]
			if not os.path.exists(f"{args.outputpath}/{nfile}.pkl"):
				pyaanote_vad(args,input_wav,args.dataset,HYPER_PARAMETERS)
	
	
main()


