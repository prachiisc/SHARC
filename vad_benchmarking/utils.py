
import argparse 
import subprocess
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from pdb import set_trace as bp

def silero_conversion(SR,times):
	sample_rate = 16000
	#times = [{'start': 29845536, 'end': 29902816}, {'start': 29905440, 'end': 29956576}, {'start': 29959200, 'end': 29982688}, {'start': 29990432, 'end': 29999072}, {'start': 30008352, 'end': 30050272}, {'start': 30083616, 'end': 30097888}, {'start': 30100512, 'end': 30124512}, {'start': 30126624, 'end': 30150112}, {'start': 30156832, 'end': 30172640}, {'start': 30176800, 'end': 30191072}, {'start': 30238240, 'end': 30288352}]
	for t in times:
		t['start'] = t['start'] / sample_rate
		t['end'] = t['end'] / sample_rate
	return times
def load_silero_vad_pkl(f_name):
	segments_dict=np.load(f_name,allow_pickle=True)
	segments_dict=silero_conversion(16000,segments_dict)
	#segment=''
	return [[i['start'],i['end']] for i in segments_dict]
	
	return segments_dict 
def load_vbxVAD(f_name):
	with open(f_name,"r") as f:
		l=f.readlines()
	#print(l)
	l=[ i.split("\tspeech")[0]+"\n" for i in l]
	l=[ i.split("\t")[0]+" "+ i.split("\t")[1] for i in l]
	s="".join(l)
	return s
	
def load_pyannote(f_name):
	segments_dict=np.load(f_name,allow_pickle=True)
	d=[i for i in segments_dict.get_timeline()]
	d=[[j for j in i] for i in d]
	return d

def rttm_to_segment():
	paths=["/home/coder/exp_nitk_iisc/DISPLACE_Baselines/Displace_data/set1/DISPLACE_2023_Dev-Part1_Label_Release/RTTM/Track-1_SD","/home/coder/exp_nitk_iisc/DISPLACE_Baselines/Displace_data/set2/DISPLACE_2023_Dev-Part2_Label_Release/RTTM/Track1_SD","/home/coder/exp_nitk_iisc/DISPLACE_Baselines/Displace_data/set3/DISPLACE_2023_Dev-Part3_Label_Release/RTTM/Track1_SD"]
	for j in paths:
		for i in os.listdir(j):
			with open(j+"/"+i,"r") as f:
				content=f.readlines()
			f.close()	
			content_loop=[[i.split(" ")[3],float(i.split(" ")[3])+float(i.split(" ")[4])] for i in content] 
			string_content="".join(str(i[0])+" "+str(i[1])+"\n" for i in content_loop)
			with open(j+"/"+i.split(".")[0]+'.segment','w') as k:
				k.write(string_content)
	
def results_segments(args):
	vad_dir_path=args.vad_dir_path 
	file_dir=os.listdir(vad_dir_path)
	#vad_file_dir=[[vad_dir_path+"/"+i+"/"+j for j in os.listdir(vad_dir_path+"/"+i)] for i in file_dir]
	#files_path=[[[j+"/"+h for h in os.listdir(j)] for j in i] for i in vad_file_dir]
	#file_dir_path=[vad_dir_path+"/"+i for i in file_dir]
	try:
		os.mkdir(vad_dir_path+"/segments")
	except:
		print('folder exists')

	if args.vad_type == 'pyannote':
		for i in range(len(file_dir)):
			if file_dir[i].split(".")[-1]=='pkl':	
				segment=load_pyannote(vad_dir_path+"/"+file_dir[i])
				string_content="".join(str(k[0])+" "+str(k[1])+"\n" for k in segment)

				f_name=vad_dir_path+"/segments/"+file_dir[i].split('.')[0]+"_pyannote.segments"
				with open(f_name,'w') as d:
					d.write(string_content)
	elif args.vad_type == 'silero':
		for i in range(len(file_dir)):
			if file_dir[i].split(".")[-1]=='pkl':	
				segment=load_silero_vad_pkl(vad_dir_path+"/"+file_dir[i])
				string_content="".join(str(k[0])+" "+str(k[1])+"\n" for k in segment)

				f_name=vad_dir_path+"/segments/"+file_dir[i].split('.')[0]+"_pyannote.segments"
				with open(f_name,'w') as d:
					d.write(string_content)
	
	
		
parser = argparse.ArgumentParser()
parser.add_argument('--vad_dir_path',type=str, default="pyannote_vad/ami_sdm_eval", help="pyannote sad pickle file")
parser.add_argument('--vad_type',type=str,default='pyannote')
args = parser.parse_args()	
	
results_segments(args)
