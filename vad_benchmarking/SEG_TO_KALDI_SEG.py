import sys
import os
import random
from pdb import set_trace as bp

with open(sys.argv[1], 'r') as f:
    segments = f.readlines()

input_file_name = sys.argv[1].split("/")[-1].split('_')[0]
if "ami" in sys.argv[2]:
    input_file_name = f'AMI_{input_file_name}_SDM'
else:
    input_file_name = f'{input_file_name}'



with open(sys.argv[2], 'w') as f:
    prev_end_time = 0
    for i in range(len(segments)):
        start_time, end_time = segments[i].strip().split()
        if float(start_time) >= prev_end_time:
            start_time_pt=str("%.3f"%float(start_time))
            end_time_pt=str("%.3f"%float(end_time))
            utt_id_suffix = "0"*(7-len(start_time_pt.replace(".","")))+start_time_pt.replace(".","")+"-"+"0"*(7-len(end_time_pt.replace(".","")))+end_time_pt.replace(".","")
            utt_id = f"{input_file_name}-{utt_id_suffix}"
            f.write(f"{utt_id} {input_file_name} {start_time_pt} {end_time_pt}\n")
            prev_end_time = float(end_time)

#python silero_to_kaldi.py silero_segments.txt kaldi_segments.txt

