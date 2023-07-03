#!/usr/bin/env python

import sys
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import warnings
from pdb import set_trace as bp
warnings.filterwarnings("ignore")

#import torchaudio

def gen_VAD_segments(data_dir, op_dir, tol):
  
  rttm = os.path.join(data_dir,'rttm')
  reco2dur = os.path.join(data_dir,'reco2dur')
  segments = os.path.join(op_dir,'segments')

  if not os.path.exists(rttm):
        raise RuntimeError(f"cannot locate the rttm file at {rttm}")
  if not os.path.exists(reco2dur):
        raise RuntimeError(f"cannot locate the reco2dur file at {reco2dur}")

  
  rec2dur_np = np.genfromtxt(reco2dur, dtype=[object,float], delimiter=" ")
  #print(rec2dur_np)
  recordings = {}
  for row in rec2dur_np:
    print(row[0].decode("utf-8"), row[1])
    dim = int(row[1]*100)
    recordings[row[0].decode("utf-8")] = np.zeros(dim, dtype=np.bool)
    # print(recordings[row[0].decode("utf-8")].shape)
  
  fptr = open(rttm,'r')
  lines = fptr.readlines()
  fptr.close()

  for row in lines:
    cols = row.split()
    #print(len(cols))
    start = int(float(cols[3])*100) 
    dur = int(float(cols[4])*100)
    #print(cols[1])
    recordings[cols[1]][start:start+dur] = True
  
  
  #print(recordings["data_simu_wav_lib_vox_tr_ns1_beta2_100_56_mix_0000056"])  
  fptr = open(segments,'w')

  file_id = 0
  for key in recordings.keys():
    file_id += 1
    utt_id = 1
    idx = 0
    print (key)
    while idx < recordings[key].shape[0] and recordings[key][idx] == False:
      idx += 1
    while idx < recordings[key].shape[0]:
      start_idx = idx
      while idx+tol < recordings[key].shape[0] and np.any(recordings[key][idx:idx+tol+1]):
        #print(recordings[key][idx:idx+tol+1],np.any(recordings[key][idx:idx+tol+1]))
        idx += 1
      if idx+tol >= recordings[key].shape[0]:
        end_idx = recordings[key].shape[0] 
      else:
        end_idx = idx
      fptr.write("%04d_%s_%05d %s %.2f %.2f\n"%(file_id,key,utt_id,key,(start_idx/100),(end_idx/100)))
      if end_idx >= recordings[key].shape[0]:
            break
      utt_id += 1   
      idx += 1   
      while idx < recordings[key].shape[0] and recordings[key][idx] == False:
        idx += 1
      
def gen_VAD_segments_ami(data_dir, op_dir, tol):
      
  rttm = os.path.join(data_dir,'rttm')
  reco2dur = os.path.join(data_dir,'reco2dur')
  segments = os.path.join(op_dir,'segments')

  if not os.path.exists(rttm):
        raise RuntimeError(f"cannot locate the rttm file at {rttm}")
  if not os.path.exists(reco2dur):
        raise RuntimeError(f"cannot locate the reco2dur file at {reco2dur}")

  
  rec2dur_np = np.genfromtxt(reco2dur, dtype=[object,float], delimiter=" ")
  #print(rec2dur_np)
  recordings = {}
  for row in rec2dur_np:
    print(row[0].decode("utf-8"), row[1])
    dim = int(row[1]*100)
    recordings[row[0].decode("utf-8")] = np.zeros(dim, dtype=np.bool)
    #print(recordings[row[0].decode("utf-8")].shape)
  
  fptr = open(rttm,'r')
  lines = fptr.readlines()
  fptr.close()

  for row in lines:
    cols = row.split()
    #print(len(cols))
    start = int(float(cols[3])*100) 
    dur = int(float(cols[4])*100)
    #print(cols[1])
    recordings[cols[1]][start:start+dur] = True
  
  
  #print(recordings["data_simu_wav_lib_vox_tr_ns1_beta2_100_56_mix_0000056"])  
  fptr = open(segments,'w')

  file_id = 0
  for key in recordings.keys():
    file_id += 1
    utt_id = 1
    idx = 0
    while idx < recordings[key].shape[0] and recordings[key][idx] == False:
      idx += 1
    while idx < recordings[key].shape[0]:
      start_idx = idx
      while idx+tol < recordings[key].shape[0] and np.any(recordings[key][idx:idx+tol+1]):
        #print(recordings[key][idx:idx+tol+1],np.any(recordings[key][idx:idx+tol+1]))
        idx += 1
      if idx+tol >= recordings[key].shape[0]:
        end_idx = recordings[key].shape[0] 
      else:
        end_idx = idx
      fptr.write("%s_%05d %s %.2f %.2f\n"%(key,utt_id,key,(start_idx/100),(end_idx/100)))
      if end_idx >= recordings[key].shape[0]:
            break
      utt_id += 1   
      idx += 1   
      while idx < recordings[key].shape[0] and recordings[key][idx] == False:
        idx += 1
      

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        description='Create VAD segments from rttm', add_help=True)
  parser.add_argument(
 	'data_dir', metavar='data-dir', type=Path,
        help='dir containing rttm file and reco2dur file.')
  parser.add_argument(
 	'op_dir', metavar='op-dir', type=Path,
        help='dir to place segments file created.')
  parser.add_argument(
 	'tolerance', metavar='tolerance', type=int,
        help='allowed gap between speech in milliseconds')
  if(len(sys.argv) != 4):
    parser.print_help() 
    sys.exit(1) 
  args = parser.parse_args() 
  
  # gen_VAD_segments(args.data_dir, args.op_dir, args.tolerance) 
  gen_VAD_segments_ami(args.data_dir, args.op_dir, args.tolerance) 
