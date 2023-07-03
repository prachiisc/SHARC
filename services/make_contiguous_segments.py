#!/usr/bin/env python3

# Copyright  2016  David Snyder
#            2017  Matthew Maciejewski
# Apache 2.0.

"""This script converts a segments and labels file to a NIST RTTM
file. It creates flat segmentation (i.e. no overlapping regions)
from overlapping segments, e.g. the output of a sliding-window
diarization system. The speaker boundary between two overlapping
segments by different speakers is placed at the midpoint between
the end of the first segment and the start of the second segment.

The segments file format is:
<segment-id> <recording-id> <start-time> <end-time>
The labels file format is:
<segment-id> <speaker-id>

The output RTTM format is:
<type> <file> <chnl> <tbeg> \
        <tdur> <ortho> <stype> <name> <conf> <slat>
where:
<type> = "SPEAKER"
<file> = <recording-id>
<chnl> = "0"
<tbeg> = start time of segment
<tdur> = duration of segment
<ortho> = "<NA>"
<stype> = "<NA>"
<name> = <speaker-id>
<conf> = "<NA>"
<slat> = "<NA>"
"""

import argparse
import sys
from pdb import set_trace as bp
import os

#sys.path.append('services/steps/libs')
sys.path.append(os.path.join(os.getcwd(),'/data1/prachis/Dihard_2020/SSC/services/steps/libs'))

import common_kaldi as common_lib


def get_args():
  parser = argparse.ArgumentParser(
    description="""This script converts a segments and labels file
    to a NIST RTTM file. It handles overlapping segments (e.g. the
    output of a sliding-window diarization system).""")

  parser.add_argument("--withshift", action='store_true',
                      help="average with shift resolution")
  parser.add_argument("segments", type=str,
                      help="Input segments file")
  parser.add_argument("avg_segments", type=str,
                      help="Output RTTM file")

  args = parser.parse_args()
  return args
args = get_args()

def main():
  # Segments file
  reco2segs = {}
  with common_lib.smart_open(args.segments) as segments_file:
    for line in segments_file:
      seg, reco, start, end = line.strip().split()

      if reco in reco2segs:
        reco2segs[reco] = reco2segs[reco] + "#" + start + "," + end + "," + seg
      else:
        reco2segs[reco] = reco + "#" + start + "," + end + "," + seg

  # Cut up overlapping segments so they are contiguous
  contiguous_segs = []
  for reco in sorted(reco2segs):
    segs = reco2segs[reco].strip().split('#')
    new_segs = ""
    for i in range(1, len(segs)-1):
      # bp()
      start, end, label = segs[i].split(',')
      next_start, next_end, next_label = segs[i+1].split(',')
      if float(end) > float(next_start):
        done = False
        avg = str((float(next_start) + float(end)) / 2.0)
        segs[i+1] = ','.join([avg, next_end, next_label])
        new_segs += "#" + start + "," + avg + "," + label
      else:
        new_segs += "#" + start + "," + end + "," + label
    start, end, label = segs[-1].split(',')
    new_segs += "#" + start + "," + end + "," + label
    contiguous_segs.append(reco + new_segs)
  # bp()
  # reco_segs={}
  with common_lib.smart_open(args.avg_segments, 'w') as rttm_writer:
    for reco_line in contiguous_segs:
      segs = reco_line.strip().split('#')
      reco = segs[0]
      towrite = {}

      for i in range(1, len(segs)):
        start, end, label = segs[i].strip().split(',')
        labels = label.split()
        for lab in labels:
           print("{0} {1} {2:7.3f} {3:7.3f}".format(
            lab, reco, float(start), float(end)), file=rttm_writer)
      print(reco," done")
           # print("{0} {1} {2:7.3f} {3:7.3f}".format(
            # lab, reco, float(start), float(end)))
      #     if lab in towrite:
      #       towrite[lab] += "{},{},{}\n".format(start, end, lab)
      #     else:
      #       towrite[lab] = "{},{},{}\n".format(start, end, lab)
      # reco_segs[reco] = towrite

  # Edited merged segs
 
  # merged_segs = {}
  # for reco in reco_segs:
    
  #   towrite = reco_segs[reco]
  #   new_segs = {}
  #   for lab in towrite:
  #     segs = towrite[lab].split('\n')[:-1]
  #     new_segs[lab] = ""
  #     for i in range(len(segs)-1):
  #       start, end, label = segs[i].split(',')
  #       next_start, next_end, next_label = segs[i+1].split(',')
  #       # labels = label.split()
  #       # next_labels = next_label.split()
  #       # num_labels = len(labels)
  #       # num_next_labels = len(next_labels)
       
  #       if float(end) == float(next_start) and label == next_label:
  #         segs[i+1] = ','.join([start, next_end, next_label])

  #       else:
  #         new_segs[lab] += " " + start + "," + end + "," + label
       
  #     start, end, label = segs[-1].split(',')
  #     new_segs[lab] += " " + start + "," + end + "," + label
  #   merged_segs[reco] = new_segs


  # with common_lib.smart_open(args.rttm_file, 'w') as rttm_writer:
  #   for reco in reco_segs:
  #     towrite = reco_segs[reco]
  #     segs = reco_line.strip().split()
  #     reco = segs[0]
  #     for i in range(1, len(segs)):
  #       start, end, label = segs[i].strip().split(',')
  #       labels = label.split()
  #       for lab in towrite:
  #         print("{0} {1} {2:7.3f} {3:7.3f}".format(
  #           lab, reco, float(start), float(end)), file=rttm_writer)

def main_new():
  
  # Segments file
  reco2segs = {}
  with common_lib.smart_open(args.segments) as segments_file:
    for line in segments_file:
      seg, reco, start, end = line.strip().split()

      if reco in reco2segs:
        reco2segs[reco] = reco2segs[reco] + "#" + start + "," + end + "," + seg
      else:
        reco2segs[reco] = reco + "#" + start + "," + end + "," + seg

  # Cut up overlapping segments so they are contiguous
  contiguous_segs = []
  for reco in sorted(reco2segs):
    segs = reco2segs[reco].strip().split('#')
    new_segs = ""
    for i in range(1, len(segs)-1):
      # bp()
      start, end, label = segs[i].split(',')
      next_start, next_end, next_label = segs[i+1].split(',')
      if float(end) > float(next_start):
        done = False
        avg = str(float(next_start))
        segs[i+1] = ','.join([avg, next_end, next_label])
        new_segs += "#" + start + "," + avg + "," + label
      else:
        new_segs += "#" + start + "," + end + "," + label
    start, end, label = segs[-1].split(',')
    new_segs += "#" + start + "," + end + "," + label
    contiguous_segs.append(reco + new_segs)
  # bp()
  # reco_segs={}
  with common_lib.smart_open(args.avg_segments, 'w') as rttm_writer:
    for reco_line in contiguous_segs:
      segs = reco_line.strip().split('#')
      reco = segs[0]
      towrite = {}

      for i in range(1, len(segs)):
        start, end, label = segs[i].strip().split(',')
        labels = label.split()
        for lab in labels:
           print("{0} {1} {2:7.3f} {3:7.3f}".format(
            lab, reco, float(start), float(end)), file=rttm_writer)
      print(reco," done")

if __name__ == '__main__':
  if args.withshift:
    main_new()
  else:
    main()
  
