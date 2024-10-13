import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_filename',
                        required=True,
                        help='input segment file')
    parser.add_argument('--output_filename',
                        required=True,
                        help='output segment file')
    args = parser.parse_args()

    return args
args = get_args()
# input segments file
inp_segment = open(f"{args.input_filename}","r").readlines()
out_segment = open(f"{args.output_filename}","a")
for line in inp_segment:
    _,filename,start,end = line.split()
    start = round(float(start),2)
    end  = round(float(end),2)
    startid = f"{int(start*100):06}"
    endid = f"{int(end*100):06}"
    towrite = f"{filename}-{startid}-{endid} {filename} {start} {end}\n"
    out_segment.writelines(towrite)
