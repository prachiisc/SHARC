import numpy as np
import os
from pdb import set_trace as bp
import sys

#totalfiles=5000
totalfiles=int(sys.argv[2]) #250
totalsplits= int(sys.argv[3]) #40

os.system('./path.sh')
#filelist=open('lists/test_mixture.list').readlines()
# foldpath='lists/dihard_dev_2020_track1_fbank_jhu'
# foldpath=f'lists/{sys.argv[1]}'
foldpath= sys.argv[1]

listname='full.list'
# listname = sys.argv[2]
# totalfiles = len(open(f'{foldpath}/vox_diar_test.list').readlines())

#filelist = np.arange(totalfiles)
#np.savetxt(f'{foldpath}/full.list',filelist)
filelist = open('{}/{}'.format(foldpath,listname)).readlines()

#totalsplits=len(filelist)
splitlist = np.array_split(filelist,totalsplits)
for i,split in enumerate(splitlist):
    writefold='{}/split{}/{}'.format(foldpath,totalsplits,i+1)
    cmd='mkdir -p {}'.format(writefold)
    os.system(cmd)
    # bp()
    filename=split[0].rsplit()[0]
    print(filename)
    
    #continue
    a1=open('{}/{}'.format(writefold,listname),'w')
    a1.writelines(split)
    #np.savetxt('{}/dataset.list'.format(writefold),split,fmt='%s',delimiter='')
