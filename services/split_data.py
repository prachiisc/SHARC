import numpy as np
import os
from pdb import set_trace as bp
totalfiles=414
totalsplits=80

os.system('./path.sh')
#filelist=open('lists/test_mixture.list').readlines()
foldpath='xvectors/xvectors_ami_dev_resnet_0.75s/'
filelist=open('{}/temp.list'.format(foldpath)).readlines()
xveclist=open(foldpath+'/xvector.scp').readlines()
totalsplits=len(filelist)
splitlist = np.array_split(filelist,totalsplits)
for i,split in enumerate(splitlist):
    writefold='{}/split{}/{}'.format(foldpath,totalsplits,i+1)
    cmd='mkdir -p {}'.format(writefold)
    os.system(cmd)
    #bp()
    print(split[0].rsplit()[0])
    filename=split[0].rsplit()[0]
    #continue
    cmd2='grep {} {}/xvector.scp > {}/xvector.scp'.format(filename,foldpath,writefold)
    os.system(cmd2)
    #a1=open('{}/xvector.scp'.format(writefold),'w')
    #a1.writelines(split)
    cmd='copy-vector scp:{}/xvector.scp ark,scp:{}/xvector.{}.ark,{}/xvector.{}.scp'.format(writefold,writefold,i+1,writefold,i+1)
    os.system(cmd)
    #np.savetxt('{}/dataset.list'.format(writefold),split,fmt='%s',delimiter='')
