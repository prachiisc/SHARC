import numpy as np
from collections import defaultdict
# import services.pic_dihard_ami as pic
import services.agglomerative as ahc
from services.run_spectralclustering import do_spectral_clustering
from utils_cluster import mkdir_p
import os
import subprocess
from pdb import set_trace as bp
from  sklearn.metrics.cluster import contingency_matrix
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

def unique(arr, return_ind=False):
    if return_ind:
        k = 0
        d = dict()
        uniques = np.empty(arr.size, dtype=arr.dtype)
        indexes = np.empty(arr.size, dtype='i')
        for i, a in enumerate(arr):
            if a in d:
                indexes[i] = d[a]
            else:
                indexes[i] = k
                uniques[k] = a
                d[a] = k
                k += 1
        return uniques[:k], indexes
    else:
        _, idx = np.unique(arr, return_index=True)
        return arr[np.sort(idx)]

def translateLabels(masterList, listToConvert):    
  contMatrix = contingency_matrix(masterList, listToConvert)
  labelMatcher = munkres.Munkres()
  labelTranlater = labelMatcher.compute(contMatrix.max() - contMatrix)

  uniqueLabels1 = list(set(masterList))
  uniqueLabels2 = list(set(listToConvert))

  tranlatorDict = {}
  for thisPair in labelTranlater:
    tranlatorDict[uniqueLabels2[thisPair[1]]] = uniqueLabels1[thisPair[0]]

  return [tranlatorDict[label] for label in listToConvert]

def write_results_dict(args,fname, output_file, results_dict, reco2utt):
        """Writes the results in label file"""
        f = fname
        output_label = open(output_file+'/'+f+'.labels','w')

        hypothesis = results_dict[f]
        meeting_name = f
        reco = reco2utt.split()[0]
        utts = reco2utt.rstrip().split()[1:]
        if reco == meeting_name:
            for j,utt in enumerate(utts):
                towrite = utt +' '+str(hypothesis[j])+'\n'
                output_label.writelines(towrite)
        output_label.close()

        rttm_channel=1
        segmentsfile = args.segments+'/'+f+'.segments'
        python = args.which_python
        # python = '/home/prachis/miniconda3/envs/mytorch/bin/python'
        kaldi_recipe_path="./"
        cmd = '{} {}/diarization/make_rttm.py --rttm-channel  {} {} {}/{}.labels {}/{}.rttm' .format(python,kaldi_recipe_path,rttm_channel, segmentsfile,output_file,f,output_file,f)        
        os.system(cmd)

def write_results_dict_overlap(args,fname, output_file, results_dict, reco2utt):
        """Writes the results in label file"""
        f = fname
        output_label = open(output_file+'/'+f+'.labels','w')

        hypothesis = results_dict[f]
        meeting_name = f
        reco = reco2utt.split()[0]
        utts = reco2utt.rstrip().split()[1:]
        if reco == meeting_name:
            for j,utt in enumerate(utts):
                if np.isscalar(hypothesis[j]): 
                    towrite = utt +' '+str(hypothesis[j])+'\n'
                else:
                    if hypothesis[j,1]==-1:
                        towrite = utt +'\t'+str(hypothesis[j,0])+'\n'
                    else: 
                        towrite = utt +'\t'+str(hypothesis[j,0])+' '+str(hypothesis[j,1])+'\n'
                output_label.writelines(towrite)
        output_label.close()

        rttm_channel=1
        segmentsfile = args.segments+'/'+f+'.segments'
        python = args.which_python
        # python = '/home/prachis/miniconda3/envs/mytorch/bin/python'
        kaldi_recipe_path="./"
        cmd = '{} {}/diarization/make_rttm_for_overlap.py --rttm-channel  {} {} {}/{}.labels {}/{}.rttm' .format(python,kaldi_recipe_path,rttm_channel, segmentsfile,output_file,f,output_file,f)        
        os.system(cmd)


def compute_score(args,rttm_gndfile,rttm_newfile,outpath,overlap):
      fold_local='services/'
      scorecode='score.py -r '
      fold_local = 'services/'
      # print('--------------------------------------------------')
      if not overlap:
          cmd = '{} {}/dscore-master/{}{} --ignore_overlaps --collar 0.25 -s {} > {}.txt 2> err.txt'.format(args.which_python,fold_local,scorecode,rttm_gndfile,rttm_newfile,outpath)
          # cmd=args.which_python +' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' --ignore_overlaps --collar 0.25 -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
          bashCommand="cat {}.txt | grep OVERALL |awk '{{print $4}}'".format(outpath)
      
      else:
          cmd = '{} {}/dscore-master/{}{} -s {} > {}_overlap.txt 2> err.txt'.format(args.which_python,fold_local,scorecode,rttm_gndfile,rttm_newfile,outpath)
          # cmd=args.which_python + ' '+ fold_local + 'dscore-master/' + scorecode + rttm_gndfile + ' -s ' + rttm_newfile + ' > ' + outpath + '.txt'
          os.system(cmd)
          bashCommand="cat {}_overlap.txt | grep OVERALL |awk '{{print $4}}'".format(outpath)
      
      # print('----------------------------------------------------')
      # subprocess.check_call(cmd,stderr=subprocess.STDOUT)
      # print('scoring ',rttm_gndfile)
      
      output=subprocess.check_output(bashCommand,shell=True)
      # output=subprocess.check_output(bashCommand,stderr=subprocess.DEVNULL)
      return float(output.decode('utf-8').rstrip())

def validate_spectral_clustering(args,fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_',rttm_gndfile=None,overlap=1):
    f = fname
    print('Spectral Clustering')
    # overlap =1

    results_dict = defaultdict(np.array)

    # threshold = None
    labelfull=np.arange(output_new.shape[0])
    clusterlen=[1]*len(labelfull)    

    nframe = output_new.shape[0]
    # distance_matrix = (output_new+1)/2
    # output_new = output_new/np.max(abs(output_new))
    # distance_matrix = 1/(1+np.exp(-output_new))
    distance_matrix = output_new
    
    if flag:
        if n_clusters is not None:
            minK = n_clusters
            maxK = n_clusters
            th = 1e-2
        else:
            minK = 1
            maxK = 10
            th = 0.5
    # custom_dist='cosine',scoretype='laplacian',
    labelfull = do_spectral_clustering(distance_matrix,
                                        gauss_blur=0.1,
                                        p_percentile=0.95,
                                        minclusters=minK,
                                        maxclusters=maxK,
                                        truek=4,custom_dist='cosine',
                                        scoretype='laplacian',
                                        stop_eigenvalue=th)
   
    uniq_labels,labelfull=unique(labelfull,True)

    labelfull = labelfull.reshape(-1,1)
    adj_bool = labelfull == labelfull.T
    adj_label = np.ones((nframe,nframe))*adj_bool
    plt.imshow(adj_label)
    plt.title('{}_spec_trained'.format(f))
    plt.savefig('testpic_images/{}/spec_trained.png'.format(f))

    # uniq_labels = np.unique(labelfull)
    n_clusters=len(uniq_labels)
    clusterlen = []
    for lab in uniq_labels:
        clusterlen.append(len(np.where(labelfull==lab)[0]))
    # bp()
    print('clusterlen:',clusterlen,' n_clusters:',n_clusters)
    results_dict[f]=labelfull
    if final:
        out_file=args.outf+'/'+'final_spectral{}rttms/'.format(pref)
        # out_file=args.outf+'/'+'final_spectral_test_rttms/'
    else:
        out_file=args.outf+'/'+'rttms/'
    # if not os.path.isdir(out_file):
    #     os.makedirs(out_file)
    mkdir_p(out_file)
    outpath=out_file +'/'+f
    rttm_newfile=out_file+'/'+f+'.rttm'
    
    # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
    write_results_dict(args,fname, out_file, results_dict, reco2utt)
    # self.write_results_dict(out_file)
    # bp()
    der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,0)
    if overlap:
        overlap_der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,1)
        print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
    print("\n%s DER: %.2f" % (fname, der))
    return overlap_der


def validate_path_integral(args,fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_',rttm_gndfile=None,overlap=1):
            print('Path integral clustering')
            f = fname
            # overlap =1
            # bp()
            results_dict = defaultdict(np.array)

            # threshold = None
            labelfull=np.arange(output_new.shape[0])
            clusterlen=[1]*len(labelfull)    
            N = len(labelfull)
            clusterlen_org = clusterlen.copy()
            nframe = output_new.shape[0]
            # plda scores
            z=0.5
            K = 30
            # ground scores
            # z=0.01
            # K = nframe
            # distance_matrix = (output_new+1)/2
            # output_new = output_new/np.max(abs(output_new))
            # distance_matrix = 1/(1+np.exp(-output_new))
            if flag==1:
                # z = 0.01
                # K = nframe
                neb = 5
                beta1 = 0.95
                toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
                toep[toep>neb] = neb
                weighting = beta1**(toep)
                distance_matrix = weighting*output_new
            else:
                distance_matrix = output_new
            # bp()
            # ev_s, eig_s , _ = np.linalg.svd(distance_matrix,full_matrices=True)
            if args.threshold != None:
                 n_clusters = 1
            final_k = min(K, nframe - 1) 
            print('final_k: ',final_k)
            mypic =pic.PIC_dihard_threshold(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),args.threshold,K=final_k,z=z) 
            # bp()
            # if N < 200 or flag ==2:
            if flag == 2:
                if args.threshold == None:
                    labelfull,clusterlen = mypic.gacCluster_oracle_org()
                else:
                    if n_clusters > 1:
                        labelfull,clusterlen = mypic.gacCluster_org()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
            else:
                if args.threshold == None:
                    labelfull,clusterlen= mypic.gacCluster_oracle()
                else:
                    if n_clusters > 1:
                        labelfull,clusterlen = mypic.gacCluster()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
            
            labelfull = labelfull.reshape(-1,1)
            adj_bool = labelfull == labelfull.T
            adj_label = np.ones((nframe,nframe))*adj_bool
            # plt.imshow(adj_label)
            # plt.title('{}_pic_temp_trained'.format(f))
            # plt.savefig('testpic_images/{}/pic_temp_trained.png'.format(f))

            n_clusters=len(clusterlen)
            print('clusterlen:',clusterlen, 'n_clusters:',n_clusters)
            results_dict[f]=labelfull
            if final:
                out_file=args.outf+'/'+'final_pic{}rttms/'.format(pref)
                rttm_valfile = out_file+'/valrttm'
                # out_file=args.outf+'/'+'final_spectral_test_rttms/'
            else:
                out_file=args.outf+'/'+'picrttms{}/'.format(pref)
                rttm_valfile = out_file+'/valbaserttm'

            mkdir_p(out_file)
            
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
            
            write_results_dict(args,fname, out_file, results_dict, reco2utt)
            # self.write_results_dict(out_file)
                
            bashCommand="cat {} >> {}".format(rttm_newfile,rttm_valfile)
            os.system(bashCommand)
            # write_results_dict(fname, out_file, results_dict, reco2utt)

            der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                overlap_der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,1)
                print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
            print("\n%s DER: %.2f" % (fname, der))
            # return overlap_der
            return rttm_valfile, outpath


def validate_path_integral_ami(args,fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_',rttm_gndfile=None,overlap=1,K=30,nb=5):
            print('Path integral clustering')
            f = fname
            
            # overlap =1
            # bp()
            results_dict = defaultdict(np.array)

            # threshold = None
            labelfull=np.arange(output_new.shape[0])
            clusterlen=[1]*len(labelfull)    
            N = len(labelfull)
            clusterlen_org = clusterlen.copy()
            nframe = output_new.shape[0]
            # plda scores
            z=args.z
            K = args.K
            
            # ground scores
            # z=0.01
            # K = nframe
            # distance_matrix = (output_new+1)/2
            # output_new = output_new/np.max(abs(output_new))
            # distance_matrix = 1/(1+np.exp(-output_new))
            if flag==1:
                # z = 0.01
                # K = nframe
                neb = args.nb
                beta1 = args.beta
                toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
                toep[toep>neb] = neb
                weighting = beta1**(toep)
                distance_matrix = weighting*output_new
            else:
                distance_matrix = output_new
            # bp()
            # ev_s, eig_s , _ = np.linalg.svd(distance_matrix,full_matrices=True)
            if args.threshold != None:
                 n_clusters = 1
            final_k = min(K, nframe - 1) 
            print('final_k: ',final_k)
            mypic =pic.PIC_ami_threshold(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),args.threshold,K=final_k,z=z) 
            # bp()
            # if N < 200 or flag ==2:
            if flag == 2:
                if args.threshold == None:
                    if n_clusters > 1:
                        labelfull,clusterlen = mypic.gacCluster_oracle_org()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
                else:
                    labelfull,clusterlen = mypic.gacCluster_org()
            else:
                if args.threshold == None:
                    if n_clusters > 1:
                        labelfull,clusterlen= mypic.gacCluster_oracle()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
                else:
                    labelfull,clusterlen = mypic.gacCluster()
                   
            
            # labelfull = labelfull.reshape(-1,1)
            # adj_bool = labelfull == labelfull.T
            # adj_label = np.ones((nframe,nframe))*adj_bool
            # plt.imshow(adj_label)
            # plt.title('{}_pic_temp_trained'.format(f))
            # plt.savefig('testpic_images/{}/pic_temp_trained.png'.format(f))

            n_clusters=len(clusterlen)
            print('clusterlen:',clusterlen, 'n_clusters:',n_clusters)
            results_dict[f]=labelfull
            if final:
                out_file=args.outf+'/'+'final_pic{}rttms/'.format(pref)
                rttm_valfile = out_file+'/valrttm'
                # out_file=args.outf+'/'+'final_spectral_test_rttms/'
            else:
                out_file=args.outf+'/'+'picrttms{}/'.format(pref)
                rttm_valfile = out_file+'/valbaserttm'

            mkdir_p(out_file)
            
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
            
            write_results_dict(args,fname, out_file, results_dict, reco2utt)
            # self.write_results_dict(out_file)
                
            # bashCommand="cat {} >> {}".format(rttm_newfile,rttm_valfile)
            # os.system(bashCommand)
            # write_results_dict(fname, out_file, results_dict, reco2utt)

            der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                overlap_der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,1)
                print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
            print("\n%s DER: %.2f" % (fname, der))
            # return overlap_der
            return rttm_valfile, outpath

def validate_path_integral_ami_gnd(args,fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_',rttm_gndfile=None,overlap=1,K=30,nb=5,clean_ind=None):
            print('Path integral clustering')
            f = fname
            ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+args.dataset_str+'/threshold_0.5_avg/labels_'+fname).readlines()
            full_gndlist=[g.split()[1:] for g in ground_labels]
            gnd_list = np.array([g[0] for g in full_gndlist])


            # uni_gnd_letter = np.unique(gnd_list)

            clean_list = np.array([len(f) for f in full_gndlist])
            overlap_ind =np.where(clean_list >1)[0]

            N_org = output_new.shape[0]
            output_new = output_new[np.ix_(clean_ind,clean_ind)]
            
            # overlap =1
            # bp()
            results_dict = defaultdict(np.array)

            # threshold = None
            labelfull=np.arange(output_new.shape[0])
            clusterlen=[1]*len(labelfull)    
            N = len(labelfull)
            clusterlen_org = clusterlen.copy()
            nframe = output_new.shape[0]
            # plda scores
            z=0.1
            K = K
            # ground scores
            # z=0.01
            # K = nframe
            # distance_matrix = (output_new+1)/2
            # output_new = output_new/np.max(abs(output_new))
            # distance_matrix = 1/(1+np.exp(-output_new))
            if flag==1:
                # z = 0.01
                # K = nframe
                neb = nb
                beta1 = 0.95
                toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
                toep[toep>neb] = neb
                weighting = beta1**(toep)
                distance_matrix = weighting*output_new
            else:
                distance_matrix = output_new
            # bp()
            # ev_s, eig_s , _ = np.linalg.svd(distance_matrix,full_matrices=True)
            if args.threshold != None:
                 n_clusters = 1
            final_k = min(K, nframe - 1) 
            print('final_k: ',final_k)
            mypic =pic.PIC_ami_threshold(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),args.threshold,K=final_k,z=z) 
            # bp()
            # if N < 200 or flag ==2:
            if flag == 2:
                if args.threshold == None:
                    labelfull,clusterlen = mypic.gacCluster_oracle_org()
                else:
                    if n_clusters > 1:
                        labelfull,clusterlen = mypic.gacCluster_org()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
            else:
                if args.threshold == None:
                    labelfull,clusterlen= mypic.gacCluster_oracle()
                else:
                    if n_clusters > 1:
                        labelfull,clusterlen = mypic.gacCluster()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
            
            labelfull_temp = labelfull.reshape(-1,)

            gnd_list_overlap = gnd_list[overlap_ind]
            labelfull = np.ones((N_org,))*(-1)
            labelfull[clean_ind] = labelfull_temp

            uni_clean_letter = np.unique(gnd_list[clean_ind])
            uni_labelfull_temp = np.unique(labelfull_temp)

            # bp()
            mapping  = np.argmax(contingency_matrix(labelfull_temp,gnd_list[clean_ind]), axis=1)
            mapdict = {}
            for i in range(len(mapping)):
                mapdict[uni_clean_letter[i]] = uni_labelfull_temp[mapping[i]]

            uni_overlap_letter = np.unique(gnd_list[overlap_ind])
            for letter in uni_overlap_letter:
                gnd_list_overlap[gnd_list_overlap==letter] = mapdict[letter]

            labelfull[overlap_ind] = gnd_list_overlap
            
            n_clusters=len(clusterlen)
            print('clusterlen:',clusterlen, 'n_clusters:',n_clusters)
            results_dict[f]=labelfull
            if final:
                out_file=args.outf+'/'+'final_pic{}rttms/'.format(pref)
                rttm_valfile = out_file+'/valrttm'
                # out_file=args.outf+'/'+'final_spectral_test_rttms/'
            else:
                out_file=args.outf+'/'+'picrttms{}/'.format(pref)
                rttm_valfile = out_file+'/valbaserttm'

            mkdir_p(out_file)
            
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
            
            write_results_dict(args,fname, out_file, results_dict, reco2utt)
            # self.write_results_dict(out_file)
                
            # bashCommand="cat {} >> {}".format(rttm_newfile,rttm_valfile)
            # os.system(bashCommand)
            # write_results_dict(fname, out_file, results_dict, reco2utt)

            der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                overlap_der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,1)
                print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
            print("\n%s DER: %.2f" % (fname, der))
            # return overlap_der
            return rttm_valfile, outpath

def validate_gnd(args,fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_',rttm_gndfile=None,overlap=1):
            print('Path integral clustering')
            f = fname
            ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+args.dataset_str+'/threshold_0.5_avg/labels_'+fname).readlines()
            full_gndlist=[g.split()[1:] for g in ground_labels]
            gnd_list = np.array([g[0] for g in full_gndlist])


            # uni_gnd_letter = np.unique(gnd_list)

            clean_list = np.array([len(f) for f in full_gndlist])
            overlap_ind =np.where(clean_list >1)[0]
            labelfull = gnd_list
            results_dict = defaultdict(np.array)

            results_dict[f]=labelfull
            if final:
                out_file=args.outf+'/'+'final_gnd{}rttms/'.format(pref)
                rttm_valfile = out_file+'/valrttm'
                # out_file=args.outf+'/'+'final_spectral_test_rttms/'
            else:
                out_file=args.outf+'/'+'gndrttms{}/'.format(pref)
                rttm_valfile = out_file+'/valbaserttm'

            mkdir_p(out_file)
            
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
            
            write_results_dict(args,fname, out_file, results_dict, reco2utt)
            # self.write_results_dict(out_file)
                
            # bashCommand="cat {} >> {}".format(rttm_newfile,rttm_valfile)
            # os.system(bashCommand)
            # write_results_dict(fname, out_file, results_dict, reco2utt)

            der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                overlap_der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,1)
                print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
            print("\n%s DER: %.2f" % (fname, der))
            # return overlap_der
            return rttm_valfile, outpath
            
            
def validate_path_integral_ami_gndadj(args,fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_',rttm_gndfile=None,overlap=1,K=30,nb=5,clean_ind=None):
            print('Path integral clustering')
            f = fname
            ground_labels=open('/data1/prachis/Dihard_2020/gae-pytorch/gae/tools_diar/ALL_GROUND_LABELS/'+args.dataset_str+'/threshold_0.5_avg/labels_'+fname).readlines()
            full_gndlist=[g.split()[1:] for g in ground_labels]
            gnd_list = np.array([g[0] for g in full_gndlist])


            # uni_gnd_letter = np.unique(gnd_list)

            clean_list = np.array([len(f) for f in full_gndlist])
            overlap_ind =np.where(clean_list >1)[0]

            N_org = output_new.shape[0]
            output_new[np.ix_(overlap_ind,clean_ind)] *=0.5
            
            output_new[np.ix_(clean_ind,overlap_ind)] *=0.5

            # overlap =1
            # bp()
            results_dict = defaultdict(np.array)

            # threshold = None
            labelfull=np.arange(output_new.shape[0])
            clusterlen=[1]*len(labelfull)    
            N = len(labelfull)
            clusterlen_org = clusterlen.copy()
            nframe = output_new.shape[0]
            # plda scores
            z=0.1
            K = K
            # ground scores
            # z=0.01
            # K = nframe
            # distance_matrix = (output_new+1)/2
            # output_new = output_new/np.max(abs(output_new))
            # distance_matrix = 1/(1+np.exp(-output_new))
            if flag==1:
                # z = 0.01
                # K = nframe
                neb = nb
                beta1 = 0.95
                toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
                toep[toep>neb] = neb
                weighting = beta1**(toep)
                distance_matrix = weighting*output_new
            else:
                distance_matrix = output_new
            # bp()
            # ev_s, eig_s , _ = np.linalg.svd(distance_matrix,full_matrices=True)
            if args.threshold != None:
                 n_clusters = 1
            final_k = min(K, nframe - 1) 
            print('final_k: ',final_k)
            mypic =pic.PIC_ami_threshold(n_clusters,clusterlen_org,labelfull,distance_matrix.copy(),args.threshold,K=final_k,z=z) 
            # bp()
            # if N < 200 or flag ==2:
            if flag == 2:
                if args.threshold == None:
                    labelfull,clusterlen = mypic.gacCluster_oracle_org()
                else:
                    if n_clusters > 1:
                        labelfull,clusterlen = mypic.gacCluster_org()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
            else:
                if args.threshold == None:
                    labelfull,clusterlen= mypic.gacCluster_oracle()
                else:
                    if n_clusters > 1:
                        labelfull,clusterlen = mypic.gacCluster()
                    else:
                        labelfull = np.zeros((nframe,1))
                        clusterlen = [nframe]
            
            labelfull = labelfull.reshape(-1,)

            
            n_clusters=len(clusterlen)
            print('clusterlen:',clusterlen, 'n_clusters:',n_clusters)
            results_dict[f]=labelfull
            if final:
                out_file=args.outf+'/'+'final_pic{}rttms/'.format(pref)
                rttm_valfile = out_file+'/valrttm'
                # out_file=args.outf+'/'+'final_spectral_test_rttms/'
            else:
                out_file=args.outf+'/'+'picrttms{}/'.format(pref)
                rttm_valfile = out_file+'/valbaserttm'

            mkdir_p(out_file)
            
            outpath=out_file +'/'+f
            rttm_newfile=out_file+'/'+f+'.rttm'
            # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
            
            write_results_dict(args,fname, out_file, results_dict, reco2utt)
            # self.write_results_dict(out_file)
                
            # bashCommand="cat {} >> {}".format(rttm_newfile,rttm_valfile)
            # os.system(bashCommand)
            # write_results_dict(fname, out_file, results_dict, reco2utt)

            der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                overlap_der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,1)
                print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
            print("\n%s DER: %.2f" % (fname, der))
            # return overlap_der
            return rttm_valfile, outpath


def validate_ahc(args,fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_',rttm_gndfile=None,overlap=1):
            print('AHC')
            f = fname
            # overlap =1
            threshold = args.threshold
            
            results_dict = defaultdict(np.array)
            # threshold = None
            nframe = output_new.shape[0]
            N = nframe
          
            if flag==1:
                neb = 5
                beta1 = 0.95
                toep = np.abs(np.arange(N).reshape(N,1)-np.arange(N).reshape(1,N))
                toep[toep>neb] = neb
                weighting = beta1**(toep)
                distance_matrix = weighting*output_new
            else:
                distance_matrix = output_new

            # if args.threshold != None:
            #     threshold = args.threshold
            #     n_clusters = None
                
            # bp()
            labelfull = ahc.AHC(distance_matrix, threshold= threshold,nspeaker=n_clusters)
            uniq_labels,labelfull=unique(labelfull,True)

            labelfull = labelfull.reshape(-1,1)
            adj_bool = labelfull == labelfull.T
            adj_label = np.ones((nframe,nframe))*adj_bool
            # plt.imshow(adj_label)
            # plt.title('{}_ahc_trained'.format(f))
            # plt.savefig('testpic_images/{}/ahc_trained.png'.format(f))

            n_clusters=len(uniq_labels)
            clusterlen = []
            for lab in uniq_labels:
                clusterlen.append(len(np.where(labelfull==lab)[0]))
            
            print('clusterlen:',clusterlen, 'n_clusters:',n_clusters)
            results_dict[f]=labelfull
            if final:
                out_file=args.outf+'/'+'final_ahc{}rttms/'.format(pref)
                rttm_valfile = out_file+'/val.rttm'
                # out_file=args.outf+'/'+'final_spectral_test_rttms/'
            else:
                out_file=args.outf+'/'+'ahcrttms{}/'.format(pref)
                rttm_valfile = out_file+'/valbase.rttm'

            mkdir_p(out_file)

            rttm_newfile=out_file+'/'+f+'.rttm'
            
            # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
            
            # outpath = out_file +'/val_labels'
            outpath=out_file +'/'+f
            
            write_results_dict(args,fname, out_file, results_dict, reco2utt)
            # self.write_results_dict(out_file)
                
                    
            # bashCommand="cat {} >> {}".format(rttm_newfile,rttm_valfile)
            # os.system(bashCommand)
            # subprocess.check_output(bashCommand,shell=False)
            
            # return rttm_valfile, outpath
            # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
            # write_results_dict(fname, out_file, results_dict, reco2utt)
            # # self.write_results_dict(out_file)
            # # bp()
            der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,0)
            if overlap:
                overlap_der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,1)
                print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
            print("\n%s DER: %.2f" % (fname, der))
            return overlap_der

def validate_spectral_clustering_weighted(args,fname,output_new,reco2utt,flag,final,n_clusters=None,pref='_',rttm_gndfile=None,overlap=1,scoretype=None,overlap_th=0.7,cent=None):
    f = fname
    # print('Spectral Clustering')
    # overlap =1
    
    forcing_label = 0
    results_dict = defaultdict(np.array)

    # threshold = None
    labelfull=np.arange(output_new.shape[0])
    clusterlen=[1]*len(labelfull)    

    nframe = output_new.shape[0]
    # distance_matrix = (output_new+1)/2
    # output_new = output_new/np.max(abs(output_new))
    # distance_matrix = 1/(1+np.exp(-output_new))
    distance_matrix = output_new
    if cent is not None:
        clean_ind = np.where(cent>=overlap_th)[0]
        pred_overlap_ind = np.where(cent<overlap_th)[0]
    else:
        clean_ind = []
        pred_overlap_ind = []
    # bp()

    #temporal 
    # neb = 5
    # beta1 = 0.95
    # toep = np.abs(np.arange(nframe).reshape(nframe,1)-np.arange(nframe).reshape(1,nframe))
    # toep[toep>neb] = neb
    # weighting = beta1**(toep)
    # distance_matrix = weighting*output_new
 
    # print(f'neb:{neb} beta: {beta1}')
            
    if flag:
        if n_clusters is not None:
            minK = n_clusters
            maxK = n_clusters
            th = 1e-2
        else:
            minK = 1
            maxK = 10
            th = args.threshold
    # custom_dist='cosine',scoretype='laplacian',
    # bp()
    labelfull,W = do_spectral_clustering(distance_matrix,
                                        gauss_blur=0.1,
                                        p_percentile=0.95,
                                        minclusters=minK,
                                        maxclusters=maxK,
                                        truek=4,
                                        scoretype=scoretype,
                                        stop_eigenvalue=th,
                                        clean_ind=clean_ind)
   
    # bp()
    # uniq_labels,labelfull=unique(labelfull,True)
    
    
    # overlap_th = 0.7
    # mx = overlap_th*np.max(W,axis=1,keepdims=True)
    # mx = 0.2
    
    if len(clean_ind)==0:
        mx = overlap_th
        mask_overlap = (W>=mx)
        pred_overlap_ind = np.where(np.sum(mask_overlap,axis=1)>1)[0]
        
    label_withoverlap = np.ones((nframe,2),dtype=int)*(-1)
    
    
    label_withoverlap[:,0]=labelfull
    uniq_labels =  np.unique(labelfull)
    n_clusters = len(uniq_labels)
    # using predicted overlap labels
    if len(pred_overlap_ind)>0 and n_clusters > 1:
        sort_ind = np.argsort(W[pred_overlap_ind],axis=1)[:,::-1]
        label_withoverlap[pred_overlap_ind,0]=sort_ind[:,0]
        label_withoverlap[pred_overlap_ind,1]=sort_ind[:,1]
    
    labelfull = label_withoverlap
            
    clusterlen = []
    for lab in uniq_labels:
        clusterlen.append(len(np.where(labelfull==lab)[0]))
    # bp()
    print('clusterlen:',clusterlen,' n_clusters:',n_clusters)

    results_dict[f]=labelfull
    if final:
        out_file=args.outf+'/'+'final_spectral{}rttms/'.format(pref)
        rttm_valfile = out_file+'/valrttm'
        # out_file=args.outf+'/'+'final_spectral_test_rttms/'
    else:
        out_file=args.outf+'/'+'rttms{}/'.format(pref)
        rttm_valfile = out_file+'/valbaserttm'

    mkdir_p(out_file)
    # if not os.path.isdir(out_file):
    #     os.makedirs(out_file)
    outpath=out_file +'/'+f

    rttm_newfile=out_file+'/'+f+'.rttm'
    
    
    # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
    # bp()
    # outpath = out_file +'/val_labels'
    write_results_dict_overlap(args,fname, out_file, results_dict, reco2utt)
    # self.write_results_dict(out_file)
    der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,0)
    if overlap:
        overlap_der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,overlap)
        print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
    print("\n%s DER: %.2f" % (fname, der))
            
    # bashCommand="cat {} >> {}".format(rttm_newfile,rttm_valfile)
    # os.system(bashCommand)
    # return rttm_valfile, outpath
    return overlap_der


def validate_clustering_weighted(args,fname,W,reco2utt,flag,final,pref='_',rttm_gndfile=None,overlap=1,overlap_th=0.7):
    f = fname
    # print('Spectral Clustering')
    # overlap =1
    
    results_dict = defaultdict(np.array)
    nframe = W.shape[0]
    # distance_matrix = (output_new+1)/2
    # output_new = output_new/np.max(abs(output_new))
    # distance_matrix = 1/(1+np.exp(-output_new))
    
    clean_ind = []
    pred_overlap_ind = []
    
    # uniq_labels,labelfull=unique(labelfull,True)
    
    # overlap_th = 0.7
    # mx = overlap_th*np.max(W,axis=1,keepdims=True)
    # mx = 0.2
    
    if len(clean_ind)==0:
        mx = overlap_th
        mask_overlap = (W>=mx)
        pred_overlap_ind = np.where(np.sum(mask_overlap,axis=1)>1)[0]
        
    label_withoverlap = np.ones((nframe,2),dtype=int)*(-1)
    
    
    label_withoverlap[:,0]=np.argmax(W,axis=1)
    uniq_labels =  np.unique(label_withoverlap[:,0])
    n_clusters = len(uniq_labels)
    # using predicted overlap labels
    if len(pred_overlap_ind)>0 and n_clusters > 1:
        sort_ind = np.argsort(W[pred_overlap_ind],axis=1)[:,::-1]
        label_withoverlap[pred_overlap_ind,0]=sort_ind[:,0]
        label_withoverlap[pred_overlap_ind,1]=sort_ind[:,1]
    
    labelfull = label_withoverlap
            
    clusterlen = []
    for lab in uniq_labels:
        clusterlen.append(len(np.where(labelfull==lab)[0]))
    # bp()
    print('clusterlen:',clusterlen,' n_clusters:',n_clusters)

    results_dict[f]=labelfull
    if final:
        out_file=args.outf+'/'+'final_clustering{}rttms/'.format(pref)
        rttm_valfile = out_file+'/valrttm'
        # out_file=args.outf+'/'+'final_spectral_test_rttms/'
    else:
        out_file=args.outf+'/'+'rttms{}/'.format(pref)
        rttm_valfile = out_file+'/valbaserttm'

    mkdir_p(out_file)
    # if not os.path.isdir(out_file):
    #     os.makedirs(out_file)
    outpath=out_file +'/'+f

    rttm_newfile=out_file+'/'+f+'.rttm'
    
    
    # rttm_gndfile = args.rttm_ground_path+'/'+f+'.rttm'
    # bp()
    # outpath = out_file +'/val_labels'
    write_results_dict_overlap(args,fname, out_file, results_dict, reco2utt)
    # self.write_results_dict(out_file)
    der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,0)
    if overlap:
        overlap_der = compute_score(args,rttm_gndfile,rttm_newfile,outpath,overlap)
        print("\n%s  overlap DER: %.2f" % (fname, overlap_der))
    print("\n%s DER: %.2f" % (fname, der))
            
    # bashCommand="cat {} >> {}".format(rttm_newfile,rttm_valfile)
    # os.system(bashCommand)
    # return rttm_valfile, outpath
    return overlap_der


