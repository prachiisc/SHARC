import os
import argparse
import numpy as np
from pdb import set_trace as bp

def indices_with_intersecting_durs(seg_time_boundaries, rttm_bins, threshold):
   
    rttm_bins[:,1] += rttm_bins[:,0]
    seg_dur = seg_time_boundaries[1] - seg_time_boundaries[0]
    intersect_values = np.minimum(seg_time_boundaries[1], rttm_bins[:,1]) - np.maximum(seg_time_boundaries[0], rttm_bins[:,0])
    threshold_values = min(seg_dur,threshold)
   
    return intersect_values, intersect_values >= threshold_values

def generate_labels(segmentsfile, labelsfiledir, ground_truth_rttm, threshold,overlap=None):
   
    if not os.path.exists(labelsfiledir):
        os.makedirs(labelsfiledir, 0o777)
    
    
    print("\t\t Threshold for generating label is {}".format(threshold))
    segments = np.genfromtxt(segmentsfile, dtype='str')
    utts = segments[:,0]
    segments = segments[:,1:]
    filenames = np.unique(segments[:,0])
    segment_boundaries =[]
    utts_filewise =[]
    gt_rttm = np.genfromtxt(ground_truth_rttm, dtype='str')
    rttm_idx = np.asarray([False,False,False,True,True,False,False,True,False, False])
   
    for i,f in enumerate(filenames):
        print('filename:',f)
        segment_boundaries.append((segments[:,1:][segments[:,0]==f]).astype(float))
        utts_filewise.append((utts[segments[:,0]==f]))
        labelsfilepath = os.path.join(labelsfiledir, "labels_{}".format(f))
        if os.path.isfile(labelsfilepath):
            continue
        labels = open(labelsfilepath,'w')
        if i % 5 == 0:
            print("\t\t Generated labels for {} files".format(i))
       

        labels_f = []
        flag = []
         
        rttm = gt_rttm[gt_rttm[:,1] == f] 
        try:
            rttm = rttm[:, rttm_idx]
        except:
            bp()
        for j in range(len(segment_boundaries[i])):
            intersect_values, label_idx = indices_with_intersecting_durs(segment_boundaries[i][j],rttm[:,0:2].astype(float), threshold)
            
            labels_f = rttm[label_idx][:,2]
            
            if np.sum(label_idx) > 2:
                labels_f = np.unique(labels_f)  
                # bp()              

            elif np.sum(label_idx) == 0:
                intersect_values, label_idx = indices_with_intersecting_durs(segment_boundaries[i][j],rttm[:,0:2].astype(float),0)
                labels_f = rttm[np.argmax(intersect_values)][2]
                labels_f = np.array([labels_f])
                
            towrite= "{} {}\n".format(utts_filewise[i][j], ' '.join(labels_f.tolist()))
            
            labels.writelines(towrite)
  
        
        labels.close()
    print('DONE with labels')

def generate_labels_overlap(segmentsfile, labelsfiledir, ground_truth_rttm, threshold,overlap=None):
   
    if not os.path.exists(labelsfiledir):
        os.makedirs(labelsfiledir, 0o777)
    
    
    print("\t\t Threshold for generating label is {}".format(threshold))
    segments = np.genfromtxt(segmentsfile, dtype='str')
    utts = segments[:,0]
    segments = segments[:,1:]
    filenames = np.unique(segments[:,0])
    segment_boundaries =[]
    utts_filewise =[]
    gt_rttm = np.genfromtxt(ground_truth_rttm, dtype='str')
    rttm_idx = np.asarray([False,True,True,True])
   
    for i,f in enumerate(filenames):
        print('filename:',f)
        segment_boundaries.append((segments[:,1:][segments[:,0]==f]).astype(float))
        utts_filewise.append((utts[segments[:,0]==f]))
        labelsfilepath = os.path.join(labelsfiledir, "labels_{}".format(f))
        if os.path.isfile(labelsfilepath):
            continue
        labels = open(labelsfilepath,'w')
        if i % 5 == 0:
            print("\t\t Generated labels for {} files".format(i))
       

        labels_f = []
        flag = []
         
        rttm = gt_rttm[gt_rttm[:,0] == f] 
        try:
            rttm = rttm[:, rttm_idx].astype(float)
            rttm[:,1] = rttm[:,1] - rttm[:,0] # keep it like rttm, start, duration
        except:
            bp()
        for j in range(len(segment_boundaries[i])):
            intersect_values, label_idx = indices_with_intersecting_durs(segment_boundaries[i][j],rttm[:,0:2].astype(float), threshold)
            
            labels_f = rttm[label_idx][:,2].astype(int)
            labels_f = labels_f.astype(str)
            if np.sum(label_idx) > 1:
                labels_f = np.unique(labels_f)            
            elif np.sum(label_idx) == 0:
                # intersect_values, label_idx = indices_with_intersecting_durs(segment_boundaries[i][j],rttm[:,0:2].astype(float),0)
                # labels_f = rttm[np.argmax(intersect_values)][2]
                labels_f = np.array(['0'])
                
            towrite= "{} {}\n".format(utts_filewise[i][j], ' '.join(labels_f.tolist()))
            
            labels.writelines(towrite)
  
        
        labels.close()
    print('DONE with labels')


if __name__=="__main__":

 
    default_dataset="dihard_dev_2020_track1"
    
    kaldi_recipe_path="/data1/prachis/Dihard_2020/Dihard_2020_track1"
    threshold = 0.5
    #default_segments = "lists/{}_xvec0_5s/tmp/segments".format(default_dataset)
    default_segments = "lists/{}/tmp/avg_segments".format(default_dataset)
    default_gt_rttm = "{}/data/{}/rttm".format(kaldi_recipe_path,default_dataset)
    #default_labels_dir = "ALL_GROUND_LABELS/{}_xvec0_5s/threshold_{}".format(default_dataset,threshold)
    default_labels_dir = "ALL_GROUND_LABELS/{}/threshold_{}_avg".format(default_dataset,threshold)

    print("In the label generation script...")
    parser = argparse.ArgumentParser(description='Speaker Label generation for embeddings')
    # parser.add_argument('--dataset', default=default_dataset, type=str, help='dataset', nargs='?')
    parser.add_argument('--segmentsfile', default=default_segments, type=str, metavar='PATH', help='path of the embedding segments file', nargs='?')
    parser.add_argument('--labelsfiledir', default=default_labels_dir, type=str, metavar='PATH', help='path of the labels file', nargs='?')
    parser.add_argument('--ground_truth_rttm', default=default_gt_rttm, type=str, metavar='PATH', help='path of the ground truth rttm file', nargs='?')
    parser.add_argument('--threshold', default=threshold, type=float, metavar='N', help='threshold duration to assign label')
    parser.add_argument('--overlap', action='store_true')
    args = parser.parse_args()
   
    if args.overlap:
        # to generate label files using pyannote-overlap labels
        generate_labels_overlap(**vars(args))
    else:
         generate_labels(**vars(args))




    
