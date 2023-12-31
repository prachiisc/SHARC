import os
import argparse
import numpy as np
from pdb import set_trace as bp

def indices_with_intersecting_durs(seg_time_boundaries, rttm_bins, threshold):
    threshold = seg_time_boundaries[1]-seg_time_boundaries[0]
    intersect_values = np.minimum(seg_time_boundaries[1], rttm_bins[:,1]) - np.maximum(seg_time_boundaries[0], rttm_bins[:,0])
    return intersect_values, intersect_values >= threshold

def generate_labels(segmentsfile_fine, labelsfiledir, segmentsfile_coarse, threshold):
   
    if not os.path.exists(labelsfiledir):
        os.makedirs(labelsfiledir, 0o777)
    
    
    # print("\t\t Threshold for generating label is {}".format(threshold))
    segments = np.genfromtxt(segmentsfile_fine, dtype='str')
    utts = segments[:,0]
    segments = segments[:,1:]
    filenames = np.unique(segments[:,0])
    segment_boundaries =[]
    utts_filewise =[]
    for f in filenames:
        segment_boundaries.append((segments[:,1:][segments[:,0]==f]).astype(float))
        utts_filewise.append((utts[segments[:,0]==f])) 
        
    gt_segments_coarse = np.genfromtxt(segmentsfile_coarse, dtype='str')
    seg_idx = np.asarray([True,False,True,True])
    
    for i,f in enumerate(filenames):
        labelsfilepath = os.path.join(labelsfiledir, "fine2coarse_{}".format(f))
        if os.path.isfile(labelsfilepath):
            continue
        labels = open(labelsfilepath,'w')
        if i % 5 == 0:
            print("\t\t Generated labels for {} files".format(i))
       

        labels_f = []
        flag = []
         
        segments_coarse = gt_segments_coarse[gt_segments_coarse[:,1] == f] 
        segments_coarse = segments_coarse[:, seg_idx]
        
        for j in range(len(segment_boundaries[i])):
            _, label_idx = indices_with_intersecting_durs(segment_boundaries[i][j],segments_coarse[:,1:3].astype(float), threshold)
            labels_f = segments_coarse[label_idx][:,0]

            if np.sum(label_idx) > 2:
                label_f = np.unique(labels_f)                

            elif np.sum(label_idx) == 0:
                intersect_values, label_idx = indices_with_intersecting_durs(segment_boundaries[i][j],segments_coarse[:,1:3].astype(float),0)
                labels_f = segments_coarse[np.argmax(intersect_values)][0]
                labels_f = np.array([labels_f])
                


            towrite= "{} {}\n".format(utts_filewise[i][j], ' '.join(labels_f.tolist()))
            
            labels.writelines(towrite)
  
        
        labels.close()
    print('DONE with labels')


if __name__=="__main__":

    default_dataset="dihard_dev_2020_track1"
    kaldi_recipe_path="/data1/prachis/Dihard_2020/Dihard_2020_track1"
    threshold = 0.25
    default_segments_fine = "lists/{}_xvec0_5s/tmp/avg_segments".format(default_dataset)
    default_segments_coarse = "lists/{}/tmp/avg_segments".format(default_dataset)
    default_labels_dir = "ALL_GROUND_LABELS/{}_xvec0_5s/avg_mapping".format(default_dataset)

    print("In the label generation script...")
    parser = argparse.ArgumentParser(description='Speaker Label generation for embeddings')
    # parser.add_argument('--dataset', default=default_dataset, type=str, help='dataset', nargs='?')
    parser.add_argument('--segmentsfile_fine', default=default_segments_fine, type=str, metavar='PATH', help='path of the embedding segments file', nargs='?')
    parser.add_argument('--labelsfiledir', default=default_labels_dir, type=str, metavar='PATH', help='path of the labels file', nargs='?')
    parser.add_argument('--segmentsfile_coarse', default=default_segments_coarse, type=str, metavar='PATH', help='path of the ground truth rttm file', nargs='?')
    parser.add_argument('--threshold', default=threshold, type=float, metavar='N', help='threshold duration to assign label')

    args = parser.parse_args()
    generate_labels(**vars(args))
    
