                    #!/usr/bin/env python3
# coding=utf-8
# author: flk24@cam.ac.uk, ql264@cam.ac.uk
"""Run Spectral Clustering"""

import os
import sys
import argparse
import json
import itertools
import numpy as np
from tqdm import tqdm
# import kaldiio
import utils
from SpectralCluster.spectralcluster import SpectralClusterer
from scipy.special import expit
from pdb import set_trace as bp
def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='Do speaker clsutering based on'\
                                                    'refined version of spectral clustering',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparser.add_argument('--gauss-blur', help='gaussian blur for spectral clustering',
                           type=float, default=0.1)
    cmdparser.add_argument('--p-percentile', help='p_percentile for spectral clustering',
                           type=float, default=0.95)
    cmdparser.add_argument('--custom-dist', help='e.g. euclidean, cosine', type=str, default=None)
    cmdparser.add_argument('--reco2utt', help='spk2utt to create labels', default=None)
    cmdparser.add_argument('--reco2num', help='reco2num_spk to find oracle speakers',type=str, default='None')
    cmdparser.add_argument('--label-out', dest='output_file',
                           help='output file used for storing labels', default=None)
    cmdparser.add_argument('--minMaxK', nargs=2, default=[1, 10])
    cmdparser.add_argument('--score_file', help='file containing list of score matrices', type=str)
    cmdparser.add_argument('--score_path', help='path of scores', type=str)
    cmdparser.add_argument('--out_path', help='path of output scores', type=str, default=None)
    cmdparser.add_argument('--stop_eigenvalue', help='threshold for clustering', type=float, default=1e-2)
    cmdargs = cmdparser.parse_args()
    # setup output directory and cache commands
    if cmdargs.output_file is not None:
        outdir = os.path.dirname(cmdargs.output_file)
        utils.check_output_dir(outdir, True)
        utils.cache_command(sys.argv, outdir)
    return cmdargs

def do_spectral_clustering(dvec_list, gauss_blur=1.0, p_percentile=0.95,
                           minclusters=2, maxclusters=4, truek=4, custom_dist=None,stop_eigenvalue=1e-2,
                           scoretype=None, custom_dist_maxiter = (2,10),clean_ind=None):
    """Does spectral clustering using SpectralCluster, see import"""
    if minclusters < 1 and maxclusters < 1:
        if truek == 1:
            return [0] * dvec_list.shape[0]
        clusterer = SpectralClusterer(min_clusters=truek, max_clusters=truek,
                                      p_percentile=p_percentile,
                                      gaussian_blur_sigma=gauss_blur, custom_dist=custom_dist)
    else:
        clusterer = SpectralClusterer(min_clusters=minclusters, max_clusters=maxclusters,
                                      p_percentile=p_percentile,
                                      gaussian_blur_sigma=gauss_blur, custom_dist=custom_dist,stop_eigenvalue=stop_eigenvalue,
                                      custom_dist_maxiter=custom_dist_maxiter)
    
    if scoretype is None:
        return clusterer.predict_withscores(dvec_list)
    elif scoretype == 'laplacian':
        return clusterer.predict_withscores_laplacian(dvec_list)
    elif scoretype == 'cmeans':
        return clusterer.predict_with_cmeans(dvec_list)
    elif scoretype == 'softkmeans':
        return clusterer.predict_with_softkmeans(dvec_list)
    elif scoretype == 'softkmeans_v2':
        return clusterer.predict_with_softkmeans_v2(dvec_list)
    else:
        return clusterer.predict_with_softkmeans_modified(dvec_list,clean_ind)
    
def permutation_invariant_seqmatch(hypothesis, reference_list):
    """For calculating segment level error rate calculation"""
    num_perm = max(4, len(set(hypothesis)))
    permutations = itertools.permutations(np.arange(num_perm))
    correct = []
    for permutation in permutations:
        mapping = {old:new for old, new in zip(np.arange(num_perm), permutation)}
        correct.append(sum([1 for hyp, ref in zip(hypothesis, reference_list)
                            if mapping[hyp] == ref]))
    return max(correct)

def evaluate_spectralclustering(args):
    """Loops through all meetings to call spectral clustering function"""
    total_correct = 0
    total_length = 0
    with open(args.injson) as _json_file:
        json_file = json.load(_json_file)
    results_dict = {}
    for midx, meeting in tqdm(list(json_file["utts"].items())):
        meeting_input = meeting["input"]
        meeting_output = meeting["output"]
        assert len(meeting_input) == 1
        assert len(meeting_output) == 1
        meeting_input = meeting_input[0]
        meeting_output = meeting_output[0]
        cur_mat = kaldiio.load_mat(meeting_input["feat"])#(samples,features)
        reference = meeting_output["tokenid"].split()
        reference = [int(ref) for ref in reference]
        assert len(reference) == cur_mat.shape[0]
        if len(reference) == 1:
            results_dict[midx] = [0]
            continue
        try:
            hypothesis = do_spectral_clustering(cur_mat,
                                                gauss_blur=args.gauss_blur,
                                                p_percentile=args.p_percentile,
                                                minclusters=int(args.minMaxK[0]),
                                                maxclusters=int(args.minMaxK[1]),
                                                truek=len(set(reference)),
                                                custom_dist=args.custom_dist)
        except:
            print("ERROR:: %s %s" % (str(reference), str(cur_mat)))
            raise
        results_dict[midx] = hypothesis
        _correct = permutation_invariant_seqmatch(hypothesis, reference)
        total_length += len(reference)
        total_correct += _correct
    print("Total Correct: %s, Total Length: %s, Percentage Correct: %s" %
          (str(total_correct), str(total_length), str(total_correct * 100 / total_length)))
    return results_dict

def refine_scores(args):
    """Loops through all meetings to call spectral clustering function"""

    score_list= open(args.score_file).readlines()
    scorepath = args.score_path
        # json_file = json.load(_json_file)

    for score in score_list:
        score = score.rstrip()
        # bp()
        # if score=='iadm':
        #     bp()
        cur_mat=np.load(scorepath+'/'+score+'.npy')
        clusterer = SpectralClusterer(min_clusters=int(args.minMaxK[0]), max_clusters=int(args.minMaxK[1]),
                                      p_percentile=args.p_percentile,
                                      gaussian_blur_sigma=args.gauss_blur, custom_dist=args.custom_dist)
        refine = clusterer.refinementonly(cur_mat,0)

        np.save(args.out_path+'/'+score+'.npy',refine)


def evaluate_spectralclusteringwithscores(args):

    """Loops through all meetings to call spectral clustering function"""

    score_list= open(args.score_file).readlines()
    scorepath = args.score_path
    if args.reco2num != 'None':
        reco2num_spk = open(args.reco2num).readlines()
       
        # json_file = json.load(_json_file)
    results_dict = {}
    for i,score in enumerate(score_list):
        score = score.rstrip()
        if args.reco2num != 'None':            
            n_clusters = int(reco2num_spk[i].split()[1])
            minK = n_clusters
            maxK = n_clusters
        else:
            minK = int(args.minMaxK[0])
            maxK= int(args.minMaxK[1])
     
        cur_mat=np.load(scorepath+'/'+score+'.npy')
        # cur_mat = (cur_mat +1)/2
        # cur_mat = (cur_mat-np.min(cur_mat))
        # cur_mat = cur_mat/np.max(cur_mat)
        # cur_mat = 1/(1+np.exp(-5*cur_mat))
        cur_mat = expit(cur_mat)
        hypothesis = do_spectral_clustering(cur_mat,
                                                gauss_blur=args.gauss_blur,
                                                p_percentile=args.p_percentile,
                                                minclusters=minK,
                                                maxclusters=maxK,
                                                truek=4,
                                                custom_dist=args.custom_dist,
                                                stop_eigenvalue=args.stop_eigenvalue)


        results_dict[score] = hypothesis
    #     _correct = permutation_invariant_seqmatch(hypothesis, reference)
    #     total_length += len(reference)
    #     total_correct += _correct
    # print("Total Correct: %s, Total Length: %s, Percentage Correct: %s" %
    #       (str(total_correct), str(total_length), str(total_correct * 100 / total_length)))
    return results_dict

def write_results_dict(results_dict, output_json):
    """Writes the results dictionary into json file"""
    output_dict = {"utts":{}}
    for meeting_name, hypothesis in results_dict.items():
        hypothesis = " ".join([str(i) for i in hypothesis]) + " 4"
        output_dict["utts"][meeting_name] = {"output":[{"rec_tokenid":hypothesis}]}
    with open(output_json, 'wb') as json_file:
        json_file.write(json.dumps(output_dict, indent=4, sort_keys=True).encode('utf_8'))
    return

def write_results_dict_(results_dict, output_file,reco2utt):
    """Writes the results in label file"""

    output_label = open(output_file,'w')
    reco2utt = open(reco2utt,'r').readlines()
    i=0
    for meeting_name, hypothesis in results_dict.items():
        # bp()
        reco = reco2utt[i].split()[0]
        utts = reco2utt[i].rstrip().split()[1:]
        if reco == meeting_name:
            for j,utt in enumerate(utts):
                towrite = utt +' '+str(hypothesis[j])+'\n'
                output_label.writelines(towrite)
        i=i+1




def main():
    """main"""
    args = setup()
    # results_dict = evaluate_spectralclustering(args)
    # bp()

    results_dict = evaluate_spectralclusteringwithscores(args)

    if args.output_file is not None:
        write_results_dict_(results_dict, args.output_file,args.reco2utt)
    # bp()
if __name__ == '__main__':
    main()
    # refine_scores(setup())

