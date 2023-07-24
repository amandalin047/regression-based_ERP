# the bins in BDF must be mutually exclusive

from collections import OrderedDict
from collections import namedtuple
import numpy as np
import copy
import os
import mne
from re import findall
from amanda_rerp_ols import *

def parse_bdf(BDF_txt):
    f1 = open(BDF_txt)
    f2 = f1.read().split()
    f1.close()
    ev_seq, bins = [f2[3+4*i] for i in range(int((len(f2))/4))], [f2[2+4*i] for i in range(int((len(f2))/4))]
    tlock = [i.split('.') for i in ev_seq]
    
    parse1 = [[[] for j in i] for i in tlock]
    for i in range(len(tlock)):
        for j, y in enumerate(tlock[i]):
            parse1[i][j] = y.split('}')[:-1]
            
    parse2 = [[[] for j in i] for i in tlock]
    for i in range(len(tlock)):
        for j, y in enumerate(parse1[i]):
            for k, z in enumerate(y):
                if '-' in z:
                    l = int(findall(r'\d+', z)[0])
                    u = int(findall(r'\d+', z)[1])
                    parse2[i][j].append([str(n) for n in list(range(l, u+1))])
                else: parse2[i][j].append(findall(r'\d+', z))
    return parse2, bins


def raw_revised(raw, BDF_txt):
    bdf, bins = parse_bdf(BDF_txt)
    
    annot = copy.deepcopy(raw.annotations)
    items = {}
    for j, key in enumerate(annot[0].keys()): items[key] = []
    for i in range(len(annot)):
        for j, key in enumerate(annot[i].keys()):
            items[key].append(annot[i][key])
            
    ev = [bdf[i][1][0] for i in range(len(bdf))]
    
    for i in range(len(bdf)):
        for j in range(len(annot)):
            if annot[j]['description'] in bdf[i][1][0]:    # if the event code is the event time-locked to
                boo = [[[] for g in h] for h in bdf[i]]
                temp = [[True for g in h] for h in bdf[i]]
                for k, z in enumerate(bdf[i][0]):
                    try:
                        if annot[j-(len(bdf[i][0])-k)]['description'] in z: boo[0][k] = True
                        else: boo[0][k] = False
                    except IndexError:
                        boo[0] = None
                        break
                for k, z in enumerate(bdf[i][1]):
                    try:
                        if annot[j+k]['description'] in z: boo[1][k] = True
                        else: boo[1][k] = False
                    except IndexError:
                        boo[1] = None
                        break
                if boo == temp: items['description'][j] = bins[i]
            
    new_annot = mne.Annotations(np.array(items['onset'], dtype=object),
                                np.array(items['duration'], dtype=object),
                                np.array(items['description'], dtype=object), orig_time=None, ch_names=None)
    
    ev_arr, ev_dict = mne.events_from_annotations(raw, event_id='auto')
    for i, x in enumerate(items['description']):
        try:
            if x in bins: ev_arr[i][2] = int(x)
        except: pass
    
    new_raw = raw.copy().set_annotations(new_annot)
    return new_raw, bins


def bin_based_epoch(raw, BDF_txt, tmin, tmax, bc=None):
    new_raw, bins = raw_revised(raw, BDF_txt)
    ev_arr, ev_id = mne.events_from_annotations(new_raw)
    
    bdf = parse_bdf(BDF_txt)
    
    epoch = mne.Epochs(new_raw, ev_arr, event_id=[ev_id[k] for k in bins], tmin=tmin, tmax=tmax, baseline=bc,
                       reject_by_annotation=True)
    return epoch


def MWPtP(raw, epochs, ecodes, tmin, tmax, baseline, channel, step, win, thresh):   # ecodes must be a list of strings
    ch_list = raw.ch_names
    ch_dict = OrderedDict()
    for i in range(len(ch_list)): ch_dict.update({ch_list[i]: i})
    
    epochs_data = epochs.get_data()[:,:,0:int(1000*(tmax-tmin))]
    num_epochs, num_time = epochs_data.shape[0], epochs_data.shape[2]
    
    moves = int(1+(num_time-win)/step)
    ptps = np.empty((num_epochs, moves))
    for ep in range(num_epochs):
        for t in range(moves):
            move_win = epochs_data[ep, ch_dict[channel], step*t:step*t+win].copy().reshape((win))
            ptps[ep, t] = max(move_win)-min(move_win)
    
    raw_annot = raw.annotations
    description = np.array([raw_annot[i]['description'] for i in range(len(raw_annot))])
    
    indices = [i for i,x in enumerate(description) if x in ecodes ]  
    
    new_description = ['bad' if x in ecodes and len(np.where(ptps[indices.index(i)]>thresh)[0]) != 0 else x
                        for i, x in enumerate(description)]
    
    new_annot = mne.Annotations(np.array([raw_annot[j]['onset'] for j in range(len(raw_annot))]),
                            np.zeros(len(raw_annot)),
                            new_description,
                            orig_time=None, ch_names=None)

    new_raw = raw.copy().set_annotations(new_annot)
    events, event_id = mne.events_from_annotations(new_raw)
    
    new_epochs = mne.Epochs(new_raw, events, [event_id[k] for k in ecodes] , tmin, tmax, baseline=baseline,
                            reject_by_annotation=True)
    
    Epochs_Descriptions = namedtuple('Epochs_Descriptions', ['epochs', 'descriptions'])
    epochs_descriptions = Epochs_Descriptions(new_epochs, new_description)
    return epochs_descriptions


def compute_avg_erps(epoch, BDF_txt, binlabels=None):
    bdf = parse_bdf(BDF_txt)
    bins = [str(n) for n in list(range(301, 301+len(bdf)))]
    
    data = epoch.get_data()
    annot = epoch.annotations
    ev_codes = [annot[i]['description'] for i in range(len(annot)) if annot[i]['description'] in bins]
    
    dm = np.zeros((data.shape[0], len(bdf)))
    for i in range(data.shape[0]):
        for j in range(len(bins)):
            if ev_codes[i] == bins[j]: dm[i][j] = 1
            
    res = rerp_ols(epoch, dm, pred_names=binlabels)
    return res
