# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:30:41 2019

@author:
    using to select event from semantic lines and visiuize LDA to check consistency
"""
import os
# import sys
import argparse
import json
import numpy as np
from LDA import lda_model, corp_dict
# import random as rd
# from gensim.models import CoherenceModel
import gensim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
# from datetime import Datetime
import datetime
import matplotlib as mpl
from scipy.special import comb


# select all the event point index
# inout: win_len: windows length for smoothing and choosing events range
# inputL gran: granuliarity
# input: band_width: the parameters set to choose detect event points: larger event point will be smaller
# input: event_back_len: the event backtrading length:
# output: all_time_arr: the overall array of time used for another function only
# output: event_time_set: list (set of event time)

def find_events(win_len, gran, band_width, event_back_len):  # select all the event point index

    inp = open('test.json', 'rb')
    data = json.load(inp)
    data = pd.DataFrame(data)
    data['time'] = pd.to_datetime(data.time.values)
    # get data prepared and LDA model ready
    labels = data['label'].values
    # get semantic-scaled
    # change neg's sign
    semantic_value = data['semantic_value'].values
    semantic_value = np.array([np.array(x) for x in semantic_value])
    semantic_arr = semantic_value.max(1)  # get semantic value ready
    neg_idx = np.where(labels == 0)  # 0 represent neg, 1 represent pos
    pos_idx = np.where(labels == 1)
    semantic_arr[neg_idx] = -semantic_arr[neg_idx]  # get full representative semantics
    data['semantic_arr'] = semantic_arr
    # scale
    # scale the data so the plot be more obvious / 分别对pos和neg部分scale，拉开差距
    neg_semantic = semantic_arr[neg_idx].reshape(-1, 1)
    pos_semantic = semantic_arr[pos_idx].reshape(-1, 1)
    pos_scaler = preprocessing.StandardScaler().fit(pos_semantic)
    neg_scaler = preprocessing.StandardScaler().fit(neg_semantic)

    pos_semantic = pos_scaler.transform(pos_semantic)
    pos_semantic = np.array([float(x) for x in pos_semantic])
    neg_semantic = neg_scaler.transform(neg_semantic)
    neg_semantic = np.array([float(x) for x in neg_semantic])

    # scale the read_num
    read_num = data['read_num'].values.reshape(-1, 1)
    read_num_scaler = preprocessing.StandardScaler().fit(read_num)
    read_num = read_num_scaler.transform(read_num)
    data['scale_read_num'] = read_num

    # get timr format prepared
    str_time = data['time'].values
    str_time = [str(x).split('.')[0] for x in str_time]
    myFormat = "%Y-%m-%d %H:%M"  # the most accurate gran is minute
    datetime_arr = [datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime(myFormat) for x in str_time]
    datetime_arr = [datetime.datetime.strptime(x, myFormat) for x in datetime_arr]  # change to datetime obj
    all_time_arr = pd.to_datetime(datetime_arr)


    time_index = []  # init
    if gran == 'day':
        time_index = [x.replace(hour=0, minute=0) for x in
                      datetime_arr]  # according to granularity choose the suitable time index
        time_index = pd.to_datetime(time_index)
    elif gran == 'hour':
        time_index = [x.replace(minute=0) for x in
                      datetime_arr]  # according to granularity choose the suitable time index
        time_index = pd.to_datetime(time_index)
    elif gran == 'min':
        time_index = pd.to_datetime(datetime_arr)  # time_index do not have to change

    # groupby data and get every day's semantic value
    st_time, ed_time = args.start_time, args.end_time
    _idx = (time_index <= ed_time) & (time_index >= st_time)

    # filter user time 1st and groupbyu
    test_data = data.loc[_idx,]  # using franuliarity
    test_data['time_index'] = time_index[_idx,]
    tmp = test_data.groupby('time_index').mean()  # get granu avg

    # smooth to plot
    w_lst = np.array([comb(win_len + 1, i) for i in range(1, win_len + 1)])  # weight arr for smoothing
    w_lst = w_lst / sum(w_lst)

    def _smooth(x):  # somooth
        return sum(np.array(x) * w_lst)

    tmp['smooth_scale_semantic'] = tmp['scale_semantic'].rolling(win_len, min_periods=win_len, center=True).apply(
        _smooth)

    plot_smooth_semantic(tmp, gran)
    tmp['power'] = args.beta * abs(tmp['scale_semantic'].values) + (1 - args.beta) * tmp['scale_read_num'].values

    ## 如果长度过小，直接返回极值作为event
    if tmp.shape[0] < 2 * max(event_back_len, win_len):
        event_time_lst = np.argmax(tmp['power'])
        return event_time_lst, all_time_arr

    power_avg = tmp['power'].rolling(window=event_back_len, min_periods=2).mean().shift()
    power_std = tmp['power'].rolling(window=event_back_len, min_periods=2).std().shift()
    up_band = np.array(power_avg + band_width * power_std)
    event_time_lst = np.where((tmp['power'].values > up_band) == True)[0].tolist()  # extract all event time index
    #event_idx_lst = []  # get each ecent's index: list of list
    event_time_set = []  # get all event time used to label

    for event_time in event_time_lst:
        event_time_set.append(tmp.index[event_time])
        # sel_idx = find_index(str(tmp.index[event_time]), gran, all_time_arr, win_len)
        # event_idx_lst.append(sel_idx)
    return event_time_set, all_time_arr

    # plot


#plot smooth_scale
def plot_smooth_semantic(tmp, gran):  # input groupby data to plot smooth semantics
    _xlabel = 'time' + 'granularity: ' + gran
    _path = 'semantic_plot--' + args.start_time + '--' + args.end_time + '--' + gran + '.jpg'
    mpl.rcParams['agg.path.chunksize'] = 10000
    mpl.use('pdf')  # for cmd command
    plt.figure(figsize=(20, 10), dpi=300)
    plt.plot(tmp['smooth_scale_semantic'])
    plt.xlabel(_xlabel)
    plt.ylabel('semantic_value')
    plt.savefig(_path)


###get index from one event
# input: sel_time type:str the event time
# input: gran: granuliarity
# input: windows length
# output: selected index for the current event
def find_index(sel_time, gran, win_len):
    if gran == 'day':
        _from = datetime.datetime.strptime(str(sel_time), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        _from = datetime.datetime.strptime(_from, '%Y-%m-%d')
        _from = _from + datetime.timedelta(days=-(win_len // 2))
        _to = _from + datetime.timedelta(days=win_len)
    elif gran == 'hour':
        _from = datetime.datetime.strptime(str(sel_time), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H')
        _from = datetime.datetime.strptime(_from, '%Y-%m-%d %H')
        _from = _from + datetime.timedelta(hours=-(win_len // 2))
        _to = _from + datetime.timedelta(hours=win_len)
    elif gran == 'min':
        _from = datetime.datetime.strptime(str(sel_time), '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M')
        _from = datetime.datetime.strptime(_from, '%Y-%m-%d %H:%M')
        _from = _from + datetime.timedelta(minutes=-(win_len // 2))
        _to = _from + datetime.timedelta(minutes=win_len)

    sel_idx = (all_time_arr < _to) & (all_time_arr >= _from)
    return sel_idx


#main program used to test funtion
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-st_time", '--start_time', type=str, default=None)
    parser.add_argument('-ed_time', '--end_time', type=str, default=None)
    parser.add_argument('-granu', '--granularity', type=str, default='min')

    parser.add_argument('-wl', '--win_len', type=int, default=5)  # must be odd
    parser.add_argument('-bt', '--beta', type=float,
                        default=0.7)  # 0~1 using to adjust the weight between read_num and semantic
    parser.add_argument('-bw', '--band_width', type=float, default=5.0)
    parser.add_argument('-ebl', '--event_back_len', type=int, default = 15)
    args = parser.parse_args()

    event_time_set, all_time_arr = find_events(args.win_len, args.granularity, args.band_width, args.event_back_len)

    if len(event_time_set) == 0:
        print('No event detcted! You can change the paramerers to loose the event detection criteria!')
    all_idx = []
    for event_time in event_time_set:
        sel_idx = find_index(event_time, args.gran, args.win_len)  # use to extract one index
        # event_idx_lst.append(sel_idx)
        all_idx.append(sel_idx)
