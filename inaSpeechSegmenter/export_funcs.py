#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2018 Ina (David Doukhan - http://www.ina.fr/)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pandas as pd
import numpy as np
import os
from pytextgrid.PraatTextGrid import PraatTextGrid, Interval, Tier

def feat2npy(mfcc,loge,difflen,start,stop,fout=None):
    '''

        Args:
            mfcc (np.array - float32) 
            loge (np.array - float32)
            difflen (int)
            start (int)
            stop (int)
            fout (str)
    '''
    # Create file names using start-stop
    mfcc_out = fout+'_'+str(start)+'_'+str(stop)+'_mfcc.npy'
    loge_out = fout+'_'+str(start)+'_'+str(stop)+'_loge.npy'
    print('\n--- Starting feat2npy ---')
    print("## Output:")
    print("### mfcc file path:",mfcc_out)
    print("### loge file path:",loge_out)

    # Save mfcc as npy
    np.save(mfcc_out,mfcc)

    # Save loge as npy
    np.save(loge_out,loge)

    # Save Paths to DF
    csv_out = fout+'_feats.csv'

    data = [start,stop,difflen,mfcc_out,loge_out]
    columns = ['start_second','stop_second','difflen','mfcc_path','loge_path']
    seg2csv(data,columns,csv_out)

def seg2csv(data,columns,fout=None,from_recs=False):
    '''This method will take in the features, label segments, and store it
    within the specified output csv file. 
        
        Args:
            data (Dict): Dictionary holding data to put into csv
            columns (List): List of strings holding DataFrame column names
            fout (str): Output csv path.
    '''
    print('\n--- Starting seg2csv ---')

    # Create new DF using data
    if(from_recs):
        df = pd.DataFrame.from_records(data,columns=columns)
    else:
        df = pd.DataFrame(data=data)
        df = df.T
        df.columns=columns

    # Check if CSV exists:
    if(os.path.exists(fout)):
        print("# CSV exists -- Appending")
        # Append to CSV
        # Load data from csv to DataFrame
        df2 = pd.read_csv(fout)

        # Append data to existing DataFrame
        df = df2.append(df)

    # Save to CSV
    print("## Saving DF to CSV\n")
    print('## DF',df.head())
    print('## File',fout)
    df.to_csv(fout,sep=',',index=False)

def seg2textgrid(lseg, fout=None):
    tier = Tier(name='inaSpeechSegmenter')
    for label, start, stop in lseg:
        tier.append(Interval(start, stop, label))
    ptg = PraatTextGrid(xmin=lseg[0][1], xmax=lseg[-1][2])
    ptg.append(tier)
    ptg.save(fout)
