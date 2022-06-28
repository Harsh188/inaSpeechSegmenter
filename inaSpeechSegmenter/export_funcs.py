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
import os
from pytextgrid.PraatTextGrid import PraatTextGrid, Interval, Tier

def seg2csv(data,columns,fout=None):
    '''This method will take in the features, label segments, and store it
    within the specified output csv file. 
        
        Args:
            data (Dict): Dictionary holding data to put into csv
            columns (List): List of strings holding DataFrame column names
            fout (str): Output csv path.
    '''
    # Create new DF using data
    df = pd.DataFrame.from_dict(data=data)
    df = df.T
    df.columns=columns

    # Check if CSV exists:
    if(os.path.exists(fout)):
        print("CSV exists -- Appending")
        # Append to CSV
        # Load data from csv to DataFrame
        df2 = pd.read_csv(fout)

        # Append data to existing DataFrame
        df = df2.append(df)

    # Save to CSV
    print("Saving DF to CSV\n")
    print('DF',df.head())
    print('File',fout)
    df.to_csv(fout,sep=',',index=False)

def seg2textgrid(lseg, fout=None):
    tier = Tier(name='inaSpeechSegmenter')
    for label, start, stop in lseg:
        tier.append(Interval(start, stop, label))
    ptg = PraatTextGrid(xmin=lseg[0][1], xmax=lseg[-1][2])
    ptg.append(tier)
    ptg.save(fout)
