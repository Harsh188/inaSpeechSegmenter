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

import os
import tempfile
import warnings
from subprocess import Popen, PIPE
import math

import numpy as np


#os.environ['SIDEKIT'] = 'theano=false,libsvm=false,cuda=false'
#from sidekit.frontend.io import read_wav
#from sidekit.frontend.features import mfcc
from sidekit_mfcc import read_wav, mfcc


def _wav2feats(wavname):
    """
    Extract features for wav 16k mono
    """
    print("\n# Starting _wav2feats")
    ext = os.path.splitext(wavname)[-1]
    
    assert ext.lower() == '.wav' or ext.lower() == '.wave'
    sig, read_framerate, sampwidth = read_wav(wavname)
    shp = sig.shape
    # wav should contain a single channel
    assert len(shp) == 1 or (len(shp) == 2 and shp[1] == 1)
    # wav sample rate should be 16000 Hz
    assert read_framerate == 16000
    # current version of readwav is supposed to return 4
    # whatever encoding is detected within the wav file
    assert sampwidth == 4
    #sig *= (2**(15-sampwidth))

    with warnings.catch_warnings() as w:
        # ignore warnings resulting from empty signals parts
        warnings.filterwarnings('ignore', message='divide by zero encountered in log', category=RuntimeWarning)
        _, loge, _, mspec = mfcc(sig.astype(np.float32), get_mspec=True)
        
    # Management of short duration segments
    difflen = 0
    if len(loge) < 68:
        difflen = 68 - len(loge)
        warnings.warn("media %s duration is short. Robust results require length of at least 720 milliseconds" %wavname)
        mspec = np.concatenate((mspec, np.ones((difflen, 24)) * np.min(mspec)))
        #loge = np.concatenate((loge, np.ones(difflen) * np.min(mspec)))

    print("## wav2feats done")
    return mspec, loge, difflen


def media2feats(medianame, tmpdir, start_sec_spec, stop_sec, ffmpeg, q, tid):
    """
    Convert media to temp wav 16k file and return features
    """
    print("\n# Starting media 2 feats")

    base, _ = os.path.splitext(os.path.basename(medianame))
    print('# Base Name:',base)
    print('# Medianame:',medianame)
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdirname:

        # ##### ADDED CODE: #####
        args = ["ffprobe","-i",medianame,"-v","quiet","-show_entries","format=duration",
       "-of","default=noprint_wrappers=1:nokey=1"]
        p = Popen(args, stdout=PIPE, stderr=PIPE)
        output,error = p.communicate()
        print("\n## Finished probing duration. Returncode:",p.returncode,"Output:",output)
        assert p.returncode == 0, error

        # Convert output (bytes) ==> (int)
        i_output = int(float(output.decode("utf-8").strip()))
        print('### Duration:',i_output)

        mel_output=[]
        start_sec=0
        tmpwav=''
        
        for i in range(math.ceil(i_output/2700)):
            # build ffmpeg command line
            args = [ffmpeg, '-y', '-i', medianame, '-ar', '16000', '-ac', '1']
            
            if(math.ceil(i_output/2700)-1 == i):
                start_sec = i*2700
                args += ['-ss', '%f' % start_sec]
                tmpwav = tmpdirname + '/' + base + '_'+str(start_sec)+'.wav'
            else:
                start_sec = i*2700
                args += ['-ss', '%f' % start_sec]

                stop_sec = start_sec+2700
                args += ['-to', '%f' % stop_sec]

                tmpwav = tmpdirname + '/' + base + '_'+str(start_sec)+'_'+str(stop_sec)+'.wav'
            
            if(int(start_sec_spec)>=int(start_sec)):
                print("\n#### File has been segmented for start time:",start_sec)
                continue
            print('\n#### tmpwav name:',tmpwav)
            args += [tmpwav]

            # launch ffmpeg
            p = Popen(args, stdout=PIPE, stderr=PIPE)
            output, error = p.communicate()
            assert p.returncode == 0, error

            # Get Mel Power Spectrogram and Energy
            try:
                q.put((tid,start_sec,stop_sec,tmpwav,_wav2feats(tmpwav)))

            except:
                print('Error in puting to queue')

    q.put(('done',tid))
    return mel_output
