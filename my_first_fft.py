#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
loc="D:/audio"
noise="noise.wav"
piano="piano.wav"
violin="violin.wav"
sax="sax.wav"
ipd.Audio(os.path.join(loc, noise))


# In[10]:


ipd.Audio(os.path.join(loc, piano))


# In[11]:


ipd.Audio(os.path.join(loc, violin))


# In[12]:


ipd.Audio(os.path.join(loc,sax))


# In[19]:


piano_samples,piano_sr=librosa.load(os.path.join(loc,piano))
noise_samples,noise_sr=librosa.load(os.path.join(loc,noise))
violin_samples,violin_sr=librosa.load(os.path.join(loc,violin))
sax_samples,sax_sr=librosa.load(os.path.join(loc,sax))


# In[15]:


piano_samples.shape


# In[18]:


piano_sr


# In[33]:


def graph(signal,sr,title,fbins):
    signal_fft=np.fft.fft(signal)
    signal_freqency=np.linspace(0,sr,len(signal))
    X=np.abs(signal_fft)
    plt.figure(figsize=(18,5))
    plt.plot(signal_freqency[:int(len(signal)*fbins)],X[:int(len(signal)*fbins)])
    plt.xlabel('Freqency(HZ)')
    plt.title(title)
graph(piano_samples,piano_sr,"piano",.1)


# In[34]:


graph(noise_samples,noise_sr,"noise",.5)


# In[35]:


graph(violin_samples,violin_sr,"violin",.5)


# In[36]:


graph(sax_samples,sax_sr,"sax",.5)


# In[ ]:




