#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import os
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import librosa.display
loc="D:/audio2"
debussy="debussy.wav"
duke="duke.wav"
scale="scale.wav"
redhot="redhot.wav"
ipd.Audio(os.path.join(loc, duke))


# In[7]:


ipd.Audio(os.path.join(loc, debussy))


# In[8]:


ipd.Audio(os.path.join(loc,scale))


# In[9]:


ipd.Audio(os.path.join(loc,redhot))


# In[17]:


duke_sample,duke_sr=librosa.load(os.path.join(loc, duke))
debussy_sample,debussy_sr=librosa.load(os.path.join(loc, debussy))
scale_sample, scale_sr=librosa.load(os.path.join(loc,scale))
redhot_sample, redhot_sr=librosa.load(os.path.join(loc,redhot))
hop_size=512
frame_size=2048
s_scale=librosa.stft(scale_sample, 2048, 512)
s_scale.shape


# In[15]:


scale_sample.shape


# In[18]:


s_duke=librosa.stft(duke_sample, 2048, 512)


# In[19]:


s_debussy=librosa.stft(debussy_sample, 2048, 512)


# In[43]:


s_redhot=librosa.stft(redhot_sample, 2048, 512)
s_scale.shape


# In[44]:


y_scale=np.abs(s_scale) ** 2
y_duke=np.abs(s_duke) ** 2
y_debussy=np.abs(s_debussy) ** 2
y_redhot=np.abs(s_redhot) ** 2


# In[63]:


def plot_spectrogram(Y, sr, hop_length, y_axis="log"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,hop_length=hop_length,sr=sr,x_axis="time",y_axis=y_axis)
    plt.colorbar(format="%+2.f")
plot_spectrogram(y_scale, scale_sr, hop_size)


# In[71]:


log_scale=librosa.power_to_db(y_scale)
log_duke=librosa.power_to_db(y_duke)
log_debussy=librosa.power_to_db(y_debussy)
log_redhot=librosa.power_to_db(y_redhot)


# In[67]:


plot_spectrogram(log_scale, scale_sr, hop_size)


# In[72]:


plot_spectrogram(log_duke, scale_sr, hop_size)


# In[73]:


plot_spectrogram(log_debussy, scale_sr, hop_size)


# In[70]:


plot_spectrogram(log_redhot, scale_sr, hop_size)


# In[ ]:




