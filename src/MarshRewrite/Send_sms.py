#!/usr/bin/env python
# coding: utf-8
#Jayashree Subramanian

# In[13]:


# virtualenv sendsms
# source sendsms/bin/activate
# pip install twilio>=6.0.


# In[8]:


import twilio
from twilio.rest import Client


# In[12]:


# To import the Twilio client 
from twilio.rest import Client

# Twilio Account SID and Authentication Token
client = Client("AC21341717d2b771c618b7e38a42b55a5c", "ed010b9513776c4257f5d29c26d2cdcc")

to_list = ["+12172208821","+14803265816"]
client.messages.create(to="+12172208821", 
                       from_="+12179938734", 
                       body="PredictHERS")

