#!/usr/bin/env python
# coding: utf-8

# In[13]:


# virtualenv sendsms
# source sendsms/bin/activate
# pip install twilio>=6.0.


# In[8]:


import twilio
from twilio.rest import Client


# In[12]:


# we import the Twilio client from the dependency we just installed
from twilio.rest import Client

# the following line needs your Twilio Account SID and Auth Token
client = Client("AC21341717d2b771c618b7e38a42b55a5c", "ed010b9513776c4257f5d29c26d2cdcc")

# change the "from_" number to your Twilio number and the "to" number
# to the phone number you signed up for Twilio with, or upgrade your
# account to send SMS to any phone number
to_list = ["+12172208821","+14803265816"]
client.messages.create(to="+12172208821", 
                       from_="+12179938734", 
                       body="PredictHERS")

