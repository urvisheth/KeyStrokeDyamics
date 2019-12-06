#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pyHook
import pythoncom
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import pandas as pd
from IPython.display import clear_output
import csv


# In[3]:


global userName
userFilePath = "Collecting_keyStorke.csv"


# In[ ]:





# In[6]:


class KeyLogger:
    def __init__(self):
        self.enterPressed = False
        self.eventList = []
        self.isCaps = False
        #self.message = ""
        
    def keyDownEvent(self, event):
        if event.KeyID == 20 and self.isCaps == False:
            self.isCaps = True
        elif event.KeyID == 20 and self.isCaps == True:
            self.isCaps = False     
        if event.KeyID>= 48 and event.KeyID<=57:
            event.Ascii = event.KeyID
        if self.isCaps == True and event.Ascii>=97 and event.Ascii<=122:
            event.Ascii = event.KeyID
        self.storeEvent("Down", event) 
        return True
        # Fixes Requires Integer Bug (Got Nonetype)

    def keyUpEvent(self, event): 
        if event.KeyID>= 48 and event.KeyID<=57:
            event.Ascii = event.KeyID
        if self.isCaps == True and event.Ascii>=97 and event.Ascii<=122:
            event.Ascii = event.KeyID
        print(chr(event.Ascii),end='')
        self.storeEvent("Up", event)
        return True

    def mainLoop(self):
        while not self.enterPressed:
            pythoncom.PumpWaitingMessages()

    def storeEvent(self, activity, event):
        keystrokeTime = int(event.Time)
        self.eventList.append ((userName,event.Ascii,activity, int(keystrokeTime)))

        # Chosen to use Escape key (ESC) due to input using a similar method
        # Enter Key - KeyCode: 13 Ascii: 13 ScanCode: 28 - ESC = 27 @ Ascii
        if event.Ascii == 27:
            self.enterPressed = True
            userRecordData(self.eventList)

def userRecordData(eventList):
    print("\nouput")
    print(eventList)
    with open(userFilePath,'a',newline='\n') as f:
        writer = csv.writer(f)
        writer.writerows(eventList)
    f.close()   
    
def getUserName():
    global userName
    userName = input("Enter your Name: ")

def getKeyStroke():
    
    keyLogger = KeyLogger()
    hookManager = pyHook.HookManager()
    hookManager.KeyDown = keyLogger.keyDownEvent
    hookManager.KeyUp = keyLogger.keyUpEvent
    hookManager.HookKeyboard()

    keyLogger.mainLoop()
    # Unhooks the keyboard, no more data recorded, returns to menu
    hookManager.UnhookKeyboard()
    
getUserName()
print("Enter your text: ")
getKeyStroke()


# In[ ]:





# In[ ]:





# In[ ]:




