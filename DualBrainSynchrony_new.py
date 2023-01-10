from pylsl import StreamInlet, resolve_stream
import numpy as np
import pandas as pd
pitchlili = []
import math
from pythonosc import osc_message_builder
from pythonosc import udp_client
from numpy import mean
from scipy.fft import fft
import mne_connectivity as mc
import scipy
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs
from mne.datasets import sample
from scipy.signal import butter, lfilter
from scipy.signal import butter, sosfilt, sosfreqz
import warnings
import socket, struct, os
import numpy as np
import matplotlib.pyplot as plt




#####


import socket, struct, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

udp_socket2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
localaddr = ("192.168.43.219", 15002)

udp_socket2.bind(localaddr)
print("wow much receive")
count = 0
data_collect = []



####


def Otherbrain():
    print("wow much receive 2")
    timelili = []
    eeglili = []
    counterr = 0
    while True:
	
        recv_data = udp_socket2.recvfrom(1024)
        includee = False
        #print(recv_data.decode())
        #newstring = ""
        #for a in recv_data:
        lili = []    
        #df = pd.DataFrame(recv_data[0].decode())
        stringeru = recv_data[0].decode()
        my_list = stringeru.split("\n")
        my_list2 = my_list[0].split("\t")
        #print(str(my_list2))
        newword = ""
	
        for a in str(my_list2):
            if a == "[":
                start = True
            elif a == "\'": 
                start = False
            elif a == "]":
                start = False
            
            else:
                newword += a
            #print(newword)
            
            
        timee = True
        timeword = ""
        eegword = ""
        for a in newword:
            if a == " ":
                timee = False
            elif timee == True:
                timeword += a
            else:
                eegword += a
                
        timelili.append(timeword)
     
        eeglili.append(eegword)
        timee = True
        counterr += 1
        if counterr == 250:
            df = pd.DataFrame(eeglili)
            df.to_csv("DuoData.csv")
            timelili = []
            eeglili = []
            counter = 0
        
        #print(eeglili)
    
    udp_socket2.close()


from threading import Thread
t = Thread(target = Otherbrain)
t.start()
print("yayeet2")

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
localaddr = ("192.168.43.219", 15001)

udp_socket.bind(localaddr)

print("yayeet")

warnings.filterwarnings("ignore")

#Initialising variables
k = 0
h = 0
j = -1
pitchlist = []
FAAlili = []
udplist = []

# initialize the streaming layer
finished = False
streams = resolve_stream()
inlet = StreamInlet(streams[0])


#Setting sample frequency
n_samples = 250


# initialize the colomns of data and dictionary to capture the data.
columns=['Time','FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8','AccX','AccY','AccZ','Gyro1','Gyro2','Gyro3', 'Battery','Counter','Validation']
data_dict = dict((k, []) for k in columns)

#Keeps running until finished is True
while not finished:
   # Get the streamed data.
   # concatenate timestamp and data in 1 list
   data, timestamp = inlet.pull_sample()
   all_data = [timestamp] + data
   
#Create function for bandpass filtering
   def butter_bandpass(signal):
        #Sampling frequency
        fs = 250
        
        #Nyquist frequency
        nyq = 0.5 * fs
 
        #Lower bandpass threshold
        lowcut = 1.0
        low = lowcut / nyq
        #Upper bandpass threshold
        highcut =35.0
        high = highcut / nyq
        
        #Order of filtering (2 is chosen for processing speed)
        order = 2
        
        #Create Butterworth filter
        b, a = scipy.signal.butter(order, [low, high], "bandpass", analog = False)
        #Apply butterworth filter to data
        y = scipy.signal.filtfilt(b, a, signal, axis = 0)
        #Returns bandpassed filtered data
        return y
        

#Create function for Frontal Alpha Asymmetry (Measure for valence)   
   def FrontalAlphaAsymmetry(left, right):
    lefthem = np.array(left)
    lefthem = lefthem[~np.isnan(lefthem)]
    righthem = np.array(right)
    righthem = righthem[~np.isnan(righthem)]
    FFTLefthem = np.fft.fft(lefthem).real
    FFTRighthem = np.fft.fft(righthem).real
    
    
    FFTLefthem = np.nanmean(FFTLefthem[8:12])
    FFTRighthem = np.nanmean(FFTRighthem[8:12])
    #
    
    #FFTLefthem = np.log(abs(FFTLefthem))
    #FFTRighthem = np.log(abs(FFTRighthem))
    #print("left hem: ", FFTLefthem)
    #print("right hem: ", FFTRighthem)
    print(FFTRighthem, FFTLefthem)
    FrontalAlpha = np.log(abs(FFTRighthem) /abs(FFTLefthem))
    return FrontalAlpha
   
   # updating data dictionary with newly transmitted samples
  

   i = 0
   for keyz in list(data_dict.keys()):
      data_dict[keyz].append(all_data[i])
      i = i + 1
   j += 1
   if j >= 250:
     
    df = pd.DataFrame.from_dict(data_dict)
    
        
    #selecting the columns relevant to the channels
    cols = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    

    #create a new column with the average value for each channel
    df["average"] = df[cols].mean(axis=1)
    df.dropna()       
    faa = FrontalAlphaAsymmetry(df["C3"][j-250  : j],df["C4"][j-250:j])   #LEFT AT C NOW. FOR ULTIMATE DEMO SHOULD USE F3 AND F4
    #pitch calculated according to Fechner's law
   # print(FAA)
    if math.isnan(faa) == True:
        faa = 1
        print("FAA NAN detected")
    
    
        
    #print(pitchlist)
    #print(FFTSamples)
    #print(np.mean(amplitudes[8:12]))
    #print(np.log(np.mean(amplitudes[8:12])))
    #print("here is the pitch: ", Pitch)
    #print(pitchlist)
    import numpy as np
    import scipy.signal as sig
    
    #print("Pitch is done")
  
   
    
    #bandpass applied to limit the signal to EEG data    
    yee = butter_bandpass(np.array(df["FZ"]))
    yee2 = butter_bandpass(np.array(df["C3"]))
    yee3 = butter_bandpass(np.array(df["CZ"]))
    yee4 = butter_bandpass(np.array(df["C4"]))
    yee5 = butter_bandpass(np.array(df["PZ"]))
    yee6 = butter_bandpass(np.array(df["PO7"]))
    yee7 = butter_bandpass(np.array(df["OZ"]))
    yee8 = butter_bandpass(np.array(df["PO8"]))#'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8'
    #update data based on filtered data
    df["FZ"] = yee
    df["C3"] = yee2
    df["CZ"] = yee3
    df["C4"] = yee4
    df["PZ"] = yee5        
    df["PO7"] = yee6
    df["OZ"] = yee7
    df["PO8"] = yee8
    #print("Bandpass is done")
        #butter_bandpass_filter(df["FZ"][j-500:j])
    #select 250 samples in two channels to detect sybchrony between brain regions
    df = df.mean(axis = 1)
    y1 = df[j-250:j]
    #print("This is y1 which should show 1 1 column of 250 samples: ", y1)
    #indices = ["y1", "y2"]
    #print(y2)
    #function to get the Phase Locking Value (PLV) 
    def hilphase(y1):
        try:
            y2 = pd.read_csv("DuoData.csv")
        except:
            time.sleep(0.5)
            y2 = pd.read_csv("DuoData.csv")
        y2 = y2.iloc[:,0]
        #y2.close()
        sig1_hill=sig.hilbert(y1)
        sig2_hill=sig.hilbert(y2)
        #print ("y1: ", y1.shape)
        #print("y2: ", y2.shape)
        #print(y2)
        pdt=(np.inner(sig1_hill,np.conj(sig2_hill))/(np.sqrt(np.inner(sig1_hill,
                np.conj(sig1_hill))*np.inner(sig2_hill,np.conj(sig2_hill)))))
        phase = np.abs(np.angle(pdt))
        #print(phase)

        return phase
    
    k += 1
    #every 250 samples, check PLV and produce sounds based on the notes
    if k == 1:
        #print(df)
        #print(round(mean(pitchlist)))
        #note = round(mean(pitchlist))
        #pitchlist = []
        #soundlist = [62,64,66,67,69,71,73,74,62,64,66,67,69,71,73,74]
        phase = hilphase(y1)
        print("phase: ", phase)
        sender = udp_client.SimpleUDPClient("192.168.43.219", 4560)
        #sender.send_message('/trigger/prophet', [61+note, 100, 1,61+note+6  ])
        #f = note
        #adjustement of notes based on synchrony 
        #C scales

        ionian = [ "62", "64", "65", "67", "81", "83", "72"]  #Bright, Joyful, Stable
        dorian = [ "62", "63" ,"65", "67", "81", "82", "72"] #Jazzy, Bluesy, Rocky, Thoughtful, Uncertain, Sophisticated
        phrygian = [ "61" ,"63", "65", "67", "80", "82", "72"]  #Exotic, Latin, Lively, Dark, Mystic
        lydian =  [ "62" , "64" , "66" , "67" ,"81" ,"83", "72"] # Hopeful, Dreamy, Heavenly, Yearning, Ethereal, Uplifting
        mixolydian = ["62" , "64" , "65" , "67" , "81" , "82", "72"] #Positive, Bluesy, Rocky, Poppy, Searching 
        aeolian = [ "62" , "63" , "65" , "67" , "80" , "82", "72"] #Sad, Melancholic, Romantic, Oppressive
        locrian = [ "61" , "63" , "65" , "66" , "80" , "82", "72"] #Complex, Unstable, Exotic, Tense
        scales = {'ionian' : ionian, 'dorian' : dorian, 'phrygian': phrygian, 'lydian': lydian, 'mixolydian':mixolydian, 'aeolian':aeolian,'locrian':locrian}
        phase = 50
        def getStep(num):
            step = 2/7
            return 1-num*step
        def get_Scale(faa):
            if faa >= getStep(1):
                return scales.get('ionian')
                
            elif faa <= getStep(1) and faa >= getStep(2):
                return scales.get('dorian')
                
            elif faa <= getStep(2) and faa >= getStep(3):
                return scales.get('phrygian')
                
            elif faa <= getStep(3) and faa >= getStep(4):
                return scales.get('lydian')
                
            elif faa <= getStep(4) and faa >= getStep(5):
                return scales.get('mixolydian')
                
            elif faa <= getStep(5) and faa >= getStep(6):
                return scales.get('aeolian')
                
            elif faa <= getStep(6) and faa >= getStep(7):
                return scales.get('locrian')
            else:
                return scales.get('ionian')
    
    
        helpi= abs(np.log(abs(FAA)))
        if helpi >= 1.5:
            helpi = 1.5
        elif helpi <=1:
            helpi = 1
        #else:
         #   print("valence was used")
        loglog = (np.mean(amplitudes[1:35])/100)
        if loglog != 0:
            #Pitch = (((-40/helpi)*np.log(loglog)+10))
            Pitch = (((-40/1.10)*np.log(loglog)+10))
        #print(Pitch)
        #collect list of pitches for normalization
        if j <= 10000:
            pitchlili.append(Pitch)
        #print(pitchlili)
        mini = min(pitchlili)
        maxi = max(pitchlili)
        #normalization
        Pitch = (Pitch - mini)/ (maxi-mini)
        #print(Pitch)
        if not math.isnan(Pitch):
            Pitch = round(Pitch* 128+ 20)
            pitchlist.append(Pitch)
        
        
        import random
        my_list = ['a'] * (100-int(phase)) + ['b'] * int(phase)
        isC = random.choice(my_list)
        if isC == "b":
            Note1 = ":C4"
        else:
            print("faa", faa)
            print(get_Scale(faa))
            scale = get_Scale(faa)
            print(scale)
            Note1 = random.choice(scale)
            print(Note1)
        
        if isC == "b":
            Note2 = ":C4"
        else:
            Note2 = random.choice(scale)
        #:C4
        
        sender.send_message('/trigger2/prophet', [Note1, Note2])
        
        #sender.send_message('/trigger/prophet2', [soundlist[note+2], 100, 1])
        #sender.send_message('/trigger/prophet', [soundlist[note+4], 100, 1])
        
        k = 0
    #df["C3"]
   #print(data_dict)
   # data is collected at 250 Hz. Let's stop data collection after 10 seconds. Meaning we stop when we collected 250*10 samples.
   if (len(data_dict['Time']) >= 250*60):
      finished = True
      
      

# lastly, we can save our data to a CSV format.
data_df = pd.DataFrame.from_dict(data_dict)
data_df.to_csv('EEGdata.csv', index = False)