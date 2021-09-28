"""
Read .csv file and save time domain signals

"""
import pandas as pd # for file manipulation
import numpy as np # for math
import scipy as sp # for math
import matplotlib.pyplot as plt # for plotting
import scipy.signal as sps # for signal processing
import glob # for file name wild card 
import mysignalprocessing as msp # for self-reference fuctions
import unicodedata # for greek letter
import time

def readcsv(  file_destination
            , trigger = True
            , trigger_level = 0.1 #'f_max','v_max', or value
            , delay_time = 0.01 # in second
            , signal_length = None # in second
            , file_convert = False
            , save_name = None
            , reference_channel = 1
            , response_channel = 2
            , signal_extension = True):
    
    # Checking for Data Beginning Row-Number
    for n in range(10,25):
        if pd.read_csv(file_destination, header=None, skiprows=n, nrows=1).iloc[0,0] == 0:
            break
    
    # Data Acquisition
    data = pd.read_csv(file_destination, header=None, skiprows=n)
    
    # Sampling Rate
    sampling_period = data.iloc[1, 0]
    
    # Reading Signals
    res_time_domain = np.array(data[ response_channel])
    exc_time_domain = np.array(data[reference_channel])
    
    # Data Clipping
    start_index = 0
    stop_index = None
    delay_index = int(np.ceil(delay_time/sampling_period))
    
    # Check trigger
    if trigger == True:
        # Mark trigger_index
        if trigger_level == 'f_max':
            trigger_index = np.where( abs(exc_time_domain) == max(exc_time_domain))[0][0]
        elif trigger_level == 'v_max':
            trigger_index = np.where( abs(res_time_domain) == max(res_time_domain))[0][0]
        elif max(exc_time_domain) >= trigger_level:
            trigger_index = np.where( abs(exc_time_domain) >= trigger_level )[0][0]
        else:
            return print("Error file:"+file_destination+", signal is too low")
        
        # start and stop indice
        if delay_index > trigger_index:
            if signal_extension == False:
                print("Delay time is too large. Signals start at the beginning.")
            else:
                res_time_domain = np.insert( res_time_domain, 0, np.zeros(delay_index - trigger_index) )
                exc_time_domain = np.insert( exc_time_domain, 0, np.zeros(delay_index - trigger_index) )
                print("Delay time is too large. Zeros are added.")        
        else:
            start_index = trigger_index - delay_index

        if signal_length != None:
            stop_index = start_index + int(np.ceil(signal_length/sampling_period))

    res_time_domain = res_time_domain[start_index:stop_index]
    exc_time_domain = exc_time_domain[start_index:stop_index]
    
    "Time Range"
    time = np.arange(0, sampling_period * exc_time_domain.shape[0], sampling_period)
            
    return exc_time_domain, res_time_domain, time, sampling_period

def readxlsx(  file_location
             , file_name
             , file_convert = False
             , save_location = None
             , save_name = None):
    file_destination = file_location + file_name + ".xlsx"
    data = pd.read_excel(file_destination, header=None, skiprows = 13)
    
    fig1, (force, veloc) = plt.subplots(2)
    force.set_xlim([0.24, 0.50])
    force.legend()
    force.grid(True)
    veloc.set_xlim([0.24, 0.50])
    veloc.legend()
    veloc.grid(True)
    
    "Sampling Rate"
    sampling_period = data.iloc[1, 0]
    col_num_list = np.arange(1,len(data.columns),2)
    for i, col_num in enumerate(col_num_list):
        
        "Data Cut"
        v_time = data[col_num + 1].dropna()
        f_time = data[col_num].dropna()
        v_time = np.array(v_time[np.where( f_time >= 0.1 )[0][0] - 1000:])
        f_time = np.array(f_time[np.where( f_time >= 0.1 )[0][0] - 1000:])
    
        "Time Range"
        time = np.arange(0, sampling_period * v_time.shape[0], sampling_period)
        
        "Graph Plotting"
        force.plot(time, f_time, label = str(i+1))
        veloc.plot(time, v_time, label = str(i+1))
        
def readnpz(file_destination):
    data = np.load(file_destination)
    f = data['force_time']
    v = data['velo_time']
    t = data['time']
    sampling_period = t[1]
    return f, v, t, sampling_period
        
def savesignal(  file_location
               , file_name
               , save_location = None
               , save_name = None):
    file_destination = file_location + file_name + ".xlsx"
    data = pd.read_excel(file_destination, header=None, skiprows = 13)
    if save_location == None:
        save_location = file_location
    if save_name == None:
        save_name == file_name
    "Sampling Rate"
    sampling_period = data.iloc[1, 0]
    col_num_list = np.arange(1,len(data.columns),2)
    for i, col_num in enumerate(col_num_list):
        
        "Data Cut"
        v_time = data[col_num + 1].dropna()
        f_time = data[col_num].dropna()
        v_time = np.array(v_time[np.where( f_time >= 0.1 )[0][0] - 1000:])
        f_time = np.array(f_time[np.where( f_time >= 0.1 )[0][0] - 1000:])
    
        "Time Range"
        time = np.arange(0, sampling_period * v_time.shape[0], sampling_period)
        
        
        "Save data"
        outfile = save_location + save_name + str(i+1) + ".npz"
        np.savez(outfile, force_time=f_time, velo_time=v_time, time = time)
        
def plot_paired_signals(f, v, t, plot_width=20, plot_height=20, x_lim = [0.0,1.0], y_lim = None):
    fig, ax = plt.subplots(2, figsize=(plot_width, plot_height))

    ax[0].set_title("Hammer Signal")
    ax[0].set_ylabel("Force (N)")
    ax[0].set_xlabel("Time (s)")
    ax[0].grid(True)

    ax[1].set_title("Laser Signal")
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True)

    if x_lim == None:
        ax[0].set_xlim(auto = True)
        ax[1].set_xlim(auto = True)
    else:
        ax[0].set_xlim(x_lim)
        ax[1].set_xlim(x_lim)
    
    if y_lim == None:
        ax[0].set_ylim(auto = True)
        ax[1].set_ylim(auto = True)
    else:
        ax[0].set_ylim(y_lim)
        ax[1].set_ylim(y_lim)
        
    ax[1].plot(t,v)
    ax[0].plot(t,f)

def plot_spatial_signal(file_directory, x_position, y_position, plot_width=20, plot_height=20, x_lim = [0.0,0.2], y_lim = None):
    filename = glob.glob(file_directory + "*x"+x_position+"_y"+y_position+"*.csv")
    f, v, t, sampling_period = readcsv(filename[0])
    plot_paired_signal(f, v, t, plot_width, plot_height, x_lim, y_lim)
    
def plot_spectra(F, V, H, freq, plot_width=20, plot_height=30, x_lim = [0,1000], y_lim = None, background = None):
    fig, ax = plt.subplots(4, figsize=(plot_width, plot_height))
    ax[0].set_title("Force Signal Autospectrum")
    ax[0].set_ylabel("Power (dB)")
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_xlim(x_lim)
    ax[0].grid(True)

    ax[1].set_title("Laser Signal Autospectrum")
    ax[1].set_ylabel("Power (dB)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim(x_lim)
    ax[1].grid(True)

    ax[2].set_title("FRF (Magnitude)")
    ax[2].set_ylabel("V/F (dB)")
    ax[2].set_xlabel("Frequency (Hz)")
    ax[2].set_xlim(x_lim)
    ax[2].grid(True)
    
    ax[3].set_title("FRF (Phase)")
    ax[3].set_ylabel("Angle (rad)")
    ax[3].set_xlabel("Frequency (Hz)")
    ax[3].set_xlim(x_lim)
    ax[3].grid(True)

    if x_lim == None:
        ax[0].set_xlim(auto = True)
        ax[1].set_xlim(auto = True)
        ax[2].set_xlim(auto = True)
        ax[3].set_xlim(auto = True)
    else:
        ax[0].set_xlim(x_lim)
        ax[1].set_xlim(x_lim)
        ax[2].set_xlim(x_lim)
        ax[3].set_xlim(x_lim)
    
    if y_lim == None:
        ax[0].set_ylim(auto = True)
        ax[1].set_ylim(auto = True)
        ax[2].set_ylim(auto = True)
        ax[3].set_ylim(auto = True)
    else:
        ax[0].set_ylim(y_lim)
        ax[1].set_ylim(y_lim)
        ax[2].set_ylim(y_lim)
        ax[3].set_ylim(y_lim)
    
    ax[0].plot(freq, 20*np.log10(abs(F)))
    ax[1].plot(freq, 20*np.log10(abs(V)))
    ax[2].plot(freq, 20*np.log10(abs(V/F)))
    ax[3].plot(freq, np.angle(V/F))
    
    if background != None:
        ax[0].plot(freq, 20*np.log10(abs(background[0])))
        ax[1].plot(freq, 20*np.log10(abs(background[1])))
    
def plot_multiple(path, minimum_force = 0.0 ):
    filenames = glob.glob(path + ".csv")

    fig, ax = plt.subplots(5, figsize=(20, 40))
    #Time Domain
    x_lim_left = 0.0
    x_lim_right = 0.2

    ax[0].set_title("Hammer Signal")
    ax[0].set_ylabel("Force (N)")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_xlim([x_lim_left,x_lim_right])
    ax[0].grid(True)

    ax[1].set_title("Laser Signal")
    ax[1].set_ylabel("Velocity (m/s)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_xlim([x_lim_left,x_lim_right])
    ax[1].grid(True)

    #Frequency Domain
    x_lim_left = 0.0
    x_lim_right = 1000.0

    ax[2].set_title("Force Signal Autospectrum")
    ax[2].set_ylabel("Power (dB)")
    ax[2].set_xlabel("Frequency (Hz)")
    ax[2].set_xlim([x_lim_left,x_lim_right])
    ax[2].grid(True)

    ax[3].set_title("Laser Signal Autospectrum")
    ax[3].set_ylabel("Power (dB)")
    ax[3].set_xlabel("Frequency (Hz)")
    ax[3].set_xlim([x_lim_left,x_lim_right])
    ax[3].grid(True)

    ax[4].set_title("FRF (Amplitude)")
    ax[4].set_ylabel("F/V (dB)")
    ax[4].set_xlabel("Frequency (Hz)")
    ax[4].set_xlim([x_lim_left,x_lim_right])
    ax[4].grid(True)

    Sum = 0.0

    for file in filenames:
        f, v, t, sampling_period = msp.readcsv(file)
        if max(f) > minimum_force:
            ax[0].plot(t,f, label = file[-13:-6])
            ax[1].plot(t,v)

            F = np.fft.rfft(f,2**15)
            V = np.fft.rfft(v,2**15)
            freq = np.fft.rfftfreq(2**15, d=sampling_period)

            ax[2].plot(freq, 20*np.log10(abs(F)), label = file[-13:-6])
            ax[3].plot(freq, 20*np.log10(abs(V)), label = file[-13:-6])
            ax[4].plot(freq, 20*np.log10(abs(V/F)), label = file[-13:-6])

            Sum += abs(V/F)**2

    Sum = Sum/len(filenames)

    ax[0].legend()
    
    fig= plt.figure(figsize=(20,10))
    plt.plot(freq, 10*np.log10(Sum))
    #plt.ylim(0.1,20)
    plt.xlim(0,1000)
    plt.title("Space Average FRF (Magnitude)")
    plt.ylabel("V/F (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True)
    
def plot_results(path,
                 time_plot_range = [0.0,4.0],
                 window = None, 
                 v_window = None,
                 win_begin = 0,
                 win_end = None,
                 segment = [],
                 number_of_segments = 0,
                 BGff = [],
                 BGvv = [], 
                 label = False, 
                 force_range = [0.1,np.inf],
                 multiple_hits = False, 
                 plot = True,
                 method = 'DFT', 
                 trimming_index = [],     # array if starting trimming-time
                 average_num = 0,    #number of averages, preset is 0 meaning 'use all available data'.
                 mic = False,
                 reference_channel = 1,
                 response_channel = 2,
                 len_smooth = 15,
                 len_sig = 4,
                 tau = 1.5,
                 trigger = True, 
                 trigger_level = 0.1,
                 delay_time = None,
                 loop = False,
                 overlap_ratio = 2/3,
                 FRF_yrange = [],
                 freq_plot_range = [0.0,1000.0],
                 debug = False):
    
    """
    Read data files and calculate DFT or Segmant DFT, which are then plotted.
    Return H1, H2, frequency range, coherance, average F-F cross-spectrum, average V-V cross-spectrum, and number of averages.
    
    path = file name
    BGff, BGvv = background noise for input and output
    label = boolean, show label for file name identification
    force_range = array of 2 number [a,b], filter for maximum force between a to b Newton
    method: 'DFT' : whole signal length.
            'segmentDFT' average by segments.
    """
    if debug:
        whole_time = time.time()
    #declare sample counter
    counter = 0
    
    #Major and minor lines
    majorTicks = np.arange(0,1001,100)
    minorTicks = np.arange(0,1001,10)
    
    #Targeting files
    file_name = glob.glob(path)
    if len(file_name) == 0: # check if the input name is valid.
        return print("No file found.")
    
    if debug:
        print('Number of files: ' + str(len(file_name)))
        for name in file_name:
            print(name)
    

    #Individial Hits Comparison 5-plot set
    """
    1. Time Domain Force
    2. Time Domain Response (Laser or Mic)
    3. Freq Domain Force
    4. Freq Domain Response (Laser or Mic)
    5. Freq Domain V/F
    """
    
    if plot == True:
        fig, ax = plt.subplots(5, figsize=(20, 5*8))

        #Time Domain Plot Setting
        x_lim = time_plot_range

        if window == "Hanning" or method == "segmentDFT":
            x_lim = time_plot_range
        
        ax[0].set_title("Input Signal")
        ax[0].set_ylabel("Force (N)")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_xlim(x_lim)
        ax[0].grid(True)

        ax[1].set_title("Response Signal")
        ax[1].set_ylabel("Velocity (m/s)")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_xlim(x_lim)
        ax[1].grid(True)
        
        if mic == True:
            ax[1].set_title("Mic Signal")
            ax[1].set_ylabel("Sound Pressure (Pa)")
        
        #Frequency Domain Plot Setting
        x_lim = freq_plot_range
        
        ax[2].set_title("Input Signal Autospectrum")
        ax[2].set_ylabel("Power (dB)")
        ax[2].set_xlabel("Frequency (Hz)")
        ax[2].set_xlim(x_lim)
        ax[2].grid(True)

        ax[3].set_title("Response Signal Autospectrum")
        if mic == True:
            ax[3].set_title("Mic Signal Autospectrum")
        ax[3].set_ylabel("Power (dB)")
        ax[3].set_xlabel("Frequency (Hz)")
        ax[3].set_xlim(x_lim)
        ax[3].grid(True)

        ax[4].set_title("FRF (Magnitude)")
        ax[4].set_ylabel("V/F (dB)")
        ax[4].set_xlabel("Frequency (Hz)")
        ax[4].set_xlim(x_lim)
        ax[4].grid(True)
        
        for i in range(2,5):
            ax[i].set_xticks(majorTicks)
            ax[i].set_xticks(minorTicks, minor=True)
            ax[i].grid(which = 'both')
            ax[i].grid(which = 'minor', alpha = 0.3)
            ax[i].grid(which = 'major', alpha = 1)
            ax[i].set_xlim(x_lim)

    #Declare Variables for Average Auto- and Cross-spectra
    Gvv = 0.0
    Gff = 0.0
    Gfv = 0.0+0j
    Gvf = 0.0+0j
        
    #Signal Processing Part
    
    # reading each targeted files, all targeted files or until the number of average is hit.
    for file_destination in file_name:

        # check number of sample.
        if average_num == 0 or average_num != 0 and counter!= average_num:

            # If set delay time according to window applied
            if window == "Hanning" and delay_time == None:
                delay_time = 0.015
            elif window == "Step" and delay_time == None:
                delay_time = 0.0015
            elif window == None and delay_time == None:
                delay_time = 0.01
            elif method == 'DFT' and delay_time == None:
                delay_time = 0.01
            
            # read-the-file function, with trigger and time delay
            f, v, t, sampling_period = msp.readcsv(file_destination,
                                                   trigger = trigger,
                                                   trigger_level = trigger_level,
                                                   delay_time = delay_time,
                                                   reference_channel = reference_channel,
                                                   response_channel = response_channel)
            
            # assign f-signal maximum and minimum locations
            Max = np.where(f == max(f))[0][0]
            Min = np.where(f == min(f))[0][0]

            # Force range filter, only let f-signal with maximum within force_range through.
            if max(f) > force_range[0] and max(f) < force_range[1]:
                # Multiple hit filter
                if multiple_hits == None or multiple_hits == False and msp.hit_counts(f) == 1 or multiple_hits == True and msp.hit_counts(f) != 1:
                    # Method filter: DFT(default) or segmentDFT
                    if method == "DFT":
                        # Apply windows
                        if window == "Hanning" and len(f) > int(1/sampling_period) or window != "Hanning":
                            Window = np.zeros_like(f) # declare an array for the window
                            
                            if window == "Step":                             
                                smooth = np.linspace(-3, 3, len(Step[Min-len_smooth:Min+len_smooth])) # Declare smoothing array of real number from -3 to 3.
                                Window[Min-len_smooth:Min+len_smooth] = (sp.special.erf(-smooth)+1)/2 # Use above array to create smoothing tail with error function.
                                Window[0:Min-len_smooth] = 1 # Create final signal window. 
                                f = f * Window

                            elif window == "Hanning":
                                Window[2*Max-Min-len_smooth:Min+len_smooth] = sps.windows.hann(2*(Min-Max)+len_smooth*2)
                                f = f * Window

                            if v_window == "Exponential":
                                Window += np.exp(-tau*t) # Create exponential window
                                f = f * Window
                                v = v * Window

                            F = np.fft.rfft(f, int(len_sig /sampling_period))
                            V = np.fft.rfft(v, int(len_sig /sampling_period))
                            freq = np.fft.rfftfreq( int(len_sig /sampling_period), d=sampling_period)

                            Gvv += np.conj(V)*V
                            Gff += np.conj(F)*F
                            Gfv += np.conj(F)*V

                            if plot == True:
                                ax[0].plot(t, f, label = file_destination[-6:-4])
                                ax[1].plot(t, v)
                                ax[2].plot(freq, 20*np.log10(abs(F)))
                                ax[3].plot(freq, 20*np.log10(abs(V)))
                                ax[4].plot(freq, 20*np.log10(abs(V/F)))
                                      
                            counter += 1 # raise sample counter

                    elif method == "segmentDFT":
                        if debug:
                            indi_time = time.time()
                            
                        if loop == True and trimming_index == []:
                            trimming_index = trigger_index(f, trigger_level, len_sig, delay = delay_time, sampling_period = sampling_period)
                        else:
                            trimming_index = overlap_index(f, overlap_ratio, len_sig, sampling_period = sampling_period)
                                
                        if number_of_segments == 0:
                            number_of_segments = len(trimming_index)
                            
                         # frequency axis for plot. 

                        for n, index in enumerate(trimming_index):
                            if n == number_of_segments:
                                break
                            if average_num == 0 or average_num != 0 and counter != average_num:
                                input_segment , t = trim(f, index, index+int(len_sig/sampling_period), sampling_period, extension = True)
                                output_segment, t = trim(v, index, index+int(len_sig/sampling_period), sampling_period, extension = True)
                                
                                if segment != []:
                                    segment_N = segment/sampling_period
                                    segment_N = segment_N.astype(int)
                                    input_segment  = input_segment[ segment_N[0]:segment_N[1]]
                                    output_segment = output_segment[segment_N[0]:segment_N[1]]
                                
                                if window == "Hanning":
                                    Window = np.zeros_like(input_segment)
                                    win_begin = int(win_begin/sampling_period)
                                    if win_end == None:
                                        Window[win_begin:] = sps.windows.hann(len(Window[win_begin:]))
                                    else:
                                        Window[win_begin:int(win_end/sampling_period)] = sps.windows.hann(len(Window[win_begin:int(win_end/sampling_period)]))
                                    input_segment  = input_segment  * Window
                                    output_segment = output_segment * Window
                                
                                freq = np.fft.rfftfreq(len(input_segment), d=sampling_period)
                                F = np.fft.rfft(input_segment , len(input_segment))
                                V = np.fft.rfft(output_segment, len(input_segment))

                                if plot == True:
                                    ax[0].plot(t[0:len(input_segment )], input_segment, label = file_destination[-6:-4]+'-'+str(n+1))
                                    ax[1].plot(t[0:len(output_segment)], output_segment)
                                    ax[2].plot(freq, 20*np.log10(abs(F)))
                                    ax[3].plot(freq, 20*np.log10(abs(V)))
                                    ax[4].plot(freq, 20*np.log10(abs(V/F)))

                                Gvv += np.conj(V)*V
                                Gff += np.conj(F)*F
                                Gfv += np.conj(F)*V
                                
                                counter += 1 # raise sample counter
                                
                                if debug:
                                    print('Process ' + str(counter) + ' is done and took ' + str(time.time()-indi_time) + ' sec.')

                    else: return print("Choose: 'DFT' or 'segmentDFT' ")

                else: print(file_destination+' did not pass multiple hits filter.' )

            else: print(file_destination+' did not pass force range filter.')
        
    if counter == 0:
        return print('No sample passed into the Fourier analysis process.')
                                       
    if plot == True:
        if len(BGff) > 0 :
            ax[2].plot(freq, 20*np.log10(BGff))
        if len(BGvv) > 0 :
            ax[3].plot(freq, 20*np.log10(BGvv))

        if label == True:
            ax[0].legend()

    'Output Average'
    H1 = Gfv/Gff
    H2 = Gvv/np.conj(Gfv)

    Gff = Gff/counter
    Gfv = Gfv/counter
    Gvv = Gvv/counter
    
    'Coherance'
    gamma2 = abs(Gfv)**2 / (Gvv*Gff)
    
    #Freq Domain Averages 5-plot
    """
    1.Force average
    2.Response average
    3.Frequency Response Function
    4.Phase
    5.Coherance
    """
    if plot == True:
        "H1, H2, Phase, and Coherance"
        fig, ax = plt.subplots(5, figsize =(20,30)) 

        for i in range(5):
            ax[i].set_xlim(freq_plot_range)
        
        ax[0].set_ylabel("Gff")   
        ax[1].set_ylabel("Gvv")
        ax[2].set_ylabel("FRF")
        ax[3].set_ylim([-np.pi,np.pi])
        ax[3].set_ylabel("Phase")
        ax[4].set_ylabel("Coherance")

        ax[0].plot(freq, 10*np.log10(abs(Gff)))
        ax[1].plot(freq, 10*np.log10(abs(Gvv)))
        ax[2].plot(freq, 10*np.log10(abs(H1)),label = "H1")
        ax[2].plot(freq, 10*np.log10(abs(H2)),label = "H2")
        if FRF_yrange != []:
            ax[2].set_ylim(FRF_yrange)
        ax[3].plot(freq, np.angle(H1))
        #for i in range(-int(np.floor(min(np.unwrap(np.angle(H1[0:np.where(freq == 1001)[0][0]])))/np.pi/2))+2):
        #    ax[3].plot(freq, np.unwrap(np.angle(H2)) + i*2*np.pi, color = 'blue')
        #for i in range(int(np.ceil(max(np.unwrap(np.angle(H1[0:np.where(freq == 1001)[0][0]])))/np.pi/2))+1):
        #    ax[3].plot(freq, np.unwrap(np.angle(H2)) - i*2*np.pi, color = 'blue')
        ax[4].plot(freq,np.real(gamma2))
        
        majorTicks = np.arange(0,1001,100)
        minorTicks = np.arange(0,1001,10)

        for i in range(5):
            ax[i].set_xticks(majorTicks)
            ax[i].set_xticks(minorTicks, minor=True)
            ax[i].grid(which = 'both')
            ax[i].grid(which = 'minor', alpha = 0.2)
            ax[i].grid(which = 'major', alpha = 0.7)
        
        ax[2].legend()
        
    if debug:
        print('The process took' + str(time.time()-whole_time) + ' sec.')
    
    return H1, H2, gamma2, freq, Gff, Gvv, counter

def trigger_index(signal, trigger_level, interval_length, delay = 0, sampling_period = 1, sign = False):
    """
    Create array of indice for signal trimming by using trigger.

    Parameters
    ----------
    signal : 1-d array
        Input signal to be analyzed.
    trigger_level : float
        Signal magnitude to set off the trigger.
    interval_length : float
        Time after the trigger is set off before the next trigger. Arming period.
    delay : floar, optional
        Pre-trigger time. The default is 0.
    sampling_period : float, optional
        Signal's sampling period. The default is 1.
    sign : boolean, optional
        Set whether to account for the +/- sign of the trigger level or not. The default is False.

    Returns
    -------
    TYPE int, array
        array of indice for the starting point of each triggers.

    """
    
    #initial value
    trigger_index = []
    front = 0

    #start loop here
    while np.where(abs(signal[front:]) > abs(trigger_level))[0].size > 0:

        if sign:
            if trigger_level >= 0:
                index = np.where(signal[front:] > trigger_level)[0][0]
            else:
                index = np.where(signal[front:] < trigger_level)[0][0]

        else:
            index = np.where(abs(signal[front:]) > trigger_level)[0][0]

        trigger_index.append(int(index+front))

        front = int(trigger_index[-1] + interval_length/sampling_period)
        
    return trigger_index - int(delay/sampling_period)

def overlap_index(signal, overlap_ratio, interval_length, sampling_period = 1):
    #in case of 1 overlap_ratio input
    if overlap_ratio == 1:
        return print('overlap_ration must be less than 1.')
    #initial value
    index = [0]
    
    while index[-1]+interval_length/sampling_period < len(signal):
        front = index[-1]
        index.append(int(front+(1-overlap_ratio)*interval_length/sampling_period))

    return index
    
def trim(signal, start, end, sampling_period, extension = False):
    """
    Trim the input signal with start-time and end-time.
    
    Parameters
    ----------
    signal : 1D-array
        Input signal to be trimmed.
    start : float
        Index for trimming starting point.
    end : float
        Index for trimming ending point. If =0/False, trim-end = signal end
    sampling_freq : float
        sampling freq of the signal.
    extension : boolean
        Involve when trim-end > signal end. If true, add 0s tail. If false, stop at signal end.

    Returns
    -------
    1D-array of trimmed signal.
    New time array.
    """
    
    #turn start and end into index.
    start = int(start)
    end   = int(end)
    signal_end = len(signal)-1
    
    if end > signal_end: # In case of trimming end is longer than signal end
        if extension == True: # in case extension id true add zeros to the tail.
            trim_signal  = np.zeros(end-start)
            trim_signal[:signal_end-start] += signal[start:signal_end]
            print("Signal is shorter than desired and zeros are added.")
        else: # in case extension is false 
            trim_signal = signal[start:signal_end]
            print("Signal is shorter than desired (no zeros are added).")
    elif end == 0:
        trim_signal = signal[start:signal_end]
    else:
        trim_signal = signal[start:end]
    
    time_array = np.arange(len(trim_signal))*sampling_period
        
    return trim_signal, time_array

def CreatSeg(signal, n, sampling_period, interval_len, segment = [], overlap_ratio = 2/3, window = "Hanning", win_begin = 0, win_end = None, trimming_index = []):
    """
    Parameters
    ----------
    signal : TYPE
        time-domain signal to be trimmed.
    n : TYPE
        the n-th segment to be return in the output of the function.
    sampling_period : TYPE
        DESCRIPTION.
    interval_len : TYPE
        interval length in second (imediately converted to N).
    segment : float, [start, end], time in sec, optional
        Subsegment interval begin and end point in second (imediately converted to N).
        The default is [].
    overlap_ratio : TYPE, optional
        segment overlapping ratio. 
        The default is 2/3.
    window : TYPE, optional
        window for the segment.
        The default is "Hanning".
    win_begin : TYPE, optional
        DESCRIPTION.
        The default is 0.
    win_end : TYPE, optional
        DESCRIPTION.
        The default is None.
    trimming_index : TYPE, optional
        Array of indice indicating when each input signal starts, for signal trimming.
        If not assigned, the signal is trimmed with interval_len.
        The default is [].

    Returns
    -------
    signal_segment : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    """
    
    
    interval_N = int(interval_len/sampling_period) # convert interval length in second into index number.
    
    signal_segment = np.zeros(interval_N) # declare array for segmented signal

    win_begin = int(win_begin/sampling_period) # convert window begin time to index number.
    
    if win_end == None:
        win_end = len(signal_segment) # If window end is not assigned, assign it with segment length.
    else:
        win_end = int(win_end/sampling_period) # If it is assigned, convert it to sample unit.
    
    # Check if the n th segment length is not shorter than the desired segment length.
    if n*int(interval_N*(1-overlap_ratio))+interval_N < len(signal): 
        signal_segment += signal[n*int(interval_N*(1-overlap_ratio)):n*int(interval_N*(1-overlap_ratio))+interval_N]
    else:
        signal_segment[0:len(signal[n*int(interval_N*(1-overlap_ratio)):])] += signal[n*int(interval_N*(1-overlap_ratio)):]

    if window != None:
        Window = np.zeros(interval_N)
        if window == "Hanning":
            Window[win_begin:win_end] = sps.windows.hann(win_end-win_begin)
    else:
        Window = 1
        
    signal_segment = signal_segment*Window
    t = np.arange(0,interval_len,sampling_period)
    
    if segment == []:
        return signal_segment, t
    else:
        segment = np.array(segment)/sampling_period
        signal_segment = signal_segment[int(segment[0]):int(segment[1])]
        t = t[int(segment[0]):int(segment[1])]
        return signal_segment, t

def SegmentDFT(  InputSignal
               , ResponseSignal
               , sampling_period
               , freq_resolution = 1
               , segment_len = None
               , overlap_ratio = 2/3
               , window = "hanning"
               , plot_individual_results = False):
    """
    Input/ResponseSignal : time-domain input/response signals.
    sampling_period      : sampling period of the input signal.
    freq_resolution      : Fourier transform frequency resolution.
    segment_len          : segment length in [sec] (immediately converted to number of segment 'N').
                           ***note: freq_resolution and segment_len are corelating arguments, one determines the other.
                           By default, freq_resolution is pre-set. If segment_len is assigned in the argument, 
                           the freq_resolution will be overwritten, even when the latter is also assigned.
    overlap_ratio        : segment overlapping ratio.
    window               : window for the segment.
    plot_individual_results: plot each segment F and V results.
    """
    if segment_len == None:
        segment_len = 1/freq_resolution
    
    segment_N = int(segment_len/sampling_period)
    number_of_segments = int(np.ceil((len(InputSignal)-segment_N)/segment_N/(1-overlap_ratio)))
    freq = np.fft.rfftfreq(segment_N,d=sampling_period)
    Gff = 0
    Gvv = 0
    Gfv = 0
    H1 = 0
    H2 = 0
    
    for n in range(number_of_segments+1):
        input_segment = CreatSeg(InputSignal, n, sampling_period, segment_len, overlap_ratio = overlap_ratio, window = window)
        F = np.fft.rfft(input_segment)

        output_segment = CreatSeg(ResponseSignal, n, sampling_period, segment_len, overlap_ratio = overlap_ratio, window = window)
        V = np.fft.rfft(output_segment)

        Gvv += np.conj(V)*V
        Gff += np.conj(F)*F
        Gfv += np.conj(F)*V
        
    H1 = Gfv/Gff
    H2 = Gvv/np.conj(Gfv)
    Gff = Gff/number_of_segments
    Gvv = Gvv/number_of_segments
    Gfv = Gfv/number_of_segments
    gamma2 = abs(Gfv)**2 / (Gvv*Gff)

    return H1, H2, gamma2, freq, Gff, Gvv, Gfv, number_of_segments

def plot_FRF(results,
             labels,
             legend = True,
             plot_H = 0,
             plot_range = [None,None],
             divider = [1],
             figsize = [25,16],
             plot_linewidth = 1.5,
             offset = 0,
             y_lim = None,
             x_lim = [0,1000],
             x_minorticks = 10,
             x_majorticks = 100,
             title = None,
             titlesize = 24,
             linefontsize = 10,
             ColorList = ['m','b', 'c', 'g', 'gold','orangered', 'r','maroon'],
             res_freq = [],
             mode_num = [],
             save_resolution = None,
             save_filename = [],
             save_extension = '.png'
             ):
    """
    Plot FRF (magnitudes and phase).
    resutls: the direct output from 'plot_results' function.
    labels: array of graph names.
    plot_H: 0 or 1, 0 = H1, 1 = H2.
    figsize: size of the whole plot.
    offset: offset for each graph in dB.
    y_lim: y-axis magnitude range in dB.
    x_lim: x-axis magnitude and phase range in Hz.
    title: main title.
    titlesize: main title font size.
    linefontsize: font size for mode identifier.
    res_freq: array of natural frequencies (optional).
    mode_num: array of corresponding mode number.
    save_resolution: saving resolution for image.
    save_filename: saving file name.
    save_extension: saving extension.
    """
    linewidth = 2
    linestyles = (0,(5,5))
    linecolor_spec = 'r'
    linecolor_reg = 'orange'
    majorTicks = np.arange(0,x_lim[1]+1,x_majorticks)
    minorTicks = np.arange(0,x_lim[1]+1,x_minorticks)
    
    #Set Plot Space
    if plot_H == 0 or plot_H == 1:
        fig, ax =plt.subplots(2, figsize = figsize, gridspec_kw={'height_ratios': [2, 1]})
        ax[1].tick_params(which = 'both', bottom = False, top = True, labelbottom = False)
    else:
        fig, ax =plt.subplots(1, figsize = [figsize[0],figsize[1]/2])
    
    #Title
    if title != None:
        fig.suptitle(title, fontsize= titlesize)

    #Plot
    for n in range(len(results)):
        result = results[n]
        H = result[plot_H][plot_range[0]:plot_range[1]]/divider
        freq = result[3][plot_range[0]:plot_range[1]]
        if plot_H == 0 or plot_H == 1:
            ax[0].set_ylabel("V/F (dB)")
            ax[1].set_ylabel("Phase (radian)")
            ax[0].plot(freq,10*np.log10(abs(H))+n*offset, color = ColorList[n], label = labels[n], linewidth = plot_linewidth)
            for i in range(-int(np.floor(min(np.unwrap(np.angle(H[0:np.where(freq > x_lim[1])[0][0]])))/np.pi/2))+1):
                ax[1].plot(freq, np.unwrap(np.angle(H)) + i*2*np.pi, color = ColorList[n], linewidth = plot_linewidth)
            for i in range(int(np.ceil(max(np.unwrap(np.angle(H[0:np.where(freq > x_lim[1])[0][0]])))/np.pi/2))+1):
                ax[1].plot(freq, np.unwrap(np.angle(H)) - i*2*np.pi, color = ColorList[n], linewidth = plot_linewidth)
        else:
            ax.plot(freq,10*np.log10(H)+n*offset, color = ColorList[n], label = labels[n])

    #Axis Labels
    if plot_H == 0 or plot_H == 1:
        if y_lim != None:
            ax[0].set_ylim(y_lim)
        if legend == True:
            ax[0].legend()
        ax[1].set_ylim(-np.pi,np.pi)
        ax[1].set_yticks(np.arange(-np.pi,np.pi+1,np.pi/2))
        p = unicodedata.lookup("GREEK SMALL LETTER PI")
        ax[1].set_yticklabels(['-'+p,'-'+p+'/2',0,p+'/2',p])
        fig.subplots_adjust(hspace=0.1)
        plt.setp([ax[0].get_xticklabels()], fontsize = 10)
        for i in range(2):
            ax[i].set_xticks(majorTicks)
            ax[i].set_xticks(minorTicks, minor=True)
            ax[i].grid(which = 'both')
            ax[i].grid(which = 'minor', alpha = 0.3)
            ax[i].grid(which = 'major', alpha = 1)
            ax[i].set_xlim(x_lim)
            y_lim = ax[i].get_ylim()
            if len(res_freq) != 0:
                for n in range(len(res_freq)):
                    if mode_num[n] == 'H':
                        ax[i].axvline(res_freq[n], color = 'g', linestyle = linestyles, linewidth = linewidth)
                    elif mode_num[n][0] != '0':
                        ax[i].axvline(res_freq[n], color = linecolor_reg, linestyle = linestyles, linewidth = linewidth)
                    else:
                        ax[i].axvline(res_freq[n], color = linecolor_spec, linestyle = linestyles, linewidth = linewidth)
                    ax[i].set_ylim(y_lim)
                if len(mode_num) != 0 and i == 0:
                    for n in range(len(mode_num)):
                        if mode_num[n] == '21,':
                            ax[i].text(res_freq[n]-19, y_lim[1] + 0.01 * (y_lim[1]-y_lim[0]), mode_num[n], fontsize = linefontsize)
                        else:
                            ax[i].text(res_freq[n], y_lim[1] + 0.01 * (y_lim[1]-y_lim[0]), mode_num[n],  fontsize = linefontsize)

    else:
        if y_lim != None:
            ax.set_ylim(y_lim)
        if legend == True:
            ax.legend()
        ax.set_xticks(majorTicks)
        ax.set_xticks(minorTicks, minor=True)
        ax.grid(which = 'both')
        ax.grid(which = 'minor', alpha = 0.3)
        ax.grid(which = 'major', alpha = 1)
        ax.set_xlim(x_lim)
        y_lim = ax.get_ylim()
        if len(res_freq) != 0:
            for n in range(len(res_freq)):
                if mode_num[n][0] != '0':
                    ax.axvline(res_freq[n], color = linecolor_reg, linestyle = linestyles, linewidth = linewidth)
                else:
                    ax.axvline(res_freq[n], color = linecolor_spec, linestyle = linestyles, linewidth = linewidth)
                ax.set_ylim(y_lim)
            if len(mode_num) != 0:
                for n in range(len(mode_num)):
                    ax.text(res_freq[n], y_lim[1] + 0.01 * (y_lim[1]-y_lim[0]), mode_num[n], fontsize = linefontsize)
    
    if len(save_filename) != 0:
        plt.savefig(save_filename+save_extension, bbox_inches='tight', dpi = save_resolution)
            
def hit_counts(time_signal):
    n=1
    for i in range(len(np.where(time_signal > 0.1)[0])-1):
        if np.where(time_signal > 0.1)[0][i+1] - np.where(time_signal > 0.1)[0][i] > 1:
            n += 1
    return n

def ApproxFRF(path, delay_time = None, fwin = None, vwin = None, len_smooth = 15, len_sig = 4, tau = 1.5, ftlim = [0,1], vtlim = [0,0]):
    """
    fwin = excitation time window: "Step", "Hanning".
    vwin = response time window: "Exponential".
    if vwin is declared, fwin is also modified.
    len_sig = signal length, in second, that will be calculated.
    tau = exponential decay rate, assign as a possitive number (the code already has a negative sign).
    """
    files = glob.glob(path)

    fig, ax = plt.subplots(4,2, figsize = [32,5*4])
    ax[0,0].set_xlim(ftlim[0],ftlim[1])
    ax[1,0].set_xlim(ftlim[0],ftlim[1])
    if vtlim == [0,0]:
        ax[0,1].set_xlim(0,len_sig)
        ax[1,1].set_xlim(0,len_sig)
    else:
        ax[0,1].set_xlim(vtlim[0],vtlim[1])
        ax[1,1].set_xlim(vtlim[0],vtlim[1])
    ax[3,1].set_ylim(-np.pi,np.pi)
    ax[3,0].set_ylim(-np.pi,np.pi)


    Gff = 0
    Gvv = 0
    Gfv = 0
    Gvf = 0
    counter = 0
    
    if delay_time == None:
        if fwin == "Step":
            delay_time = 0.0015
        elif fwin == "Hanning":
            delay_time = 0.015
    
    for filename in files:
        f, v, t, sampling_period = msp.readcsv(filename, delay_time = delay_time)

        Max = np.where(f == max(f))[0][0]
        Min = np.where(f == min(f))[0][0]

        if fwin == "Step":
            Step = np.zeros_like(f)
            smooth = np.linspace(-3, 3, len(Step[Min-len_smooth:Min+len_smooth])) #Set smooth array
            Step[Min-len_smooth:Min+len_smooth] = (sp.special.erf(-smooth)+1)/2 #Tail smoothing
            Step[0:Min-len_smooth] = 1 #Signal window
            f = f*Step
        
        elif fwin == "Hanning":
            Hann = np.zeros_like(f)
            Hann[2*Max-Min-len_smooth:Min+len_smooth] = sps.windows.hann(2*(Min-Max)+len_smooth*2)
            f = f*Hann
        
        if vwin == "Exponential":
            Expo = np.zeros_like(signal[2])+np.exp(-tau*(signal[2])) #Expo window
            f = f*Expo #Final window
            v = v*Expo #Final window

        F = np.fft.rfft(f, int(len_sig*1/sampling_period))
        V = np.fft.rfft(v, int(len_sig*1/sampling_period))
        freq = np.fft.rfftfreq( int(len_sig*1/sampling_period), d=sampling_period)

        Gvv += np.conj(V)*V
        Gff += np.conj(F)*F
        Gfv += np.conj(F)*V

        counter += 1

        ax[0,0].plot(t,signal[0])
        ax[1,0].plot(t,f)
        ax[0,1].plot(t,signal[1])
        ax[1,1].plot(t,v)
        ax[2,0].plot(freq, 20*np.log10(abs(F)))
        ax[3,0].plot(freq, np.angle(F))
        ax[2,1].plot(freq, 20*np.log10(abs(V)))
        ax[3,1].plot(freq, np.angle(V))


    for m in range(2,4):
        for n in range(2):
            ax[m,n].set_xlim(0,1000)

    fig, ax = plt.subplots(4, figsize=(16,5*4))

    majorTicks = np.arange(0,1001,100)
    minorTicks = np.arange(0,1001,10)

    for i in range(4):
        ax[i].set_xticks(majorTicks)
        ax[i].set_xticks(minorTicks, minor=True)
        ax[i].grid(which = 'both')
        ax[i].grid(which = 'minor', alpha = 0.2)
        ax[i].grid(which = 'major', alpha = 0.7)
        ax[i].set_xlim(0,1000)

    H1 = Gfv/Gff
    H2 = Gvv/np.conj(Gfv)
    Gff = Gff/counter
    Gvv = Gvv/counter
    gamma2 = abs(Gfv)**2 / (Gvv*Gff)
                        
    ax[0].plot(freq, 10*np.log10(abs(Gff)))
    ax[1].plot(freq, 10*np.log10(abs(Gvv)))
    ax[2].plot(freq, 10*np.log10(abs(H1)))
    ax[2].plot(freq, 10*np.log10(abs(H2)))
    ax[3].set_ylim(-np.pi,np.pi)

    for i in range(-int(np.floor(min(np.unwrap(np.angle(H2[0:np.where(freq == 1000)[0][0]])))/np.pi/2))+1):
        ax[3].plot(freq, np.unwrap(np.angle(H2)) + i*2*np.pi, color = 'blue')
    for i in range(int(np.ceil(max(np.unwrap(np.angle(H2[0:np.where(freq == 1000)[0][0]])))/np.pi/2))+1):
        ax[3].plot(freq, np.unwrap(np.angle(H2)) - i*2*np.pi, color = 'blue')
    
    return H1, H2, gamma2, freq, Gff, Gvv, Gfv, counter