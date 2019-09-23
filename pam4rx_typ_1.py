# -*- coding: utf-8 -*-
"""
@author: mle@sitrus-tech.com
Copyright (C) 2018 by Michael Q. Le, Ph.D.
"""

import re
import math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import style
from scipy import signal
from pam_cfg import get_cfg_args


  # global variables assignable through the command line
kmax         = 100000     # number of samples
rx_chan_cnt  = 8          # interleaved channels
rx_data_osr  = 16         # input data oversampling ratio
#tx_data_file = 'rxDataInPam4_typ.npy'
tx_data_file = 'rxDataInPam4_noISI.npy'
tx_data_gain = 6.0        # 15.0, 8.0, 6.0
sim_mode     = 0          # simulation stage -> 1 = VGA, 2 = per chan AGC, 3 = adapt EQ, 4 = non-linearity comp, 5 = eye diagram
agc_tap_mu   = 0.0        # AGC loop gain
ffe_tap_mu   = 0.0        # FFE loop gain
dfe_tap_mu   = 0.0        # DFE loop gain
dco_tap_mu   = 0.0        # DCO loop gain
lvl_tap_mu   = 0.0        # Lvl loop gain
vga_tap_mu   = 0.0        # VGA loop gain
phz_tap_mu   = 0.0        # phase trim loop gain
tr_prop_mu   = 0.0        # Timing recovery proportional path gain
tr_integ_mu  = 0.0        # Timing recovery integral path gain
load_flag    = True
save_flag    = True
plot_flag    = True

  # add non-idealities
afe_rms_noise  = 0.02      # rms noise at the input (0.067 -> TIA PAM4 SNR = 23.5dB)
afe_rms_gain   = 0.05      # rms gain variation between channels (normalized to 1)
afe_rms_offset = 0.05      # rms offset between channels (normalized to 1)
adc_rms_level  = 0.05      # rms slicer level mismatches (normalized to 1, 1/22 = 3*sigma -> 0.02 sigma)
rx_rms_phase   = 0.05      # rms sample time phase error (0.006 = 210fs @28GHz)
rx_rms_jitter  = 0.006     # rms jitter (0.006 = 210fs @28GHz)
rx_loop_demux  = 16        # receiver parallel factor
tr_sym_mode_flag  = False  # Use only symetric transitions for TR if true

afe_rms_noise  = 0.0      # rms noise at the input (0.067 -> TIA PAM4 SNR = 23.5dB)
afe_rms_gain   = 0.0      # rms gain variation between channels (normalized to 1)
afe_rms_offset = 0.0      # rms offset between channels (normalized to 1)
adc_rms_level  = 0.0      # rms slicer level mismatches (normalized to 1, 1/22 = 3*sigma -> 0.02 sigma)
rx_rms_phase   = 0.0      # rms sample time phase error (0.006 = 210fs @28GHz)
rx_rms_jitter  = 0.0     # rms jitter (0.006 = 210fs @28GHz)

  # FFE
ffe_tap_fix  = 0          # fixed tap, set to out of bounds to disable, 0 = fix pre-cursor coefficient
ffe_tap_cnt  = 5
ffe_tap_ref  = 1
#ffe_tap_pre  = -0.1
#ffe_tap_post = -0.3
ffe_tap_pre  = -0.01
ffe_tap_post = -0.05
ffe_tap_wts  = []

  # DFE
dfe_tap_cnt = 5
dfe_tap_ref = 4
dfe_tap_wts = []

agc_tap_step = 0.01
ffe_tap_step = 0.01
dfe_tap_step = 0.01
dco_tap_step = 0.01
lvl_tap_step = 0.01
vga_tap_step = 0.01
phz_tap_step = 0.006      # phase trim step size

phz_tap_max  = 15.0        # maximum LSB phase steps


agc_tap_wts = []          # per channel AGC coefficients
dco_tap_wts = []          # DCO coefficients
lvl_tap_wts = []          # slicer level adjustment for all channels
phz_tap_wts = []          # phase trim coefficients
vga_tap_wts = []


args = get_cfg_args()
if args['kmax']:
    kmax = args['kmax']
if args['chan']:
    rx_chan_cnt = args['chan']
if args['osr']:
    rx_data_osr = args['osr']
if args['txdata']:
    tx_data_file = args['txdata']
if args['txgain']:
    tx_data_gain = args['txgain']
if args['mode']:
    sim_mode = args['mode']    

if args['agcmu']:
    agc_tap_mu = args['agcmu']
if args['ffemu']:
    ffe_tap_mu = args['ffemu']
if args['dfemu']:
    dfe_tap_mu = args['dfemu']
if args['dcomu']:
    dco_tap_mu = args['dcomu']
if args['lvlmu']:
    lvl_tap_mu = args['lvlmu']
if args['vgamu']:
    vga_tap_mu = args['vgamu']
if args['promu']:
    tr_prop_mu = args['promu']
if args['intmu']:
    tr_integ_mu = args['intmu']

if args['load']:
    load_flag = True
if args['save']:
    save_flag = True
if args['plot']:
    plot_flag = True


  ######################
  # internal variables #
  ######################
chan                 = 0           # current interleaved channel
rx_data_sample       = [0.0]*kmax  # sampled data input
rx_data_sample_sign  = [0.0]*kmax  # sign of the sampled input
rx_eq_out            = [0.0]*kmax  # equalized rx output
rx_eq_out_pam        = [0.0]*kmax  # decision
rx_eq_error          = [0.0]*kmax  # slicer error
rx_eq_error_sign     = [0.0]*kmax  # sign of error
ffe_tap_out          = [0.0]*kmax  # ffe output
dfe_tap_out          = [0.0]*kmax  # dfe output
tr_loop_phase        = [0.0]*kmax  # timing recover loop phase
demux_count          = 0           # demux counter

rx_eq_out_pam_bins = [0, 0, 0, 0, 0, 0, 0, 0]

  # Timing Recovery
tr_loop_time      = 0.5*float(rx_data_osr)   # initial phase (0.0 to 1.0)
tr_loop_time_floor = 0
tr_loop_time_ceil  = 0
tr_loop_sample    = 0.0
tr_loop_period    = np.float(1.0000*rx_data_osr)
#tr_loop_period    = np.float(1.000100*rx_data_osr)      # -250PPM frequency offset
#tr_loop_period    = np.float(0.99975*rx_data_osr)      # +250PPM frequency offset

tr_loop_prop     = [0.0]*kmax
tr_loop_integ    = [0.0]*kmax
tr_loop_latency  = 32
tr_loop_pdout    = [0.0]*tr_loop_latency

plot_skip_points = 100         # number of points to skip between plot points
plot_window_snr  = 1000        # number of samples to use for SNR calculation


  ###########################
  # load ideal channel data #
  ###########################
if re.search(".npy", tx_data_file):
    rxDataIn   = np.load(tx_data_file)
else:
      # read TX data from Cadence
    tx = open(tx_data_file, "r")
    txData = tx.read().split()
    for i in range(0, len(txData)):
        txData[i] = float(txData[i])
    txDataMean = np.mean(txData)
    rxDataIn = txData - txDataMean


  #############################
  # configure simulation mode #
  #############################
if args['mode']:

    sim_mode = args['mode']

      # adjust the coarse VGA
    if sim_mode == 1:

        load_flag = False
        save_flag = True
        plot_flag = True
    
        agc_tap_mu = 0.0
        ffe_tap_mu = 0.0
        dco_tap_mu = 0.0
        dfe_tap_mu = 0.0
        lvl_tap_mu = 0.0
        phz_tap_mu = 0.0
        vga_tap_mu = 0.01
        tr_prop_mu = 0.00005
        tr_integ_mu = tr_prop_mu/512.0

      # adjust the phase trim
    elif sim_mode == 2:

        load_flag = True
        save_flag = True
        plot_flag = True
       
        agc_tap_mu = 0.0
        ffe_tap_mu = 0.0
        dco_tap_mu = 0.0
        dfe_tap_mu = 0.0
        lvl_tap_mu = 0.0
        phz_tap_mu = 0.002
        vga_tap_mu = 0.0
        tr_prop_mu = 0.000025
        tr_integ_mu = tr_prop_mu/512.0

      # adapt the equalizers
    elif sim_mode == 3:

        load_flag = True
        save_flag = True
        plot_flag = True
    
        agc_tap_mu = 0.002
        ffe_tap_mu = 0.002
        dco_tap_mu = 0.002
        dfe_tap_mu = 0.002
        lvl_tap_mu = 0.0
        phz_tap_mu = 0.0
        vga_tap_mu = 0.0
        tr_prop_mu = 0.000025
        tr_integ_mu = tr_prop_mu/512.0    

      # freeze equalizers and adjust slicer levels
    elif sim_mode == 4:

        load_flag = True
        save_flag = True
        plot_flag = True
    
        agc_tap_mu = 0.0
        ffe_tap_mu = 0.0
        dco_tap_mu = 0.0
        dfe_tap_mu = 0.0
        lvl_tap_mu = 0.002
        phz_tap_mu = 0.0
        vga_tap_mu = 0.0
        tr_prop_mu = 0.000025
        tr_integ_mu = tr_prop_mu/512.0    

      # plot eye diagram
    elif sim_mode == 5:

        load_flag = True
        save_flag = False
        plot_flag = False
    
        agc_tap_mu = 0.0
        ffe_tap_mu = 0.0
        dco_tap_mu = 0.0
        dfe_tap_mu = 0.0
        lvl_tap_mu = 0.0
        phz_tap_mu = 0.0
        vga_tap_mu = 0.0
        tr_prop_mu  = 0.0
        tr_integ_mu = 0.0
        beat_frequency = 1.0 + np.float(1.0/kmax)
        tr_loop_period = np.float(beat_frequency*rx_data_osr)
        plot_skip_points  = 1
        plot_window_snr   = 100 


  # initialize ffe tap weights
for i in range(0,rx_chan_cnt):
    ffe_tap_wts.append([])
    for j in range(0,ffe_tap_cnt):
        if j == ffe_tap_ref:
            ffe_tap_wts[i].append(1.0/ffe_tap_step)
        elif j == ffe_tap_ref+1:
            ffe_tap_wts[i].append(ffe_tap_post/ffe_tap_step)
        elif j == ffe_tap_ref-1:
            ffe_tap_wts[i].append(ffe_tap_pre/ffe_tap_step)
        else:
            ffe_tap_wts[i].append(0.0)
        
  # initialize dfe tap weights
for i in range(0,rx_chan_cnt):
    dfe_tap_wts.append([])
    for j in range(0,dfe_tap_cnt):
        dfe_tap_wts[i].append(0.0)

  # initialize agc and dco tap weights
for i in range(0,rx_chan_cnt):
    agc_tap_wts.append(0.0)
    dco_tap_wts.append(0.0)
    phz_tap_wts.append(0.0)

  
vga_tap_wts.append(tx_data_gain/vga_tap_step)


  # initialize non-idealities
adc_random_sample_sign_levels = np.random.normal(scale=adc_rms_level,  size=rx_chan_cnt)
afe_random_channel_offsets    = np.random.normal(scale=afe_rms_offset, size=rx_chan_cnt)
afe_random_channel_gains      = np.random.normal(scale=afe_rms_gain,   size=rx_chan_cnt)
afe_random_channel_phases     = np.random.normal(scale=rx_rms_phase,   size=rx_chan_cnt)

  # TEST channel indexes
#afe_random_channel_offsets = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]       # correlates with coefficient at index = chan + ffe_tap_ref because FFEtap B1 = 1.0
#afe_random_channel_gains   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]       # correlates with coefficient at FFE B1
#afe_random_channel_phases  = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]      # correlates with coefficient at FFE B2

  # pam 4 slicer thresholds
adc_random_channel_levels = []
for i in range(0,rx_chan_cnt):
    adc_random_channel_levels.append([])

      # add residual random offset to each comparator level [6:0] = [3 2 1 0 -1 -2 -3]
    adc_random_channel_levels[i].append(-3.0 + np.random.normal(scale=adc_rms_level))
    adc_random_channel_levels[i].append(-2.0 + np.random.normal(scale=adc_rms_level))
    adc_random_channel_levels[i].append(-1.0 + np.random.normal(scale=adc_rms_level))
    adc_random_channel_levels[i].append( 0.0 + np.random.normal(scale=adc_rms_level))
    adc_random_channel_levels[i].append( 1.0 + np.random.normal(scale=adc_rms_level))
    adc_random_channel_levels[i].append( 2.0 + np.random.normal(scale=adc_rms_level))
    adc_random_channel_levels[i].append( 3.0 + np.random.normal(scale=adc_rms_level))


  ##########################
  # variables for plotting #
  ##########################
plot_shift_snr = 100         # shift the window by this amount for each SNR calculation
plot_window_pd = 200         # number of PD outputs to average
plot_end_pd    = 10000       # last PD output sample to plot
plot_size_x    = 16
plot_size_y    = 9
plot_period    = 35000       # period in fempto-seconds
plot_count     = 0
plot_samples   = np.int(kmax/plot_skip_points)
plot_xdata     = [0]*plot_samples
for i in range (0,plot_samples):
    plot_xdata[i] = i*plot_skip_points

agc_tap_wts_plot = []
ffe_tap_wts_plot = []
dco_tap_wts_plot = []
dfe_tap_wts_plot = []
lvl_tap_wts_plot = []
vga_tap_wts_plot = []
phz_tap_wts_plot = []


ffe_tap_wts_plot.append([])
for i in range(0,ffe_tap_cnt):
    ffe_tap_wts_plot.append([])
    if i == ffe_tap_ref:
        for j in range(0,plot_samples):
            ffe_tap_wts_plot[i].append(1.0/ffe_tap_step)
    elif i == ffe_tap_ref+1:
        for j in range(0,plot_samples):
            ffe_tap_wts_plot[i].append(ffe_tap_post/ffe_tap_step)
    elif i == ffe_tap_ref-1:
        for j in range(0,plot_samples):
            ffe_tap_wts_plot[i].append(ffe_tap_pre/ffe_tap_step)
    else:
        for j in range(0,plot_samples):
            ffe_tap_wts_plot[i].append(0.0)

dfe_tap_wts_plot.append([])
for i in range(0,dfe_tap_cnt):
    dfe_tap_wts_plot.append([])
    for j in range(0,plot_samples):
        dfe_tap_wts_plot[i].append(0.0)

agc_tap_wts_plot.append([])
dco_tap_wts_plot.append([])
phz_tap_wts_plot.append([])
for i in range(0,rx_chan_cnt):
    agc_tap_wts_plot.append([])
    dco_tap_wts_plot.append([])
    phz_tap_wts_plot.append([])
    for j in range(0,plot_samples):
        agc_tap_wts_plot[i].append(0.0)
        dco_tap_wts_plot[i].append(0.0)
        phz_tap_wts_plot[i].append([])

lvl_tap_wts_plot.append([])
for i in range(0,7):
    lvl_tap_wts.append(0.0)
    lvl_tap_wts_plot.append([])
    for j in range(0,plot_samples):
        lvl_tap_wts_plot[i].append(0.0)    

vga_tap_wts_plot.append([])
for j in range(0,plot_samples):
    vga_tap_wts_plot[0].append(0.0)


  #########################
  # load config from file #
  #########################
if load_flag :

      # coefficients
    agc_tap_wts = np.load('agc_tap_wts.npy')
    ffe_tap_wts = np.load('ffe_tap_wts.npy')
    dco_tap_wts = np.load('dco_tap_wts.npy')
    dfe_tap_wts = np.load('dfe_tap_wts.npy')
    lvl_tap_wts = np.load('lvl_tap_wts.npy')
    phz_tap_wts = np.load('phz_tap_wts.npy')
    vga_tap_wts = np.load('vga_tap_wts.npy')


      # random mismatches
    adc_random_sample_sign_levels = np.load('adc_random_sample_sign_levels.npy')
    afe_random_channel_offsets   = np.load('afe_random_channel_offsets.npy')
    afe_random_channel_gains     = np.load('afe_random_channel_gains.npy')
    afe_random_channel_phases    = np.load('afe_random_channel_phases.npy')
    adc_random_channel_levels = np.load('adc_random_channel_levels.npy')

      # initial sampling phase
    tr_loop_time = np.load('tr_loop_phase.npy')
    tr_loop_integ_val = np.load('tr_loop_integ.npy')
    for i in range(0,kmax):
        tr_loop_integ[i] = tr_loop_integ_val



  ########################
  # process kmax samples #
  ########################
kStart = ffe_tap_cnt + dfe_tap_cnt
for k in range(kStart,kmax):
    
      # get next data sample
    tr_loop_time       = tr_loop_time + tr_loop_period - rx_data_osr*(tr_integ_mu*tr_loop_integ[k-1] + tr_prop_mu*tr_loop_prop[k-1])
    tr_loop_phase[k]   = np.remainder(tr_loop_time, rx_data_osr)/rx_data_osr
    tr_loop_time_quant = tr_loop_time + rx_data_osr*(np.random.normal(scale=rx_rms_jitter) + afe_random_channel_phases[chan] - phz_tap_step*np.around(phz_tap_wts[chan]))
#    tr_loop_time_quant = tr_loop_time + rx_data_osr*(np.random.normal(scale=rx_rms_jitter) + afe_random_channel_phases[chan])
    tr_loop_time_floor = int(np.floor(tr_loop_time_quant))
    tr_loop_time_ceil  = int(np.ceil(tr_loop_time_quant))
    tr_loop_sample     = np.interp(tr_loop_time_quant, [tr_loop_time_floor, tr_loop_time_ceil], [rxDataIn[tr_loop_time_floor], rxDataIn[tr_loop_time_ceil]])

      # add channel dependent gain error
    rx_data_sample[k] = (1.0 + afe_random_channel_gains[chan])*vga_tap_step*np.round(vga_tap_wts[0])*(1.0 + agc_tap_step*np.around(agc_tap_wts[chan]))*tr_loop_sample
    
      # add random noise and channel dependent offset
    rx_data_sample[k] = rx_data_sample[k] + np.random.normal(scale=afe_rms_noise) + afe_random_channel_offsets[chan]

      # get the sign of the input sample with channel dependent slicer offset
    if rx_data_sample[k] > adc_random_sample_sign_levels[chan]:
        rx_data_sample_sign[k] = 1.0
    else:
        rx_data_sample_sign[k] = -1.0


          # save coefficient data of one channel for plotting
        if k > (plot_count*plot_skip_points):
            for i in range (0,rx_chan_cnt):
                agc_tap_wts_plot[i][plot_count] = agc_tap_step*np.round(agc_tap_wts[i])
            for i in range (0, ffe_tap_cnt):
                ffe_tap_wts_plot[i][plot_count] = ffe_tap_step*np.round(ffe_tap_wts[chan][i] - ffe_tap_wts_plot[i][plot_count])
            for i in range (0,rx_chan_cnt):
                dco_tap_wts_plot[i][plot_count] = dco_tap_step*np.round(dco_tap_wts[i])
                phz_tap_wts_plot[i][plot_count] = phz_tap_step*np.round(phz_tap_wts[i])
            for i in range (0, dfe_tap_cnt):
                dfe_tap_wts_plot[i][plot_count] = dfe_tap_step*np.round(dfe_tap_wts[chan][i])
            for i in range (0,7):
                lvl_tap_wts_plot[i][plot_count] = lvl_tap_step*np.round(lvl_tap_wts[i])
            vga_tap_wts_plot[0][plot_count] = vga_tap_step*np.round(vga_tap_wts[0])
            plot_count += 1

      # compute the FFE output
    for i in range (0, ffe_tap_cnt):
        ffe_tap_out[k] = ffe_tap_out[k] + ffe_tap_step*np.around(ffe_tap_wts[chan][i])*rx_data_sample[k-i]

      # compute the DFE output
    for i in range (0, dfe_tap_cnt):
        dfe_tap_out[k] = dfe_tap_out[k] + dfe_tap_step*np.around(dfe_tap_wts[chan][i])*rx_eq_out_pam[k-i-dfe_tap_ref]

      # compute the total equalized rx data output
    rx_eq_out[k] = ffe_tap_out[k] - dfe_tap_out[k] - dco_tap_step*np.around(dco_tap_wts[chan])

      # compute the error and quantize
    if   rx_eq_out[k] > (adc_random_channel_levels[chan][5] + lvl_tap_step*np.around(lvl_tap_wts[5])):
        rx_eq_out_pam[k] = 3.0
        rx_eq_error[k] = rx_eq_out[k] - (adc_random_channel_levels[chan][6] + lvl_tap_step*np.around(lvl_tap_wts[6]))
    elif rx_eq_out[k] > (adc_random_channel_levels[chan][3] + np.around(lvl_tap_wts[3], decimals=2)):
        rx_eq_out_pam[k] = 1.0
        rx_eq_error[k] = rx_eq_out[k] - (adc_random_channel_levels[chan][4] + lvl_tap_step*np.around(lvl_tap_wts[4]))
    elif rx_eq_out[k] > (adc_random_channel_levels[chan][1] + lvl_tap_step*np.around(lvl_tap_wts[1])):
        rx_eq_out_pam[k] = -1.0
        rx_eq_error[k] = rx_eq_out[k] - (adc_random_channel_levels[chan][2] + lvl_tap_step*np.around(lvl_tap_wts[2]))
    else:
        rx_eq_out_pam[k] = -3.0
        rx_eq_error[k] = rx_eq_out[k] - (adc_random_channel_levels[chan][0] + lvl_tap_step*np.around(lvl_tap_wts[0]))

      # re-center the PAM4 levels
    lvl_tap_wts[5] = lvl_tap_step*np.around((lvl_tap_wts[6] + lvl_tap_wts[4])/2.0)
    lvl_tap_wts[3] = lvl_tap_step*np.around((lvl_tap_wts[4] + lvl_tap_wts[2])/2.0)
    lvl_tap_wts[1] = lvl_tap_step*np.around((lvl_tap_wts[2] + lvl_tap_wts[0])/2.0)

      # data for PAM output histogram
    if rx_eq_error[k] > 0.0:
        rx_eq_error_sign[k] = 1.0
        if rx_eq_out_pam[k] == 3.0:
            rx_eq_out_pam_bins[7] += 1
        elif rx_eq_out_pam[k] == 1.0:
            rx_eq_out_pam_bins[5] += 1
        elif rx_eq_out_pam[k] == -1.0:
            rx_eq_out_pam_bins[3] += 1
        else:
            rx_eq_out_pam_bins[1] += 1
    else:
        rx_eq_error_sign[k] = -1.0
        if rx_eq_out_pam[k] == 3.0:
            rx_eq_out_pam_bins[6] += 1
        elif rx_eq_out_pam[k] == 1.0:
            rx_eq_out_pam_bins[4] += 1
        elif rx_eq_out_pam[k] == -1.0:
            rx_eq_out_pam_bins[2] += 1
        else:
            rx_eq_out_pam_bins[0] += 1

      # update the slicer levels
    if rx_eq_out_pam[k] == 3.0:
        lvl_tap_wts[6] += lvl_tap_mu*rx_eq_error_sign[k]
    elif rx_eq_out_pam[k] == 1.0:
        lvl_tap_wts[4] += lvl_tap_mu*rx_eq_error_sign[k]
    elif rx_eq_out_pam[k] == -1.0:
        lvl_tap_wts[2] += lvl_tap_mu*rx_eq_error_sign[k]
    else:
        lvl_tap_wts[0] += lvl_tap_mu*rx_eq_error_sign[k]

      # update FFE coefficients with sign-sign LMS algorithm
    for i in range(0,ffe_tap_cnt):
        if i == ffe_tap_ref:
            agc_chan = chan - ffe_tap_ref
            if agc_chan < 0:
                agc_chan = agc_chan + rx_chan_cnt
            agc_tap_wts[agc_chan] = agc_tap_wts[agc_chan] - agc_tap_mu*rx_eq_error_sign[k]*rx_data_sample_sign[k-i]
        elif i != ffe_tap_fix:
            ffe_tap_wts[chan][i] = ffe_tap_wts[chan][i] - ffe_tap_mu*rx_eq_error_sign[k]*rx_data_sample_sign[k-i]

      # update DFE coefficients with sign-sign LMS algorithm
    for i in range(0,dfe_tap_cnt):
        dfe_tap_wts[chan][i] = dfe_tap_wts[chan][i] + dfe_tap_mu*rx_eq_error_sign[k]*rx_eq_out_pam[k-i-dfe_tap_ref]

      # updated slice DCO
    dco_tap_wts[chan] = dco_tap_wts[chan] + dco_tap_mu*rx_eq_error_sign[k]

      # VGA only uses +/-3 PAM levels
    if (rx_eq_out_pam[k] == 3.0) or (rx_eq_out_pam[k] == -3.0):
        vga_tap_wts[0] = vga_tap_wts[0] - vga_tap_mu*rx_eq_error_sign[k]*rx_data_sample_sign[k-ffe_tap_ref]

      # add latency in the timing recovery loop
    for i in range(0, tr_loop_latency-1):
        tr_loop_pdout[i] = tr_loop_pdout[i+1]

      # quantized baud-rate Gardner timing recovery
      # discard assymmetric transitions
    if tr_sym_mode_flag :
        if ((abs(rx_eq_out_pam[k]) == 3) and (abs(rx_eq_out_pam[k-2]) == 3)) or ((abs(rx_eq_out_pam[k]) == 1) and (abs(rx_eq_out_pam[k-2]) == 1)):
            tr_loop_pdout[tr_loop_latency-1] = rx_eq_error_sign[k-1]*(rx_eq_out_pam[k] - rx_eq_out_pam[k-2])
        else:
            tr_loop_pdout[tr_loop_latency-1] = 0.0
    else:
        tr_loop_pdout[tr_loop_latency-1] = rx_eq_error_sign[k-1]*(rx_eq_out_pam[k] - rx_eq_out_pam[k-2])

  
    phz_chan = chan - ffe_tap_ref - 1
    if phz_chan < 0:
        phz_chan = phz_chan + rx_chan_cnt
    if abs(phz_tap_wts[phz_chan]) < phz_tap_max:  
        phz_tap_wts[phz_chan] = phz_tap_wts[phz_chan] + phz_tap_mu*tr_loop_pdout[tr_loop_latency-1]

      # receiver DSP runs at a lower data rate (parallel factor)
      # DTL and AGC run at Fpll/8
    if demux_count == rx_loop_demux:

          # update DTL
        tr_loop_prop[k] = 0.0
        for i in range(0, rx_loop_demux-1):
            tr_loop_prop[k] += tr_loop_pdout[i]
        tr_loop_integ[k] = tr_loop_integ[k-1] + tr_loop_prop[k]

        demux_count = 0
    else:
        tr_loop_prop[k]  = tr_loop_prop[k-1]
        tr_loop_integ[k] = tr_loop_integ[k-1]
        demux_count += 1

      # update the intervleaved channel index    
    chan = chan + 1
    if chan == rx_chan_cnt:
        chan = 0


  ##############################
  # save configuration to file #
  ##############################
if save_flag :
    np.save('agc_tap_wts.npy', agc_tap_wts)
    np.save('ffe_tap_wts.npy', ffe_tap_wts)
    np.save('dco_tap_wts.npy', dco_tap_wts)
    np.save('dfe_tap_wts.npy', dfe_tap_wts)
    np.save('lvl_tap_wts.npy', lvl_tap_wts)
    np.save('phz_tap_wts.npy', phz_tap_wts)
    np.save('vga_tap_wts.npy', vga_tap_wts)
    np.save('adc_random_sample_sign_levels.npy', adc_random_sample_sign_levels)
    np.save('afe_random_channel_offsets.npy',   afe_random_channel_offsets)
    np.save('afe_random_channel_gains.npy',     afe_random_channel_gains)
    np.save('afe_random_channel_phases.npy',    afe_random_channel_phases)
    np.save('adc_random_channel_levels.npy', adc_random_channel_levels)
    np.save('tr_loop_phase.npy', rx_data_osr*tr_loop_phase[k])
    np.save('tr_loop_integ.npy', tr_loop_integ[k])


  ##################    
  # generate plots #
  ##################
if plot_flag :
    
    style.use('seaborn')

      # plot the phase detector output
    if tr_prop_mu > 0.0:
        tr_pdsum_avg = []
        tr_pdsum_avg_x = []
        for i in range(0,int(plot_end_pd/plot_window_pd)):
            tr_pdsum_avg.append(np.mean(tr_loop_prop[i*plot_window_pd:i*plot_window_pd+plot_window_pd:1]))
            tr_pdsum_avg_x.append((i*plot_window_pd + int(plot_window_pd/2)))
        trPDsum = tr_loop_prop[1:plot_end_pd:1]
        fig0 = plt.figure()
        fig0.set_size_inches(plot_size_x, plot_size_y)
        ax0  = fig0.add_subplot(111)
        ax0.plot(trPDsum)
        ax0.plot(tr_pdsum_avg_x, tr_pdsum_avg, 'r')
        plt.grid(True)
        plt.xlim(0,plot_end_pd)
        plt.xlabel('Sample Time')
        plt.ylabel('PDout')
        plt.title('Phase Detector Output Sum')
        plt.show()

      # plot the integral path
    if tr_integ_mu > 0.0:
        tr_scale_factor = tr_integ_mu*1e6
        fig1 = plt.figure()
        fig1.set_size_inches(plot_size_x, plot_size_y)
        ax1  = fig1.add_subplot(111)
        tr_loop_integ_scaled = [np.float(val*tr_scale_factor) for val in tr_loop_integ]
        ax1.plot(tr_loop_integ_scaled)
        plt.grid(True)
        plt.xlabel('Sample Time')
        plt.ylabel('Integral Value (PPM)')
        plt.title('Timing Recovery Integral Value (PPM)')
        plt.show()

      # plot the DTL sampling phase
    if tr_prop_mu > 0.0:
        fig2 = plt.figure()
        fig2.set_size_inches(plot_size_x, plot_size_y)
        ax2  = fig2.add_subplot(111)
        ax2.plot(tr_loop_phase, '.')
        plt.grid(True)
        plt.xlabel('Sample Time')
        plt.ylabel('Phase Value')
        plt.title('Timing Recovery Phase Value')
        plt.show()

      # plot VGA coefficients
    if vga_tap_mu > 0.0:
        fig3a = plt.figure()
        fig3a.set_size_inches(plot_size_x, plot_size_y)
        ax3a  = fig3a.add_subplot(111) 
        ax3a.plot(plot_xdata,vga_tap_wts_plot[0])
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.xlabel('Sample')
        plt.ylabel('VGA Delta Values')
        plt.title('AFE VGA Coefficients')
        plt.show()

      # plot AGC coefficients
    if agc_tap_mu > 0.0:
        fig3b = plt.figure()
        fig3b.set_size_inches(plot_size_x, plot_size_y)
        ax3b  = fig3b.add_subplot(111) 
        for i in range(0,rx_chan_cnt):
            ax3b.plot(plot_xdata,agc_tap_wts_plot[i], label='A %1.0f' %i)
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.xlabel('Sample')
        plt.ylabel('AGC Delta Values')
        plt.title('Per Channel AGC Coefficients')
        plt.show()

      # plot DCO coefficients
    if dco_tap_mu > 0.0:
        fig3c = plt.figure()
        fig3c.set_size_inches(plot_size_x, plot_size_y)
        ax3c  = fig3c.add_subplot(111) 
        for i in range(0,rx_chan_cnt):
            ax3c.plot(plot_xdata,dco_tap_wts_plot[i], label='D %1.0f' %i)
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.xlabel('Sample')
        plt.ylabel('DCO Delta Values')
        plt.title('Per Channel DCO Coefficients')
        plt.show()

      # plot PHZ coefficients
    if phz_tap_mu > 0.0:
        fig3d = plt.figure()
        fig3d.set_size_inches(plot_size_x, plot_size_y)
        ax3d  = fig3d.add_subplot(111) 
        for i in range(0,rx_chan_cnt):
            ax3d.plot(plot_xdata,phz_tap_wts_plot[i], label='PHZ %1.0f' %i)
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.xlabel('Sample')
        plt.ylabel('Phase Trim Values')
        plt.title('Per Channel Phase Trim Coefficients')
        plt.show()

      # plot FFE coefficients
    if ffe_tap_mu > 0.0:
        fig3 = plt.figure()
        fig3.set_size_inches(plot_size_x, plot_size_y)
        ax3  = fig3.add_subplot(111) 
        for i in range(0,ffe_tap_cnt):
            ax3.plot(plot_xdata,ffe_tap_wts_plot[i], label='B %1.0f' %i)
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.xlabel('Sample')
        plt.ylabel('FFE Delta Values')
        plt.title('FFE Coefficients')
        plt.show()

      # plot DFE coefficients
    if dfe_tap_mu > 0.0 and dfe_tap_cnt > 0:
        fig4 = plt.figure()
        fig4.set_size_inches(plot_size_x, plot_size_y)
        ax4  = fig4.add_subplot(111) 
        for i in range(0,dfe_tap_cnt):
            ax4.plot(plot_xdata,dfe_tap_wts_plot[i], label='C %1.0f' %i)
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.xlabel('Sample')
        plt.ylabel('DFE Values')
        plt.title('DFE Coefficients')
        plt.show()

      # FFE+DFE frequency response
    if (ffe_tap_mu > 0.0) and (dfe_tap_mu > 0.0):
        B = np.array(ffe_tap_wts[0])*ffe_tap_step
        A = [ 1.0 ]
        for i in range(1,dfe_tap_ref):
            A.append(0.0)
        for i in range(0,dfe_tap_cnt):
            A.append(dfe_tap_step*dfe_tap_wts[0][i])
        w, h = signal.freqz(B, A)
        h_dB = 20 * np.log10(abs(h))
        h_boost = max(h_dB) - min(h_dB)
        fig34 = plt.figure()
        fig34.set_size_inches(plot_size_x, plot_size_y)
        ax34  = fig34.add_subplot(111)
        ax34.plot(w, h_dB, 'b')
        ax34.set_ylabel('Amplitude [dB]', color='b')
        ax34.set_xlabel('Frequency [rad/sample]')
        ax34.set_title('FFE + DFE Frequency Response')
        plt.grid(True)
        plt.text(max(w), max(h_dB), "Frequency boost = %1.2f dB" % h_boost, ha='right')
        plt.show()       

      # plot slicer level coefficients
    if lvl_tap_mu > 0.0:
        fig5 = plt.figure()
        fig5.set_size_inches(plot_size_x, plot_size_y)
        ax5  = fig5.add_subplot(111) 
        ax5.plot(plot_xdata,lvl_tap_wts_plot[0], label='S0 -3')
        ax5.plot(plot_xdata,lvl_tap_wts_plot[2], label='S2 -1')
        ax5.plot(plot_xdata,lvl_tap_wts_plot[4], label='S4 +1')
        ax5.plot(plot_xdata,lvl_tap_wts_plot[6], label='S6 +3')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.xlabel('Sample')
        plt.ylabel('Slicer Level Values')
        plt.title('Slicer Level Adjustments')
        plt.show()

      # plot RX input
    eqIn = [0.0]*int(kmax/plot_skip_points)
    fig6 = plt.figure()
    fig6.set_size_inches(plot_size_x, plot_size_y)
    ax6  = fig6.add_subplot(111) 
    for i in range(0,int(kmax/plot_skip_points)):
        eqIn[i] = rx_data_sample[i*plot_skip_points]
    ax6.plot(plot_xdata,eqIn, '.', color='b')
    plt.grid(True)
    plt.xlim(0,kmax)
    plt.xlabel('Sample')
    plt.ylabel('Input Sample')
    plt.title('FFE Input Samples')
    plt.show()

      # plot slicer input
    sliceIn = [0.0]*int(kmax/plot_skip_points)
    fig7 = plt.figure()
    fig7.set_size_inches(plot_size_x, plot_size_y)
    ax7  = fig7.add_subplot(111) 
    for i in range(0,int(kmax/plot_skip_points)):
        sliceIn[i] = rx_eq_out[i*plot_skip_points]
    ax7.plot(plot_xdata,sliceIn, '.', color='b')
    plt.grid(True)
    plt.ylim(-3.5, 3.5)
    plt.xlim(0,kmax)
    plt.xlabel('Sample')
    plt.ylabel('Soft Decisions')
    plt.title('Equalized Slicer Input')
    plt.show()

      # plot histogram
    binPts = np.int(kmax/8)
    binCnt = np.int(binPts/1000)
    if binCnt > 500:
        binCnt = 500
    elif binCnt < 100:
        binCnt = 100
    fig8 = plt.figure()
    fig8.set_size_inches(plot_size_x, plot_size_y)
    ax8  = fig8.add_subplot(111)
    binHits, binVals, binPack = ax8.hist(rx_eq_out[kmax-binPts:kmax-1:1], bins=binCnt)
    plt.grid(True)
    plt.xlabel('Bins')
    plt.ylabel('Hits')
    plt.title('Histogram of Equalized Signal (%1.0f Samples)' %binPts)
    plt.show()

      # plot MSE
    pend   = int(kmax/plot_shift_snr)
    pstart = int(plot_window_snr/plot_shift_snr)
    MSE = [0.0]*pend
    MSEx = [0.0]*pend
    fig9 = plt.figure()
    fig9.set_size_inches(plot_size_x, plot_size_y)
    ax9  = fig9.add_subplot(111)
    for i in range(pstart,pend):
        STDE = np.std(rx_eq_error[i*plot_shift_snr:i*plot_shift_snr+plot_window_snr:1])
        MSE[i] = 20*math.log10(1/STDE)
        MSEx[i] = (i+1)*plot_shift_snr
    ax9.plot(MSEx, MSE)
    plt.grid(True)
    plt.xlim(0,kmax)
    plt.xlabel('Sample Time')
    plt.ylabel('MSE')
    plt.title('Slicer MSE')
    plt.show()

      # plot CMP bins
    xb = []
    for i in range(len(rx_eq_out_pam_bins)):
        xb.append(i)
    fig10 = plt.figure()
    fig10.set_size_inches(plot_size_x, plot_size_y)
    ax10  = fig10.add_subplot(111)
    ax10.bar(xb, rx_eq_out_pam_bins)
    plt.grid(True)
    plt.xlabel('Bin')
    plt.ylabel('Hits')
    plt.title('Comparator Bins')
    plt.show()

      # calculate the eye opening
    binEye = []
    zeroFlag = 1
    lastInd = 0
    lastDeltaMin = np.int(0.05*binCnt/4)
    for i in range(0,len(binHits)):
        if zeroFlag == 1:
            if binHits[i] == 0:
                if i > (lastInd + lastDeltaMin):
                    zeroFlag = 0
                    binEye.append(i)
                    lastInd = i
        else:
            if binHits[i] > 0:
                if i > (lastInd + lastDeltaMin):
                    zeroFlag = 1
                    binEye.append(i)
                    lastInd = i
    
    print ("***********************************************")
    if len(binEye) > 5:
        deltaM1M3 = (binVals[binEye[1]] - binVals[binEye[0]])/2
        deltaP1M1 = (binVals[binEye[3]] - binVals[binEye[2]])/2
        deltaP3P1 = (binVals[binEye[5]] - binVals[binEye[4]])/2
        logM1M3 = 20*math.log10(deltaM1M3)
        logP1M1 = 20*math.log10(deltaP1M1)
        logP3P1 = 20*math.log10(deltaP3P1)
          # re-center the PAM4 levels
        snrFullScale = []
        snrLoss = 0.0
        lvl_tap_wts = np.array(lvl_tap_wts)*lvl_tap_step
        snrFullScale.append(20*math.log10(1.0 + lvl_tap_wts[1] - lvl_tap_wts[0]))
        snrFullScale.append(20*math.log10(1.0 + lvl_tap_wts[2] - lvl_tap_wts[1]))
        snrFullScale.append(20*math.log10(1.0 + lvl_tap_wts[3] - lvl_tap_wts[2]))
        snrFullScale.append(20*math.log10(1.0 + lvl_tap_wts[4] - lvl_tap_wts[3]))
        snrFullScale.append(20*math.log10(1.0 + lvl_tap_wts[5] - lvl_tap_wts[4]))
        snrFullScale.append(20*math.log10(1.0 + lvl_tap_wts[6] - lvl_tap_wts[5]))
        snrLoss = min(snrFullScale)

        print ("* Eye opening from -1 to -3 = %1.3f (%1.3f dB)" % (deltaM1M3, logM1M3))
        print ("* Eye opening from +1 to -1 = %1.3f (%1.3f dB)" % (deltaP1M1, logP1M1))
        print ("* Eye opening from +3 to +1 = %1.3f (%1.3f dB)" % (deltaP3P1, logP3P1))
        print ("*")
        print ("* Threshold adjustment SNR Penalty = %1.3f dB" % snrLoss)
        print ("***********************************************")

    i = int(kmax/plot_shift_snr) - 1
    finalMSE = 20*math.log10(1/(np.std(rx_eq_error[i*plot_shift_snr:i*plot_shift_snr+plot_window_snr:1])))
    finalPHZ = np.std(tr_loop_phase[i*plot_shift_snr:i*plot_shift_snr+plot_window_snr:1])*plot_period
    if finalPHZ > 0.25:
        dataPHZ = tr_loop_phase[i*plot_shift_snr:i*plot_shift_snr+plot_window_snr:1]
        for k in range (0, len(dataPHZ)):
            if dataPHZ[k] > 0.5:
                dataPHZ[k] = dataPHZ[k] - 1.0
    finalPHZ = np.std(dataPHZ)*plot_period


      # print formatted output
    print ("***************************")
    print ("* Slicer MSE = %1.3f dB" % finalMSE)
    print ("* DTL Jitter = %1.3f fs" % finalPHZ)
    print ("***************************")
    print ("")

    print ("Front-End VGA Gain:")
    vga_tap_wts = np.array(vga_tap_wts)*vga_tap_step
    print (["%1.3f" % j for j in vga_tap_wts])
    print ("")

    print ("Channel Sign Slicer Offset:")
    print (["%1.3f" % j for j in adc_random_sample_sign_levels])
    print ("")

    print (" Slicer level adjustments:")
    lvl_tap_wts = np.array(lvl_tap_wts)*lvl_tap_step
    print (["%1.4f" % j for j in lvl_tap_wts])
    print ("")

    print ("Channel Gain Mismatch:")
    print (["%1.3f" % j for j in afe_random_channel_gains])
    print ("") 

    print ("Per Channel Gain Coefficients:")
    agc_tap_wts = np.array(agc_tap_wts)*agc_tap_step
    print (["%1.3f" % j for j in agc_tap_wts])
    print ("")
    
    print ("Channel Offsets:")
    print (["%1.3f" % j for j in afe_random_channel_offsets])
    print ("")

    print ("DCO Coefficients:")
    dco_tap_wts = np.array(dco_tap_wts)*dco_tap_step
    print (["%1.3f" % j for j in dco_tap_wts])
    print ("")

    print ("Channel Phase Mismatch:")
    print (["%1.3f" % j for j in afe_random_channel_phases])
    print ("")

    print ("Phase Trim Coefficients:")
    phz_tap_wts = np.array(phz_tap_wts)*phz_tap_step
    print (["%1.3f" % j for j in phz_tap_wts])
    print ("")

    slice_cnt = rx_chan_cnt - 1
    if ffe_tap_cnt > 0:
        print ("FFE Coefficients, slice 0 to %i:" % slice_cnt)
        for i in range(0, rx_chan_cnt):
            ffe_tap_wts[i] = np.array(ffe_tap_wts[i])*ffe_tap_step
            print (["%1.3f" % j for j in ffe_tap_wts[i]])
        print ("")

    if dfe_tap_cnt > 0:
        print ("DFE Coefficients, slice 0 to %i:" % slice_cnt)
        for i in range(0, rx_chan_cnt):
            dfe_tap_wts[i] = np.array(dfe_tap_wts[i])*dfe_tap_step
            print (["%1.3f" % j for j in dfe_tap_wts[i]])
        print ("")


elif sim_mode == 5:

    style.use('seaborn')

    kStart += 2
    plot_xdata2 = [0.0]*(plot_samples-kStart)
    for i in range (0,plot_samples-kStart):
        plot_xdata2[i] = np.float(i+1)/(plot_samples-kStart)
    
          # plot RX input
    eqIn = [0.0]*(plot_samples-kStart)
    fig6 = plt.figure()
    fig6.set_size_inches(plot_size_x, plot_size_y)
    ax6  = fig6.add_subplot(111) 
    for i in range(0,int(kmax/2)):
        eqIn[i] = rx_data_sample[i + int(kmax/2)]
    for i in range(kStart,int(kmax/2)):
        eqIn[i - kStart + int(kmax/2)] = rx_data_sample[i]
    ax6.plot(plot_xdata2, eqIn, '.', color='b')
    plt.grid(True)
    plt.xlim(0,1)
    plt.xticks(np.arange(0, 1.0, 0.1))
    plt.ylim(-6, 6)
    plt.xlabel('UI')
    plt.ylabel('Input Sample')
    plt.title('FFE Input Samples')
    plt.show()

      # plot slicer input
    sliceIn = [0.0]*(plot_samples-kStart)
    fig7 = plt.figure()
    fig7.set_size_inches(plot_size_x, plot_size_y)
    ax7  = fig7.add_subplot(111) 
    for i in range(0,int(kmax/2)):
        sliceIn[i] = rx_eq_out[i + int(kmax/2)]
    for i in range(kStart,int(kmax/2)):
        sliceIn[i - kStart  + int(kmax/2)] = rx_eq_out[i]
    ax7.plot(plot_xdata2, sliceIn, '.', color='b')
    plt.grid(True)
    plt.xlim(0,1)
    plt.xticks(np.arange(0, 1.0, 0.1))
    plt.ylim(-6, 6)
    plt.xlabel('UI')
    plt.ylabel('Soft Decisions')
    plt.title('Equalized Slicer Input')
    plt.show()

      # plot MSE
    pend   = int(kmax/plot_shift_snr)
    pstart = int(plot_window_snr/plot_shift_snr)
    MSE = [0.0]*pend
    MSE2 = [0.0]*pend
    MSEx = [0.0]*pend
    fig9 = plt.figure()
    fig9.set_size_inches(plot_size_x, plot_size_y)
    ax9  = fig9.add_subplot(111)
    for i in range(pstart,pend):
        STDE = np.std(rx_eq_error[i*plot_shift_snr:i*plot_shift_snr+plot_window_snr:1])
        MSE[i] = 20*math.log10(1/STDE)
        MSEx[i] = np.float(i+1)/pend
    for i in range(0,int(pend/2)):
        MSE2[i] = MSE[i + int(pend/2)]
    for i in range(kStart,int(pend/2)):
        MSE2[i - kStart + int(pend/2)] = MSE[i]
    ax9.plot(MSEx, MSE2)
    plt.grid(True)
    plt.xlim(0,1)
    plt.xticks(np.arange(0, 1.0, 0.1))
    plt.xlabel('UI')
    plt.ylabel('MSE')
    plt.title('Slicer MSE')
    plt.show()

