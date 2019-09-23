
class Parameters:

    def __init__(self):
        self.kmax = 10000
        self.rx_chan_cnt = 8
        self.rx_data_osr = 16
        self.tx_data_file = 'rxDataInPam4_noISI.npy'
        self.tx_data_gain = 6.0
        self.sim_mode = 0
        self.agc_tap_mu = 0.0
        self.ffe_tap_mu = 0.0
        self.dfe_tap_mu = 0.0
        self.dco_tap_mu = 0.0
        self.lvl_tap_mu = 0.0
        self.vga_tap_mu = 0.0
        self.phz_tap_mu = 0.0
        self.tr_prop_mu = 0.0
        self.tr_integ_mu = 0.0
        self.load_flag = True
        self.save_flag = True
        self.plot_flag = True

        self.afe_rms_noise = 0.02
        self.afe_rms_gain = 0.05
        self. afe_rms_offset = 0.05
        self. adc_rms_level = 0.05
        self. rx_rms_phase = 0.05
        self.rx_rms_jitter = 0.006
        self.rx_loop_demux = 16
        self.tr_sym_mode_flag = False
        self.tr_loop_latency = 32
        self.plot_skip_points = 100
        self.plot_window_snr = 1000

        self.ffe_tap_fix = 0
        self.ffe_tap_cnt = 5
        self.ffe_tap_ref = 1
        self.ffe_tap_pre = -0.01
        self.ffe_tap_post = -0.05
        self.ffe_tap_wts = []

        self.dfe_tap_cnt = 5
        self.dfe_tap_ref = 4
        self.dfe_tap_wts = []

        self.agc_tap_step = 0.01
        self.ffe_tap_step = 0.01
        self.dfe_tap_step = 0.01
        self.dco_tap_step = 0.01
        self.lvl_tap_step = 0.01
        self.vga_tap_step = 0.01
        self.phz_tap_step = 0.006  # phase trim step size

        self.phz_tap_max = 15.0
        self.beat_frequency = 0.0
        self.tr_loop_period = 0.0
