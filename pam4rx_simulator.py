import kivy
kivy.require('1.0.5')

import os
from kivy.uix.floatlayout import FloatLayout
from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.popup import Popup
from parameters import Parameters
import numpy as np
from pam4rx_typ import simulate


class Gui(FloatLayout):
    pass


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class GuiApp(App):

    beat_frequency = 0

    def build(self):
        return Gui()

    def dismiss_popup(self):
        self._popup.dismiss()

    def configure(self):
        sim_mode = self.root.ids.sim_mode_input.text

        if sim_mode == '1':
            self.root.ids.load_flag_input.text = 'False'
            self.root.ids.save_flag_input.text = 'True'
            self.root.ids.plot_flag_input.text = 'True'
            self.root.ids.agc_tap_mu_input.text = '0.0'
            self.root.ids.ffe_tap_mu_input.text = '0.0'
            self.root.ids.dco_tap_mu_input.text = '0.0'
            self.root.ids.dfe_tap_mu_input.text = '0.0'
            self.root.ids.lvl_tap_mu_input.text = '0.0'
            self.root.ids.phz_tap_mu_input.text = '0.0'
            self.root.ids.vga_tap_mu_input.text = '0.01'
            self.root.ids.tr_prop_mu_input.text = '0.00005'
            tr_integ_mu = float(0.00005/512.0)
            self.root.ids.tr_integ_mu_input.text = str(tr_integ_mu)

        elif sim_mode == '2':
            self.root.ids.load_flag_input.text = 'True'
            self.root.ids.save_flag_input.text = 'True'
            self.root.ids.plot_flag_input.text = 'True'
            self.root.ids.agc_tap_mu_input.text = '0.0'
            self.root.ids.ffe_tap_mu_input.text = '0.0'
            self.root.ids.dco_tap_mu_input.text = '0.0'
            self.root.ids.dfe_tap_mu_input.text = '0.0'
            self.root.ids.lvl_tap_mu_input.text = '0.0'
            self.root.ids.phz_tap_mu_input.text = '0.002'
            self.root.ids.vga_tap_mu_input.text = '0.0'
            self.root.ids.tr_prop_mu_input.text = '0.000025'
            tr_integ_mu = float(0.000025/512.0)
            self.root.ids.tr_integ_mu_input.text = str(tr_integ_mu)

        elif sim_mode == '3':
            self.root.ids.load_flag_input.text = 'True'
            self.root.ids.save_flag_input.text = 'True'
            self.root.ids.plot_flag_input.text = 'True'
            self.root.ids.agc_tap_mu_input.text = '0.002'
            self.root.ids.ffe_tap_mu_input.text = '0.002'
            self.root.ids.dco_tap_mu_input.text = '0.002'
            self.root.ids.dfe_tap_mu_input.text = '0.002'
            self.root.ids.lvl_tap_mu_input.text = '0.0'
            self.root.ids.phz_tap_mu_input.text = '0.0'
            self.root.ids.vga_tap_mu_input.text = '0.0'
            self.root.ids.tr_prop_mu_input.text = '0.000025'
            tr_integ_mu = float(0.000025 / 512.0)
            self.root.ids.tr_integ_mu_input.text = str(tr_integ_mu)

        elif sim_mode == '4':
            self.root.ids.load_flag_input.text = 'True'
            self.root.ids.save_flag_input.text = 'True'
            self.root.ids.plot_flag_input.text = 'True'
            self.root.ids.agc_tap_mu_input.text = '0.0'
            self.root.ids.ffe_tap_mu_input.text = '0.0'
            self.root.ids.dco_tap_mu_input.text = '0.0'
            self.root.ids.dfe_tap_mu_input.text = '0.0'
            self.root.ids.lvl_tap_mu_input.text = '0.002'
            self.root.ids.phz_tap_mu_input.text = '0.0'
            self.root.ids.vga_tap_mu_input.text = '0.0'
            self.root.ids.tr_prop_mu_input.text = '0.000025'
            tr_integ_mu = float(0.000025 / 512.0)
            self.root.ids.tr_integ_mu_input.text = str(tr_integ_mu)

        elif sim_mode == '5':
            # need to add beat frequency and tr_loop_period
            self.root.ids.load_flag_input.text = 'True'
            self.root.ids.save_flag_input.text = 'False'
            self.root.ids.plot_flag_input.text = 'False'
            self.root.ids.agc_tap_mu_input.text = '0.0'
            self.root.ids.ffe_tap_mu_input.text = '0.0'
            self.root.ids.dco_tap_mu_input.text = '0.0'
            self.root.ids.dfe_tap_mu_input.text = '0.0'
            self.root.ids.lvl_tap_mu_input.text = '0.0'
            self.root.ids.phz_tap_mu_input.text = '0.0'
            self.root.ids.vga_tap_mu_input.text = '0.0'
            self.root.ids.tr_prop_mu_input.text = '0.0'
            self.root.ids.tr_integ_mu_input.text = '0.0'
            self.root.ids.plot_skip_points_input.text = '1'
            self.root.ids.plot_window_snr_input.text = '100'

    def simulate(self):

        # create instance of parameter class
        params = Parameters()

        # safeguards for running simulation; make sure all params have safe values

        # assign params new values from gui
        params.kmax = int(self.root.ids.kmax_input.text)
        params.rx_chan_cnt = int(self.root.ids.rx_chan_cnt_input.text)
        params.rx_data_osr = int(self.root.ids.rx_data_osr_input.text)
        params.tx_data_file = self.root.ids.tx_data_file_input.text
        params.tx_data_gain = float(self.root.ids.tx_data_gain_input.text)
        params.sim_mode = int(self.root.ids.sim_mode_input.text)
        if params.sim_mode == 5:
            params.beat_frequency = 1.0 + np.float(1.0/params.kmax)
            params.tr_loop_period = np.float(params.beat_frequency*params.rx_data_osr)
        params.agc_tap_mu = float(self.root.ids.agc_tap_mu_input.text)
        params.ffe_tap_mu = float(self.root.ids.ffe_tap_mu_input.text)
        params.dfe_tap_mu = float(self.root.ids.dfe_tap_mu_input.text)
        params.dco_tap_mu = float(self.root.ids.dco_tap_mu_input.text)
        params.lvl_tap_mu = float(self.root.ids.lvl_tap_mu_input.text)
        params.vga_tap_mu = float(self.root.ids.vga_tap_mu_input.text)
        params.phz_tap_mu = float(self.root.ids.phz_tap_mu_input.text)
        params.tr_prop_mu = float(self.root.ids.tr_prop_mu_input.text)
        params.tr_integ_mu = float(self.root.ids.tr_integ_mu_input.text)

        if self.root.ids.load_flag_input.text == 'True':
            params.load_flag = True
        else:
            params.load_flag = False

        if self.root.ids.load_flag_input.text == 'True':
            params.load_flag = True
        else:
            params.load_flag = False

        if self.root.ids.load_flag_input.text == 'True':
            params.load_flag = True
        else:
            params.load_flag = False

        params.afe_rms_noise = float(self.root.ids.afe_rms_noise_input.text)
        params.afe_rms_gain = float(self.root.ids.afe_rms_gain_input.text)
        params.afe_rms_offset = float(self.root.ids.afe_rms_offset_input.text)
        params.rx_rms_phase = float(self.root.ids.rx_rms_phase_input.text)
        params.rx_rms_jitter = float(self.root.ids.rx_rms_jitter_input.text)
        params.rx_loop_demux = int(self.root.ids.rx_loop_demux_input.text)
        params.tr_sym_mode_flag = bool(self.root.ids.tr_sym_mode_flag_input.text)
        params.tr_loop_latency = int(self.root.ids.tr_loop_latency_input.text)
        params.plot_skip_points = float(self.root.ids.plot_skip_points_input.text)
        params.plot_window_snr = float(self.root.ids.plot_window_snr_input.text)

        params.ffe_tap_fix = int(self.root.ids.ffe_tap_fix_input.text)
        params.ffe_tap_cnt = int(self.root.ids.ffe_tap_cnt_input.text)
        params.ffe_tap_ref = int(self.root.ids.ffe_tap_ref_input.text)
        params.ffe_tap_pre = float(self.root.ids.ffe_tap_pre_input.text)
        params.ffe_tap_post = float(self.root.ids.ffe_tap_post_input.text)
        params.dfe_tap_cnt = int(self.root.ids.dfe_tap_cnt_input.text)
        params.dfe_tap_ref = int(self.root.ids.dfe_tap_ref_input.text)
        params.agc_tap_step = float(self.root.ids.agc_tap_step_input.text)
        params.ffe_tap_step = float(self.root.ids.ffe_tap_step_input.text)
        params.dfe_tap_step = float(self.root.ids.dfe_tap_step_input.text)
        params.dco_tap_step = float(self.root.ids.dco_tap_step_input.text)
        params.lvl_tap_Step = float(self.root.ids.lvl_tap_step_input.text)
        params.vga_tap_step = float(self.root.ids.vga_tap_step_input.text)
        params.phz_tap_step = float(self.root.ids.phz_tap_step_input.text)
        params.phz_tap_max = float(self.root.ids.phz_tap_max_input.text)

        simulate(params)

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, filename):
        self.root.ids.tx_data_file_input.text = filename[0]
        self.dismiss_popup()


if __name__ == '__main__':
    GuiApp().run()