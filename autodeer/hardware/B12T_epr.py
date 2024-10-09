from autodeer.classes import  Interface, Parameter
from autodeer.pulses import Pulse, RectPulse, ChirpPulse, HSPulse, Delay, Detection
from autodeer.sequences import *
from autodeer.FieldSweep import create_Nmodel
from autodeer.dataset import get_all_fixed_param
import yaml

import xarray as xr
import datetime
import numpy as np
import deerlab as dl
import time
import socket
import logging

import dnplab as dnp


hw_log = logging.getLogger('interface.B12TEpr')


class B12TInterface(Interface):

    def __init__(self,config_file) -> None:
        with open(config_file, mode='r') as file:
            config = yaml.safe_load(file)
            self.config = config    

        # Dummy = config['Spectrometer']['Dummy']
        Bridge = config['Spectrometer']['Bridge']
        resonator_list = list(config['Resonators'].keys())
        key1 = resonator_list[0]
        self.fc = self.config['Resonators'][key1]['Center Freq'] * 1e9 # Hz
        self.Q = self.config['Resonators'][key1]['Q']
        self.resonator_bandwidth = self.config['Resonators'][key1]['bandwidth'] * 1e6 # Hz

        # AWG:
        self.awg_state = config['Spectrometer']['AWG']['state']
        self.awg_freq = config['Spectrometer']['AWG']['frequency'] * 1e6 # Hz

        # receiver amplifier
        self.rcv_state = config['Spectrometer']['RCV']['state']

        # video amplifier
        self.vva_state = config['Spectrometer']['VideoAMP']['state']
        self.vva_gain = config['Spectrometer']['VideoAMP']['gain']

        # high power amplifier
        self.hamp_state = config['Spectrometer']['HP_AMP']['state']

        # initial field
        self.start_field = config['Spectrometer']['Field']['StartField'] * 1e3 # G
        
        self.saving_path = 'C:/SpecMan4EPRData/buffer/'

        # check cp run
        self.cp_ran = False

        # relaxataion time
        self.guess_tau1 = config['initial_params']['guess_tau1'] * 1e-9
        self.guess_tau2 = config['initial_params']['guess_tau2'] * 1e-9
        self.tau1 = self.guess_tau1 if config['initial_params']['fix_tau1'] else None # s
        self.tau2 = self.guess_tau1 if config['initial_params']['fix_tau2'] else None # s

        # pulse
        self.t90 = config['initial_params']['t90'] * 1e-9
        self.t180 = config['initial_params']['t180'] * 1e-9

        # DEER X Axis
        self.t_start = config['initial_params']['t_start'] * 1e-9
        self.t_step = config['initial_params']['t_step'] * 1e-9
        self.t_size = config['initial_params']['t_size'] 
        self.t_stop = self.t_start + (self.t_size - 1) * self.t_step

        super().__init__(log=hw_log)
         
    def connect(self):
        self._connect_epr()
        self.send_message(".server.opmode = 'Operate'")
        self.send_message(".spec.BRIDGE.Frequency = %0.03f" %(self.fc))
        self.send_message(".spec.BRIDGE.RecvAmp = %i" %(self.rcv_state))
        self.send_message(".spec.BRIDGE.VideoAmp = %i" %(self.vva_state))
        self.send_message(".spec.BRIDGE.VideoGain = %i" %(self.vva_gain))
        self.send_message(".spec.BRIDGE.IO2 = %i" %(self.hamp_state))
        self.send_message(".spec.FLD.StartField = %i" %(self.start_field))

        return super().connect()
        
    def launch(self, sequence, savename: str, **kwargs):
        hw_log.info(f"Launching {sequence.name} sequence")
        self.state = True
        self.cur_exp = sequence
        self.start_time = time.time()
        self.fguid = self.cur_exp.name + "_" +  datetime.datetime.now().strftime('%Y%m%d_%H%M_')
        # self.fguid = self.cur_exp.name
        if isinstance(self.cur_exp, FieldSweepSequence):
            self._run_fsweep(self.cur_exp)
            # self.fguid = self.cur_exp.name
            pass
        elif isinstance(self.cur_exp, ReptimeScan):
            self._run_reptimescan(self.cur_exp)
            # self.fguid = self.cur_exp.name
            pass
        elif isinstance(self.cur_exp, ResonatorProfileSequence):
            self._run_respro(self.cur_exp)
            # self.fguid = self.cur_exp.name
            pass
        elif isinstance(self.cur_exp, DEERSequence):
            if self.cp_ran:
                self._run_deer(self.cur_exp)
            else:
                self._run_CP_relax(self.cur_exp)
                
        elif isinstance(self.cur_exp, T2RelaxationSequence):
            self._run_T2_relax(self.cur_exp)

        elif isinstance(self.cur_exp, RefocusedEcho2DSequence):
            self._run_2D_relax(self.cur_exp)
        
        # elif isinstance(self.cur_exp, DEERSequence):
        #     self._run_deer(self.cur_exp)

        return super().launch(sequence, savename)
    
    def acquire_dataset(self,**kwargs):
        hw_log.debug("Acquiring dataset")

        dset = self._create_dataset_from_b12t(self.saving_path + self.fguid + '.exp')
        
        return super().acquire_dataset(dset)
    
    def tune_rectpulse(self,*,tp, LO, B, reptime,**kwargs):

        # rabi_freq = self.mode(LO)
        # def Hz2length(x):
        #     return 1 / ((x/1000)*2)
        # rabi_time = Hz2length(rabi_freq)
        # if rabi_time > tp:
        #     p90 = tp
        #     p180 = tp*2
        # else:
        #     p90 = rabi_time/tp
        #     p180 = p90*2

        self.pulses[f"p90_{tp}"] = RectPulse(tp=16)
        self.pulses[f"p180_{tp}"] = RectPulse(tp=32)

        return self.pulses[f"p90_{tp}"], self.pulses[f"p180_{tp}"]
    

    def tune_pulse(self, pulse, mode, LO, B , reptime, shots=400):
        # hw_log.debug(f"Tuning {pulse.name} pulse")
        # pulse.scale = Parameter('scale',0.5,unit=None,description='The amplitude of the pulse 0-1')
        # hw_log.debug(f"Setting {pulse.name} pulse to {pulse.scale.value}")
        # return pulse
        pass
            
    def isrunning(self) -> bool:
        '''
        4 - idle
        6 - run 
        '''
        if '6' in self.send_message(".daemon.state"):
            return True
        else:
            return False

    def terminate(self) -> None:
        self.state = False
        hw_log.info("Terminating sequence")
        return super().terminate()
    
    
    def _run_fsweep(self, sequence: FieldSweepSequence):
        LO = sequence.LO.value * 1e9
        # reptime = sequence.reptime.value
        reptime = 100
        print('fsweep', LO)
        self.send_message(".server.open = 'AutoDEER/AD_Fieldsweep.tpl'")
        self.send_message(".spec.BRIDGE.Frequency = %0.03f" %LO)
        self.send_message(".exp.expaxes.P.RepTime.string = '%s us'" %reptime)
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2)
    
    def _run_reptimescan(self, sequence: ReptimeScan):
        LO = sequence.LO.value * 1e9
        reptime = 100
        sweep = sequence.B.value - self.start_field
        self.send_message(".server.open = 'AutoDEER/AD_ShotRepetitionTime.tpl'")
        self.send_message(".spec.BRIDGE.Frequency = %0.03f" %LO)
        self.send_message(".exp.expaxes.P.RepTime.string = '%s us'" %reptime)
        self.send_message(f".spec.FLD.Sweep = {sweep:0.04f}")
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2)
    
    def _run_respro(self, sequence: ResonatorProfileSequence):
        # reptime = sequence.reptime.value
        reptime = 100
        sweep = sequence.B.value - self.start_field
        print('RES', sweep)
        self.gyro = sequence.gyro
        LO, Sweep = self._calculate_nutation_bandwidth(self.fc, self.awg_freq, self.start_field, sweep, self.resonator_bandwidth)
        self.send_message(".server.open = 'AutoDEER/AD_ResonatorProfile.tpl'")
        self.send_message(".exp.expaxes.Y.LO.string = '%s'" %LO)
        self.send_message(".exp.expaxes.Y.Sweep.string = '%s'" %Sweep)
        self.send_message(f".spec.FLD.Sweep = {sweep:0.04f}")
        self.send_message(".exp.expaxes.P.RepTime.string = '%s us'" %reptime)
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2) 

    def _run_CP_relax(self, sequence: DEERSequence):
        LO = sequence.LO.value * 1e9
        # reptime = sequence.reptime.value
        reptime = 100
        sweep = sequence.B.value - self.start_field
        self.send_message(".server.open = 'AutoDEER/AD_CarrPurcell.tpl'")
        self.send_message(".spec.BRIDGE.Frequency = %0.03f" %LO)
        self.send_message(f".spec.FLD.Sweep = {sweep:0.04f}")
        self.send_message(".exp.expaxes.P.RepTime.string = '%s us'" %reptime)
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2) 

    def _run_T2_relax(self, sequence: T2RelaxationSequence):
        LO = sequence.LO.value * 1e9
        # reptime = sequence.reptime.value
        reptime = 100
        sweep = sequence.B.value - self.start_field
        self.send_message(".server.open = 'AutoDEER/AD_HahnEchoDecay.tpl'")
        self.send_message(".spec.BRIDGE.Frequency = %0.03f" %LO)
        self.send_message(f".spec.FLD.Sweep = {sweep:0.04f}")
        self.send_message(".exp.expaxes.P.RepTime.string = '%s us'" %reptime)
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2) 
    
    def _run_2D_relax(self, sequence: RefocusedEcho2DSequence):
        LO = sequence.LO.value * 1e9
        sweep = sequence.B.value - self.start_field
        self.send_message(".server.open = 'AutoDEER/AD_2DRefocusedEcho.tpl'")
        self.send_message(".spec.BRIDGE.Frequency = %0.03f" %LO)
        self.send_message(f".spec.FLD.Sweep = {sweep:0.04f}")
        # self.send_message(".exp.expaxes.P.RepTime.string = %s us" %reptime)
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2) 
    
    def _run_deer(self, sequence: DEERSequence):

        # force tau1 and tau2 to a value
        if self.tau1:
            sequence.tau1.value = self.tau1 * 1e9
        
        if self.tau2:
            sequence.tau2.value = self.tau2 * 1e9


        exc_pulse_freq = sequence.exc_pulse.freq.value * 1e9 # Hz
        LO = sequence.LO.value * 1e9 + exc_pulse_freq # Hz

        sweep = sequence.B.value - self.start_field
        tau1 = sequence.tau1.value
        tau2 = sequence.tau2.value
        print('run_deer', tau1, tau2)
        self.send_message(".server.open = 'AutoDEER/AD_4PulseDEER.tpl'")
        self.send_message(".spec.BRIDGE.Frequency = %0.03f" %LO)
        self.send_message(f".exp.expaxes.P.tau1.string = '{self.tau1} s'")
        self.send_message(f".exp.expaxes.P.tau2.string = '{self.tau2} s'")
        self.send_message(f".exp.expaxes.P.t90.string = '{self.t90} s'")
        self.send_message(f".exp.expaxes.P.t180.string = '{self.t180} s'")
        self.send_message(f".spec.FLD.Sweep = {sweep:0.04f}")
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2) 
    
    def _connect_epr(self, address: str = 'localhost' , port_number: int = 8023) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (address, port_number)
        self.sock.connect(server_address)
        self.send_message(".server.message='SpecMan is controlled'")

    def send_message(self, message, convert:bool = True, raw:bool = False):
        if message[-1] != '\n':
            message += '\n'
        self.sock.send(message.encode('utf-8'))
        time.sleep(0.2)
        data = self.sock.recv(4096)

        if convert: 
            return data.decode('utf-8')#.replace("sm>"+message.replace("\n", "")+"=", "")
        return data
    
    def __del__(self):
        self.send_message(".server.opmode = 'Standby'")
        self.sock.close()

    def _create_dataset_from_b12t(self, filepath):
        data = dnp.load(filepath, autodetect_coords = True, autodetect_dims = True)        
        data_real = data['x',0].sum('x')
        data_imag = data['x',1].sum('x')
        data = data_real + 1j*data_imag

        if isinstance(self.cur_exp, RefocusedEcho2DSequence):
            self.tau2 = self._calculate_tau2(data)

        for dim in data.dims:
            if dim + '_unit' in data.attrs and data.attrs[dim + '_unit'] == 'T':
                data.coords[dim] *= 1e4 # convert unit to Gauss
                data.attrs[dim + '_unit'] = 'G'

        coord_factors = []
        default_labels = ['X','Y','Z','T']
        if isinstance(self.cur_exp, ReptimeScan):
            dims = ['reptime']
            coord_factors = [1e6]
        elif isinstance(self.cur_exp, ResonatorProfileSequence):
            dims = ['pulse0_tp', 'LO']
            coord_factors = [1e9, 1e-9]
        elif isinstance(self.cur_exp, (DEERSequence)):
            if not self.cp_ran:
                dims = ['tau1', 'tau2', 'tau3', 'tau4']
                self.cp_ran = True
            else:
                dims = ['t', 't2', 't3', 't4']
            coord_factors = [1e6, 1e6, 1e6, 1e6]
            dims = dims[:len(data.dims)]
            coord_factors = coord_factors[:len(data.dims)]
        elif isinstance(self.cur_exp, (RefocusedEcho2DSequence)):
            dims = ['tau1', 'tau2', 'tau3', 'tau4']
            coord_factors = [1e6, 1e6, 1e6, 1e6]
            dims = dims[:len(data.dims)]
            coord_factors = coord_factors[:len(data.dims)]
        elif isinstance(self.cur_exp, (T2RelaxationSequence)):
            coord_factors = [1e6, 1e6, 1e6, 1e6]
            dims = default_labels[:len(data.dims)]
            coord_factors = coord_factors[:len(data.dims)]
        else:
            dims = default_labels[:len(data.dims)]
        
        if isinstance(self.cur_exp, FieldSweepSequence):
            data.coords['Sweep'] += self.start_field
        
        coords = data.coords.coords
        if coord_factors:
            for index, dim in enumerate(dims):
                coords[index] *= coord_factors[index]

        attrs = data.attrs
        attrs['LO'] = float(attrs['BRIDGE_Frequency'].replace(' GHz', ''))
        attrs['shots'] = int(attrs['System_Shots']) if int(attrs['System_Shots']) != 0 else 1
        attrs['nAvgs'] = eval(attrs['streams_scans'])[0]
        attrs['nPcyc'] = int(attrs['idx']) if 'idx' in attrs else 1
        attrs['reptime'] = attrs['RepTime']
        attrs ={**attrs, **get_all_fixed_param(self.cur_exp)}
        return xr.DataArray(data.values, dims=dims, coords=coords, attrs=attrs)
    
    def _calculate_nutation_bandwidth(self, freq_bridge, freq_awg, field_start, field_sweep, freq_width):
        '''
        Calcaute Parameters for nutation bandwidth experiment
        Args: 
            freq_bridge: bridge frequency in Hz
            freq_awg: modulation frequency in Hz
            field_start: the start field in G
            field_sweep: the sweep field added to start field to get center field
            freq_width: the bandwidth of resonator in Hz

        Returns:
            LO, Sweep strings for nutation bandwidth experiment
        
        '''

        # convert to GHz
        freq_bridge /= 1e9 
        freq_awg /= 1e9
        freq_width /= 1e9
        
        freq = freq_bridge + freq_awg
        field = field_sweep + field_start

        field_to_freq = field / freq

        field_sweep_low = (freq_bridge + freq_awg - freq_width/2.) * field_to_freq - field_start
        field_sweep_high = (freq_bridge + freq_awg + freq_width/2.) * field_to_freq - field_start

        freq_sweep_low = freq_bridge - freq_width/2.
        freq_sweep_high = freq_bridge + freq_width/2.

        LO = '%0.04f GHz to %0.04f GHz'%(freq_sweep_low, freq_sweep_high)
        Sweep = '%0.04f G to %0.04f G'%(field_sweep_low,field_sweep_high)

        return LO, Sweep
    
    def _calculate_observe_field(self, B, g, observe_freq):
        return B + observe_freq/g
    
    def _calculate_tau2(self, data):
        tau2s = []
        tau2_errs = []
        for t in data.coords['tau1']:
            temp = data['tau1', t].sum('tau1').abs
            temp /= np.max(temp)
            try:
                fit = dnp.fit(dnp.relaxation.t2, temp, dim = 'tau2', p0 = (1.,1.0e-6,1))
                tau2s.append(fit['popt'].values[1])
                tau2_errs.append(fit['err'].values[1])
                if len(tau2s) == 5:
                    break
            except:
                tau2s.append(99)
                tau2_errs.append(99)
                continue
        
        if all([x==99 for x in tau2s]):
            print('cannot find tau2')
            return None
        
        tau2s_diff_from_guess = [np.abs(x - self.guess_tau2) for x in tau2s] # find the closest value to initial guess
        tau2 = tau2s[np.argmin(tau2s_diff_from_guess)]
        if tau2 <= self.t_stop + self.t180:
            tau2 = self.t_stop + self.t180 + 1e-9 # prevent overlapped pulse
        return tau2
        


