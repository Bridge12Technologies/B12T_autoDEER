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



# rng = np.random.default_rng(12345)

hw_log = logging.getLogger('interface.B12TEpr')

# def val_in_us(Param):
#         if len(Param.axis) == 0:
#             if Param.unit == "us":
#                 return Param.value
#             elif Param.unit == "ns":
#                 return Param.value / 1e3
#         elif len(Param.axis) == 1:
#             if Param.unit == "us":
#                 return Param.tau1.value + Param.axis[0]['axis']
#             elif Param.unit == "ns":
#                 return (Param.value + Param.axis[0]['axis']) / 1e3 

# def val_in_ns(Param):
#         if len(Param.axis) == 0:
#             if Param.unit == "us":
#                 return Param.value * 1e3
#             elif Param.unit == "ns":
#                 return Param.value 
#         elif len(Param.axis) == 1:
#             if Param.unit == "us":
#                 return (Param.tau1.value + Param.axis[0]['axis']) * 1e3
#             elif Param.unit == "ns":
#                 return (Param.value + Param.axis[0]['axis']) 

# def add_noise(data, noise_level):
#     # Add noise to the data with a given noise level for data that could be either real or complex
#     if np.isrealobj(data):
#         noise = np.squeeze(rng.normal(0, noise_level, size=(*data.shape,1)).view(np.float64))

#     else:
#         noise = np.squeeze(rng.normal(0, noise_level, size=(*data.shape,2)).view(np.complex128))
#     data = data + noise
#     return data

# def add_phaseshift(data, phase):
#     data = data.astype(np.complex128) * np.exp(-1j*phase*np.pi)
#     return data

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

        # receiver amplifier
        self.rcv_state = config['Spectrometer']['RCV']['state']

        # video amplifier
        self.vva_state = config['Spectrometer']['VideoAMP']['state']
        self.vva_gain = config['Spectrometer']['VideoAMP']['gain']

        # high power amplifier
        self.hamp_state = config['Spectrometer']['HP_AMP']['state']
        
        self.saving_path = 'C:/SpecMan4EPRData/buffer/'
        super().__init__(log=hw_log)
         
    def connect(self):
        self._connect_epr()
        self.send_message(".server.opmode = 'Operate'")
        self.send_message(".spec.BRIDGE.Frequency = %0.03f" %(self.fc))
        self.send_message(".spec.BRIDGE.RecvAmp = %i" %(self.rcv_state))
        self.send_message(".spec.BRIDGE.VideoAmp = %i" %(self.vva_state))
        self.send_message(".spec.BRIDGE.VideoGain = %i" %(self.vva_gain))
        self.send_message(".spec.BRIDGE.IO2 = %i" %(self.hamp_state))
        return super().connect()
        
    def launch(self, sequence, savename: str, **kwargs):
        hw_log.info(f"Launching {sequence.name} sequence")
        self.state = True
        self.cur_exp = sequence
        self.start_time = time.time()
        self.fguid = self.cur_exp.name + "_" +  datetime.datetime.now().strftime('%Y%m%d_%H%M_')
        
        if isinstance(self.cur_exp, FieldSweepSequence):
            self._run_fsweep(self.cur_exp)
        elif isinstance(self.cur_exp, ReptimeScan):
            self._run_reptimescan(self.cur_exp)
        elif isinstance(self.cur_exp, ResonatorProfileSequence):
            self._run_respro(self.cur_exp)
        elif isinstance(self.cur_exp, DEERSequence):
            self._run_CP_relax(self.cur_exp)
        elif isinstance(self.cur_exp, T2RelaxationSequence):
            self._run_T2_relax(self.cur_exp)
        elif isinstance(self.cur_exp, DEERSequence):
            self._run_deer(self.cur_exp)

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
        reptime = sequence.reptime.value
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
        self.send_message(".server.open = 'AutoDEER/AD_ShotRepetitionTime.tpl'")
        self.send_message(".spec.BRIDGE.Frequency = %0.03f" %LO)
        self.send_message(f".exp.expaxes.P.Sweep.string = '{sequence.B.value:0.04f} G'")
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2)
    
    def _run_respro(self, sequence: ResonatorProfileSequence):
        reptime = sequence.reptime.value
        self.send_message(".server.open = 'AutoDEER/AD_ResonatorProfile.tpl'")
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
        reptime = sequence.reptime.value
        self.send_message(".server.open = 'AutoDEER/AD_CarrPurcell.tpl'")
        self.send_message(f".exp.expaxes.P.Sweep.string = '{sequence.B.value:0.04f} G'")
        self.send_message(".exp.expaxes.P.RepTime.string = %s us" %reptime)
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2) 

    def _run_T2_relax(self, sequence: T2RelaxationSequence):
        LO = sequence.LO.value * 1e9
        reptime = sequence.reptime.value
        self.send_message(".server.open = 'AutoDEER/AD_HahnEchoDecay.tpl'")
        self.send_message(f".exp.expaxes.P.Sweep.string = '{sequence.B.value:0.04f} G'")
        self.send_message(".exp.expaxes.P.RepTime.string = %s us" %reptime)
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2) 
    
    def _run_2D_relax(self, sequence: RefocusedEcho2DSequence):
        LO = sequence.LO.value * 1e9
        self.send_message(".server.open = 'AutoDEER/AD_2DRefocusedEcho.tpl'")
        self.send_message(f".exp.expaxes.P.Sweep.string = '{sequence.B.value:0.04f} G'")
        # self.send_message(".exp.expaxes.P.RepTime.string = %s us" %reptime)
        self.send_message(".server.COMPILE")
        self.send_message(f".daemon.fguid = '{self.fguid}'")
        if '0' in self.send_message(".daemon.state"):
            self.send_message(".daemon.stop")
        self.send_message(".daemon.run")
        while self.isrunning():
            time.sleep(0.2) 
    
    def _run_deer(self, sequence: DEERSequence):
        LO = sequence.LO.value * 1e9
        self.send_message(".server.open = 'AutoDEER/AD_4PulseDEER.tpl'")
        self.send_message(f".exp.expaxes.P.tau1.string = '{sequence.tau1.value:0.04f} us'")
        self.send_message(f".exp.expaxes.P.tau2.string = '{sequence.tau2.value:0.04f} us'")
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
        elif isinstance(self.cur_exp, (DEERSequence, RefocusedEcho2DSequence)):
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

def _run_field_sweep(sequence):
    # Assuming a Nitroxide sample
    Vmodel = create_Nmodel(sequence.LO.value *1e3)

    axis = sequence.B.value + sequence.B.axis[0]['axis']
    sim_axis = axis * 0.1
    Boffset=0
    gy = 2.0061
    gz = 2.0021
    axy = 0.488
    az = 3.66
    GB = 0.45
    scale=1

    data = Vmodel(sim_axis,Boffset,gy,gz,axy,az,GB,scale)
    data = add_phaseshift(data, 0.05)
    return axis,data
    pass

def _simulate_deer(sequence,exp_type=None):

    # if sequence.name == "4pDEER":
    #     exp_type = "4pDEER"
    #     tau1 = val_in_us(sequence.tau1)
    #     tau2 = val_in_us(sequence.tau2)
    #     t = val_in_us(sequence.t)
    # elif sequence.name == "5pDEER":
    #     exp_type = "5pDEER"
    #     tau1 = val_in_us(sequence.tau1)
    #     tau2 = val_in_us(sequence.tau2)
    #     tau3 = val_in_us(sequence.tau3)
    #     t = val_in_us(sequence.t)
    # elif sequence.name == "3pDEER":
    #     exp_type = "3pDEER"
    #     tau1 = val_in_us(sequence.tau1)
    #     t = val_in_us(sequence.t)
    # elif sequence.name == "nDEER-CP":
    #     exp_type = "4pDEER"
    #     tau1 = val_in_us(sequence.tau1)
    #     tau2 = val_in_us(sequence.tau2)
    #     t = val_in_us(sequence.t)

    # if exp_type == "4pDEER":
    #     experimentInfo = dl.ex_4pdeer(tau1=tau1,tau2=tau2,pathways=[1,2,3])
    #     reftimes = dict(zip(["reftime1","reftime2","reftime3"],experimentInfo.reftimes(tau1,tau2)))
    #     mod_depths = {"lam1":0.4, "lam2":0.1, "lam3":0.2}
    # elif exp_type == "5pDEER":
    #     experimentInfo = dl.ex_fwd5pdeer(tau1=tau1,tau2=tau2,tau3=tau3,pathways=[1,2,3,4,5])
    #     reftimes = dict(zip(["reftime1","reftime2","reftime3","reftime4","reftime5"],experimentInfo.reftimes(tau1,tau2,tau3)))
    #     mod_depths = {"lam1":0.4, "lam2":0.00, "lam3":0.0, "lam4":0.00, "lam5":0.1}

    # elif exp_type == "3pDEER":
    #     experimentInfo = dl.ex_3pdeer(tau=tau1,pathways=[1,2])
    #     reftimes = dict(zip(["reftime1","reftime2"],experimentInfo.reftimes(tau1,)))
    #     mod_depths = {"lam1":0.6, "lam2":0.1}


    # r = np.linspace(0.5,10,100)
    # rmean = 4.5
    # rstd = 1.0
    # conc = 50

    # Vmodel = dl.dipolarmodel(t,r,Pmodel=dl.dd_gauss, experiment=experimentInfo)
    # Vsim = Vmodel(mean=rmean, std=rstd, conc=conc, scale=1, **reftimes, **mod_depths)
    # # Add phase shift
    # Vsim = add_phaseshift(Vsim, 0.05)
    # return t, Vsim
    pass

def _simulate_CP(sequence):

    # if isinstance(sequence, DEERSequence):
    #     xaxis = val_in_ns(sequence.tau2)
    # elif isinstance(sequence, CarrPurcellSequence):
    #     xaxis = val_in_ns(sequence.step)

    # func = lambda x, a, b, e: a*np.exp(-b*x**e)
    
    # data = func(xaxis,1,2e-6,1.8)
    # data = add_phaseshift(data, 0.05)
    # return xaxis, data
    pass

def _simulate_T2(sequence,ESEEM_depth):
    # func = lambda x, a, b, e: a*np.exp(-b*x**e)
    # xaxis = val_in_ns(sequence.tau)
    # data = func(xaxis,1,10e-6,1.6)
    # data = add_phaseshift(data, 0.05)
    # if ESEEM_depth != 0:
    #     data *= _gen_ESEEM(xaxis, 7.842, ESEEM_depth)
    # return xaxis, data
    pass

def _similate_respro(sequence, mode):

    # damped_oscilations = lambda x, f, c: np.cos(2*np.pi*f*x) * np.exp(-c*x)
    # damped_oscilations_vec = np.vectorize(damped_oscilations)
    # LO_axis = sequence.LO.value + sequence.LO.axis[0]['axis']
    # LO_len = LO_axis.shape[0]
    # tp_x = val_in_ns(sequence.pulses[0].tp)
    # tp_len = tp_x.shape[0]
    # nut_freqs = mode(LO_axis)

    # damped_oscilations_vec
    # data = damped_oscilations_vec(tp_x.reshape(tp_len,1),nut_freqs.reshape(1,LO_len)*1e-3,0.06)
    # return [tp_x, LO_axis], data
    pass

def _simulate_reptimescan(sequence):
    # def func(x,T1):
    #     return 1-np.exp(-x/T1)
    # t = sequence.reptime.value + sequence.reptime.axis[0]['axis']
    # T1 = 2000 #us

    # data = func(t,T1)
    # data = add_phaseshift(data, 0.05)
    # return t, data
    pass

def _simulate_2D_relax(sequence):
    # sigma = 0.8
    # func = lambda x, y: np.exp(-((x**2 + y**2 - 1*x*y) / (2*sigma**2)))

    # xaxis = val_in_us(sequence.tau1)
    # yaxis = val_in_us(sequence.tau2)

    # X, Y = np.meshgrid(xaxis, yaxis)
    # data = func(X, Y)
    # data = add_phaseshift(data, 0.05)
    # return [xaxis, yaxis], data
    pass

        

def _gen_ESEEM(t,freq,depth):
    # # Generate an ESEEM modulation
    # modulation = np.ones_like(t,dtype=np.float64)
    # modulation -= depth *(0.5 + 0.5*np.cos(2*np.pi*t*freq)) + depth * (0.5+0.5*np.cos(2*np.pi*t*freq/2))
    # return modulation
    pass

