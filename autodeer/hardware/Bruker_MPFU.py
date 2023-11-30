from autodeer.classes import Interface, Parameter
from autodeer.pulses import Delay, Detection, RectPulse
from autodeer.hardware.XeprAPI_link import XeprAPILink
from autodeer.hardware.Bruker_tools import PulseSpel, run_general,build_unique_progtable,PSPhaseCycle, write_pulsespel_file
from autodeer.sequences import Sequence, HahnEchoSequence
from autodeer.utils import save_file, transpose_list_of_dicts, transpose_dict_of_list
from autodeer import create_dataset_from_sequence

import tempfile
import time
from scipy.optimize import minimize_scalar, curve_fit
import numpy as np
import threading
import concurrent.futures
from PyQt6.QtCore import QThreadPool
from autodeer.gui import Worker
import os
from pathlib import Path
import datetime
from deerlab import correctphase
# =============================================================================


class BrukerMPFU(Interface):
    """
    Represents the interface for connecting to MPFU based Bruker ELEXSYS-II 
    Spectrometers.
    """
    def __init__(self, config_file:str, d0=650) -> None:
        """An interface for connecting to MPFU based Bruker ELEXSYS-II 
        Spectrometers.

        Getting Started
        ------------------
        Before a connection can be made an appropriate configuration file first
        needs to be written. 

        1. Open Xepr
        2. Processing -> XeprAPI -> Enable XeprAPI
        3. `BrukerAWG.connect()`

        Parameters
        ----------
        config_file : str
            _description_
        d0 : int, optional
            _description_, by default 600

        Attributes
        ----------
        temp_dir:str
            A linux tmp directory for storing pulsespel files

        bg_thread: None or threading.Thread
            If a background thread is needed, it is stored here.
        
        """

        self.api = XeprAPILink(config_file)
        self.spec_config = self.api.config["Spectrometer"]
        self.bridge_config = self.api.spec_config["Bridge"]

        self.MPFU = self.bridge_config["MPFU Channels"]
        
        self.temp_dir = tempfile.mkdtemp("autoDEER")

        self.d0 = self.bridge_config["d0"]

        self.bg_thread = None
        self.bg_data = None
        self.cur_exp = None
        self.tuning = False
        self.pool = QThreadPool()
        self.savename = ''
        self.savefolder = str(Path.home())
        
        super().__init__()


    def connect(self) -> None:

        self.api.connect()
        return super().connect()

    def acquire_dataset(self):
        if self.bg_data is None:
            if not self.isrunning():
                self.api.xepr_save(os.path.join(self.savefolder,self.savename))
                data = self.api.acquire_dataset(self.cur_exp)
            else:
                data = self.api.acquire_scan(self.cur_exp)
        else:

            data = create_dataset_from_sequence(self.bg_data, self.cur_exp)
        return super().acquire_dataset(data)
    
    def _launch_complex_thread(self,sequence,axID=1,tune=True):
    
        uProgTable = build_unique_progtable(sequence)
        reduced_seq = sequence.copy()
        reduced_seq.progTable = transpose_list_of_dicts([transpose_dict_of_list(sequence.progTable)[0]])

        uProgTable_py = uProgTable[axID]
        axis = uProgTable_py['axis']
        reduced_seq.averages.value = 1
        py_ax_dim = uProgTable_py['axis']['dim']

        
        self.bg_data = np.zeros((uProgTable[0]['axis']['dim'],py_ax_dim),dtype=np.complex128)
        
        print("Initial PulseSpel Launch")
        self.launch(reduced_seq,savename='test',tune=tune, update_pulsespel=True, start=False,reset_bg_data=False,reset_cur_exp=False)

        self.terminate()

        variables = uProgTable_py['variables']
        # print("Creating Thread")
        # # thread = threading.Thread(target=step_parameters,args=[self,reduced_seq,py_ax_dim,variables])
        # # self.bg_thread = self.pool.submit(step_parameters, self,reduced_seq,py_ax_dim,variables)
        # self.bg_thread = Worker(step_parameters, self,reduced_seq,py_ax_dim,variables)
        # self.pool.start(self.bg_thread)
        # print("Started Thread")

        step_parameters(self,reduced_seq,py_ax_dim,variables)
        # thread.start()

        pass
        
    def launch(self, sequence: Sequence, savename: str, start=True, tune=True,
               MPFU_overwrite=None,update_pulsespel=True, reset_bg_data = True,
               reset_cur_exp=True,**kwargs):
        
        sequence.shift_detfreq_to_zero()

        if self.isrunning():
            self.terminate()
            time.sleep(2)
        
        if reset_bg_data:
            self.bg_data = None

        if reset_cur_exp:
            timestamp = datetime.datetime.now().strftime(r'%Y%m%d_%H%M_')
            self.savename = timestamp+savename
            self.cur_exp = sequence
                    
        # First check if the sequence is pulsespel compatible
    
        if not test_if_MPFU_compatability(sequence):
            print("Launching complex sequence")
            self._launch_complex_thread(sequence,1,tune)
            return None
        
        

        channels = _MPFU_channels(sequence)
        # pcyc = PSPhaseCycle(sequence,self.MPFU)

        N_channels = len(channels)

        if N_channels > len(self.MPFU):
            raise RuntimeError(
                f"This sequence requires {N_channels} MPFU" "Channels." 
                "Only {len(self.MPFU)} are avaliable on this spectrometer.")
        
        if tune:
            self.tuning = True
            if 'ELDOR' in channels:
                dif_freq=None
                for pulse in sequence.pulses:
                    if pulse.freq.value == 0:
                        continue
                    elif dif_freq is None:
                        dif_freq = pulse.freq.value
                    elif pulse.freq.value != dif_freq:
                        raise ValueError('Only one ELDOR frequency is possible')

                ELDORtune(self,sequence,freq=dif_freq)
            MPFUtune(self,sequence, channels)
            self.tuning = False

        if MPFU_overwrite is None:
            MPFU_chans = self.MPFU
        else:
            MPFU_chans = MPFU_overwrite

        PSpel_file = self.temp_dir + "/autoDEER_PulseSpel"
        if update_pulsespel:
            # PSpel = PulseSpel(sequence, MPFU=self.MPFU)
            
            def_file, exp_file = write_pulsespel_file(sequence,False,MPFU_chans)
            
            # PSpel.save(PSpel_file)
            try:
                os.remove(PSpel_file +'.def')
            except OSError:
                pass
            try:
                os.remove(PSpel_file +'.exp')
            except OSError:
                pass
            save_file(PSpel_file +'.def',def_file)
            save_file(PSpel_file +'.exp',exp_file)
        
        
        self.api.set_field(sequence.B.value)
        self.api.set_freq(sequence.LO.value)
        
        if 'B' in sequence.progTable['Variable']:
            idx = sequence.progTable['Variable'].index('B')
            B_axis = sequence.progTable['axis'][idx]
            self.api.set_sweep_width(B_axis.max()-B_axis.min())
        
        run_general(self.api,
            ps_file= [PSpel_file],
            exp=("auto","auto"),
            settings={"ReplaceMode": False},
            variables={"d0": self.d0},
            run=False
        )
        if start:
            self.api.run_exp()
        pass

    def tune_rectpulse(self,*, tp,**kwargs):
        """Mocks the tune_rectpulse command and returns a pair of RectPulses
        with the given tp and 2*tp respectively. No scale is set."""
        p90 = RectPulse(tp=tp,freq=0,t=0,flipangle=np.pi/2)
        p180 = RectPulse(tp=tp*2,freq=0,t=0,flipangle=np.pi)

        self.pulses[f"p90_{tp}"] = p90
        self.pulses[f"p180_{tp*2}"] = p180

        return self.pulses[f"p90_{tp}"], self.pulses[f"p180_{tp*2}"]
    
    def tune_pulse(self, pulse, *args, **kwargs):
        """Mocks the tune_pulse command and returns the pulse unchanged.
        """
        return pulse
    

    def tune(self, sequence, B0, LO) -> None:
        channels = _MPFU_channels(sequence)
        
        for i,channel in enumerate(channels):
            ps_length = int(np.pi /(channel[0]*2))
            phase = channel[1]
            if phase == 0:
                echo = "R+"
            elif phase == np.pi/2:
                echo = "I+"
            elif phase == np.pi:
                echo = "R-"
            elif (phase == -np.pi/2) or (phase == 3*np.pi/2):
                echo = "I-"
            mpfu_tune = MPFUtune(self.api,B0=B0,LO=LO, echo="Hahn",ps_length=ps_length)
            mpfu_tune.tune({self.MPFU[i]: echo})
        
        pass

    def isrunning(self) -> bool:
        if self.tuning:
            return True
        if self.bg_thread is None:
            return self.api.is_exp_running()
        else:
            return self.bg_thread.running()

    def terminate(self,now=False) -> None:
        self.tuning = False
        if self.bg_thread is None:
            if now:
                return self.api.abort_exp()
            else:
                return self.api.stop_exp()
        else:
            attempt = self.bg_thread.cancel()
            if not attempt:
                raise RuntimeError("Thread failed to be canceled!")
        
    

# =============================================================================
def step_parameters(interface, reduced_seq, dim, variables):
    
    for i in range(dim):
        new_seq  =reduced_seq.copy()
        # Change all variables in the sequence
        for var in variables:
            if var['variable'][0] is not None:
                raise ValueError('Only exp parameters are supported at the moment')
            attr = getattr(new_seq,var['variable'][1]) 
            shift = attr.get_axis()[i]
            attr.value = (shift)
            setattr(new_seq,var['variable'][1],attr)
            print(f"{var['variable'][1]}: {getattr(new_seq,var['variable'][1]).value} ")

        # self.launch(new_seq,savename='test',tune=False, update_pulsespel=False)

        interface.api.set_field(new_seq.B.value)
        interface.api.set_freq(new_seq.LO.value)

        interface.api.run_exp()

        while interface.api.is_exp_running():
            time.sleep(1)
        single_scan_data = interface.api.acquire_dataset()
        interface.bg_data[:,i] += single_scan_data.data


def _MPFU_channels(sequence):
    """Idenitifies how many unique MPFU channels are needed for a sequence and
    applies the correct Channel infomation to each pulse.
    """
    channels = []

    for iD, pulse in enumerate(sequence.pulses):
        if type(pulse) is Delay:
            continue
        if type(pulse) is Detection:
            continue
        if ('Channels' in pulse.pcyc) and (pulse.pcyc['Channels'] == 'ELDOR'):
            continue

        if pulse.tp.value == 0:
            flip_power = np.inf
        else:
            flip_power = pulse.flipangle.value / pulse.tp.value

        if (pulse.freq.value != 0):
            # This is an ELDOR pulse
            pulse.pcyc["Channels"] = "ELDOR"
            if "ELDOR" not in channels:
                channels.append("ELDOR")
            continue

        if not "Channels" in pulse.pcyc:
            pulse.pcyc["Channels"] = []
        elif pulse.pcyc is None:
            pulse.pcyc = {}
            pulse.pcyc["Channels"] = []
        
        
        for phase in pulse.pcyc["Phases"]:
            power_phase = (flip_power, phase)
            if power_phase in channels:
                channel = channels.index(power_phase)
            else:
                channels.append(power_phase)
                channel = channels.index(power_phase)
            pulse.pcyc["Channels"].append(channel)
    
    return channels

# =============================================================================


def get_specjet_data(interface):
    n_points  = interface.api.hidden['specJet.Data'].aqGetParDimSize(0)
    array = np.zeros(n_points,dtype=np.complex128)
    for i in range(n_points):
        array[i] = interface.api.hidden['specJet.Data'][i] + 1j* interface.api.hidden['specJet.Data1'][i]

    return array
    
def tune_power(
            interface, channel: str, tol=0.1, maxiter=30,
            bounds=[0, 100],hardware_wait=3, echo='abs') -> float:
            """Tunes the attenuator of a given channel to a given target using the
            standard scipy optimisation scripts. 

            Parameters
            ----------
            channel : str
                The chosen MPFU channel. Options: ['+<x>', '-<x>', '+<y>', '-<y>']
            tol : float, optional
                The tolerance in attenuator parameter, by default 0.1
            maxiter : int, optional
                The maximum number of iterations in the optimisation, by default 30

            Returns
            -------
            float
                The optimal value of the attenuator parameter

            """
            channel_opts = ['+<x>', '-<x>', '+<y>', '-<y>','ELDOR']
            if channel not in channel_opts:
                raise ValueError(f'Channel must be one of: {channel_opts}')
            
            if channel == '+<x>':
                atten_channel = 'BrXAmp'
            elif channel == '-<x>':
                atten_channel = 'BrMinXAmp'
            elif channel == '+<y>':
                atten_channel = 'BrYAmp'
            elif channel == '-<y>':
                atten_channel = 'BrMinYAmp'
            elif channel == 'ELDOR':
                atten_channel = 'ELDORAtt'

            lb = bounds[0]
            ub = bounds[1]

            if echo=='abs':
                tran_sum = lambda x: -1 * np.sum(np.abs(x))
            elif echo=='R+':
                tran_sum = lambda x: -1 * np.sum(np.real(x))
            elif echo=='R-':
                tran_sum = lambda x: 1 * np.sum(np.real(x))
            elif echo=='I+':
                tran_sum = lambda x: -1 * np.sum(np.real(x))
            elif echo=='I-':
                tran_sum = lambda x: 1 * np.sum(np.real(x))

            def objective(x, *args):
                interface.api.hidden[atten_channel].value = x  # Set phase to value
                time.sleep(hardware_wait)
                data = get_specjet_data(interface)

                val = tran_sum(data)

                print(f'Power Setting = {x:.1f} \t Echo Amplitude = {-1*val:.2f}')

                return val

            output = minimize_scalar(
                objective, method='bounded', bounds=[lb, ub],
                options={'xatol': tol, 'maxiter': maxiter})
            result = output.x
            print(f"Optimal Power Setting for {atten_channel} is: {result:.1f}")
            interface.api.hidden[atten_channel].value = result
            return result
    
def tune_phase(interface,
    channel: str, target: str, tol=0.1, maxiter=30,bounds=[0, 100],hardware_wait=3) -> float:
    """Tunes the phase of a given channel to a given target using the
    standard scipy optimisation scripts. 

    Parameters
    ----------
    channel : str
        The chosen MPFU channel. Options: ['+<x>', '-<x>', '+<y>', '-<y>']
    target : str
        The target echo position, this can either be maximising (+) or
        minimising (-) either the real (R) or imaginary (I) of the echo. 
        Options: ['R+', 'R-', 'I+', 'I-']
    tol : float, optional
        The tolerance in phase parameter, by default 0.1
    maxiter : int, optional
        The maximum number of iterations in the optimisation, by default 30

    Returns
    -------
    float
        The optimal value of the phase parameter

    """

    channel_opts = ['+<x>', '-<x>', '+<y>', '-<y>','Main']
    phase_opts = ['R+', 'R-', 'I+', 'I-']

    if channel not in channel_opts:
        raise ValueError(f'Channel must be one of: {channel_opts}')

    if target not in phase_opts:
        raise ValueError(f'Phase target must be one of: {phase_opts}')
    if channel == '+<x>':
        phase_channel = 'BrXPhase'
    elif channel == '-<x>':
        phase_channel = 'BrMinXPhase'
    elif channel == '+<y>':
        phase_channel = 'BrYPhase'
    elif channel == '-<y>':
        phase_channel = 'BrMinYPhase'
    elif channel == 'Main':
        phase_channel = 'SignalPhase'
    

    if target == 'R+':
        test_fun = lambda x: -1 * np.real(x)
    elif target == 'R-':
        test_fun = lambda x: 1 * np.real(x)
    elif target == 'I+':
        test_fun = lambda x: -1 * np.imag(x)
    elif target == 'I-':
        test_fun = lambda x: 1 * np.imag(x)

    lb = bounds[0]
    ub = bounds[1]

    def objective(x, *args):
        # x = x[0]
        interface.api.hidden[phase_channel].value = x  # Set phase to value
        time.sleep(hardware_wait)
        data = get_specjet_data(interface)

        val = test_fun(np.sum(data))

        print(f'Phase Setting = {x:.1f} \t Echo Amplitude = {-1*val:.2f}')

        return val

    output = minimize_scalar(
        objective, method='bounded', bounds=[lb, ub],
        options={'xatol': tol, 'maxiter': maxiter})
    result = output.x
    print(f"Optimal Phase Setting for {phase_channel} is: {result:.1f}")
    try:
        interface.api.hidden[phase_channel].value = result
    except:
        pass

    return result
    
def MPFUtune(interface, sequence, channels, echo='Hahn',tol: float = 0.1,
             bounds=[0, 100],tau_value=400):

    hardware_wait=1


    def phase_to_echo(phase):

        if np.isclose(phase,0):
            return 'R+'
        elif np.isclose(phase,np.pi):
            return 'R-'
        elif np.isclose(phase,np.pi/2):
            return 'I+'
        elif np.isclose(phase,3*np.pi/2) or np.isclose(phase,-np.pi/2):
            return 'I-'
        else:
            raise ValueError('Phase must be a multiple of pi/2')

    for i,channel in enumerate(channels):
        if channel =='ELDOR':
            continue
        MPFU_chanel = interface.MPFU[i]
        if MPFU_chanel == '+<x>':
            phase_cycle = 'BrXPhase'
        elif MPFU_chanel == '-<x>':
            phase_cycle = 'BrMinXPhase'
        elif MPFU_chanel == '+<y>':
            phase_cycle = 'BrYPhase'
        elif MPFU_chanel == '-<y>':
            phase_cycle = 'BrMinYPhase'
        
        if channel[0] == np.inf:
            interface.api.set_attenuator(MPFU_chanel,100)
            continue
        
        tau = Parameter("tau",tau_value,dim=4,step=0)

        exc_pulse = RectPulse(freq=0, tp = np.around(np.pi/2 /channel[0]), scale=1, flipangle = np.pi/2)
        ref_pulse = RectPulse(freq=0, tp = np.around(np.pi / channel[0] ), scale=1, flipangle = np.pi)
        
        seq = HahnEchoSequence(
            B=sequence.B,LO=sequence.LO,reptime=sequence.reptime,averages=1,
            shots=10, tau=tau, exc_pulse=exc_pulse, ref_pulse=ref_pulse)
        seq.pulses[0].pcyc = {'Phases': [0], 'DetSigns': [1.0]}
        seq._buildPhaseCycle()
        seq.evolution([tau])

        interface.launch(seq, savename="autoTUNE", start=True, tune=False, MPFU_overwrite=[MPFU_chanel,MPFU_chanel],reset_bg_data=False,reset_cur_exp=False)

        time.sleep(3)
        interface.terminate()
        interface.api.cur_exp['ftEPR.StartPlsPrg'].value = True
        
        if not interface.api.hidden['specJet.AverageStart'].value:
            interface.api.hidden['specJet.AverageStart'].value = True
        

        # interface.api.set_PulseSpel_phase_cycling('auto')
        print(f"Tuning channel: {MPFU_chanel}")
        tune_power(interface, MPFU_chanel, tol=tol, bounds=bounds,hardware_wait=hardware_wait)
        tune_phase(interface, MPFU_chanel, phase_to_echo(channel[1]), tol=tol)


def ELDORtune(interface, sequence, freq,
             tau_value=400):

    sequence_gyro = sequence.B.value / sequence.LO.value
    new_freq = sequence.LO.value + freq
    new_B = new_freq * sequence_gyro
    interface.api.set_ELDOR_freq(new_freq)

    ref_echoseq = Sequence(name='ELDOR tune',B=new_B, LO=new_freq, reptime=sequence.reptime, averages=1, shots=10)


    # tune a pair of 90/180 pulses at the eldor frequency
    test_tp = 16
    MPFUtune(interface, ref_echoseq, [(np.pi/2 / test_tp,0)],echo='Hahn')


    tp = Parameter("tp",16)
    long_delay = Parameter("long_delay",2000)
    tau = Parameter("tau",tau_value,dim=4,step=0)
    ref_echoseq.addPulse(RectPulse(freq=0, t=0, tp=tp, flipangle=np.pi))
    ref_echoseq.addPulse(RectPulse(freq=0, t=long_delay,tp=tp, flipangle=np.pi/2))
    ref_echoseq.addPulse(RectPulse(freq=0, t=long_delay+tau,tp=tp, flipangle=np.pi))
    ref_echoseq.addPulse(Detection(t=long_delay+2*tau, tp=512))
    ref_echoseq.evolution([tau])
    ref_echoseq.pulses[0].pcyc["Channels"] = "ELDOR"

    interface.launch(ref_echoseq, savename="autoTUNE", start=True, tune=False,reset_bg_data=False,reset_cur_exp=False)
    time.sleep(3)
    interface.terminate()
    interface.api.cur_exp['ftEPR.StartPlsPrg'].value = True
    if not interface.api.hidden['specJet.AverageStart'].value:
        interface.api.hidden['specJet.AverageStart'].value = True

    print(f"Tuning channel: ELDOR")
    # tune_phase(interface, 'Main', tol=5, bounds=[0,4096],target='R-')
    
    atten_axis = np.arange(30,0,-1)
    data = np.zeros(atten_axis.shape,dtype=np.complex128)
    for i,x in enumerate(atten_axis):
        interface.api.hidden['ELDORAtt'].value = x  # Set phase to value
        time.sleep(1)
        data[i] = np.trapz(get_specjet_data(interface))
    
    data = correctphase(data)
    if data[0] < 1:
        data = -1*data
    new_value = np.around(atten_axis[data.argmin()],2)
    print(f"ELDOR Atten Set to: {new_value}")
    interface.api.hidden['ELDORAtt'].value = new_value

    # tune_power(interface, 'ELDOR', tol=1, bounds=[0,30],echo='R-')

def test_if_MPFU_compatability(seq):
    table = seq.progTable
    if 'LO' in table['Variable']:
        return False
    elif np.unique(table['axID']).size > 2:
        return False 
    else:
        return True

