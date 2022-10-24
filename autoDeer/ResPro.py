import numpy as np
import scipy.fft as fft
from scipy.interpolate import pchip_interpolate
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle


class resonatorProfile:

    def __init__(
            self, nuts: np.ndarray, freqs: np.ndarray, dt: float) -> None:
        """Analysis and calculation of resonator profiles.

        Parameters
        ----------
        nuts : np.ndarray
            An array containing the nutation data
        freqs: np.ndarray
            The respective frequencies of each nutation.
        dt : float
            The time step for the nutations.
        """
        if nuts.ndim != 2:
            raise ValueError(
                "nuts must be a 2D array where collumn 1 is the frequency")
        self.nutations = nuts
        self.frequencies = freqs
        self.n_files = np.shape(nuts)[0]
        self.nx = np.shape(nuts)[1] - 1
        self.dt = dt
        pass

    def calc_res_prof(
            self, throwlist: list = None,
            freq_lim: tuple = None) -> np.ndarray:
        """Uses a fourier transform method to calculate a resonator profile.

        Parameters
        ----------
        throwlist : list, optional
            Specific frequencies to be ignored, by default None
        freq_lim : tuple, optional
            Minimum and Maximum frequncies to used, by default None

        Returns
        -------
        np.ndarray
            The resonator profile. The first column are bridge frequncies, and
            the second are the coresponding nutation frequncies.
        """

        nu1_cutoff = 5*1e-3

        fax = fft.fftfreq(self.nx*2*10, d=self.dt)
        fax = fft.fftshift(fax)

        fnut = fft.fft(self.nutations, self.nx*2*10, axis=1)
        fnut = fft.fftshift(fnut, axes=1)

        nuoff = np.argwhere(fax > nu1_cutoff)
        nu1_id = np.argmax(np.abs(fnut[:, nuoff]), axis=1)
        # nu1 = abs(fax[nu1_id + nuoff[0]])

        data = np.transpose([abs(self.frequencies), 
                            abs(fax[nu1_id + nuoff[0]]).reshape(self.n_files)]) 

        throw_mask = np.zeros(0, dtype=np.int16)
        if freq_lim is not None:
            throw_mask = np.squeeze(np.argwhere(data[:, 0] < freq_lim[0]))
            throw_mask = np.append(
                throw_mask, np.squeeze(np.argwhere(data[:, 0] > freq_lim[1])))

        if throwlist is not None:
            for freq in throwlist:
                throw_mask = np.append(
                    throw_mask,
                    np.squeeze(np.argwhere(abs(data[:, 0] - freq) < 0.001)))

        data = np.delete(data, throw_mask, axis=0)

        self.res_prof = data
    
        return self.res_prof

    # def import_from_bruker(self, folder_path, reg_exp="nut_[0-9]+.DSC"):
        
    #     files = os.listdir(folder_path)
    #     r = re.compile(reg_exp)
    #     dsc_files = list(filter(r.match, files))
    #     n_files = len(dsc_files)
        
    #     # Load one file to get the time axis and length
    #     t, V = deerload(folder_path + files[0], False, False)
    #     t_axis = t
    #     nx = len(V)

    #     nut_data = np.zeros((n_files, nx+1), dtype=np.complex64)
        
    #     for i, files in enumerate(dsc_files):
    #         t, V, params = deerload(folder_path + files, False, True)
    #         if not np.array_equal(t, t_axis):
    #             raise Exception("all files must have the same time axis")
    #         spec_freq = re.findall(
    #             "[0-9]+.[0-9]*",
    #             params['DSL']['freqCounter']['FrequencyMon'])[0]
    #         nut_data[i, 0] = spec_freq
    #         nut_data[i, 1:] = V

    #     # sort nut_data
    #     nut_data_sorted = nut_data[nut_data[:, 0].argsort(), :]

    #     return t_axis, nut_data_sorted

    def calculate_shape(self, params=None):
        """Generates a single lorentzian fit for the resonator profile

        Parameters
        ----------
        params : list, optional
            Initial guess for the qfactor and central frequency, 
            by default None
        """
        if params is None:
            params = self.params
        
        A = params[0]
        self.q = params[1]
        self.fc = params[2]

        nu_x = self.res_prof[:, 0]
        nu_y = self.res_prof[:, 1] * 1e3  # Convert from GHz to MHz

        nu_y = nu_y/A

        # Build domains of filters
        srate = 36 * 3 * 4
        # This comes from Q/W = Bandwidth. So we want our frequency range to 
        # be 10 * Bandwidth.
        # w = 2 *pi * 1.25(IF freq)
        npts = int(2 ** (np.ceil(np.log2(10 * 50 / (2 * np.pi * 1.25) * srate)
                                 ) + 1))
        frq = np.linspace(-1/2, 1/2, npts, endpoint=False) * srate
        habs = np.zeros(frq.shape)

        # Interpolate the resonator profile onto the new frequency axis
        # Where to interpolate
        int_idx = np.squeeze(np.argwhere((nu_x[0] < frq) & (frq < nu_x[-1])))  
        habs[int_idx] = pchip_interpolate(nu_x, nu_y, frq[int_idx])
        habs_o = habs
        habs = habs/np.max(habs)
        self.nu_max = habs_o[frq > 33].max()

        # Lorentzian model
        z = self.q * (frq**2 - self.fc ** 2)/(frq*self.fc)
        model = (1-z * 1j) / (1 + z ** 2)  # Lorentzian = 1-zj/(1+z^2)
        model[np.isnan(model)] = 0 
        l_abs = np.abs(model)  # Model fit
        l_abs /= np.nanmax(l_abs)  # Normalization
        zrs = np.squeeze(np.argwhere(l_abs == 0))
        l_abs[zrs] = l_abs[zrs+1]

        # Higher frequencies, above zero
        l_h = l_abs[len(l_abs)//2+1:-1]  
        # Make symetrical about zero
        l_abs_ds = np.concatenate([np.flip(l_h), l_h])  
        # Get phase from the hilbert transform
        l_pha = np.imag(hilbert(-np.log(l_abs_ds)))  

        # Adding the exp resonantor profile into a lorentzian function
        fdec = 0.1/np.log(2)
        n_dec = int(np.ceil(fdec/(frq[1]-frq[0])))
        n_cyc = 10

        # left side
        habs[0:int_idx[0]] = l_abs[0:int_idx[0]]  # put into lorentzian

        # smooth with exp function
        s_range = np.arange(int_idx[0]-n_cyc * n_dec, int_idx[0]+1)
        habs[s_range] = l_abs[s_range] + \
            np.flip((habs[int_idx[0]] - l_abs[int_idx[0]]) *
                    np.exp(-1 * np.arange(0, n_cyc+1/n_dec, 1/n_dec)))

        # #smooth the edges with a window
        # w_pts = 10
        # win = windows.gaussian(w_pts,2.5) / 
        #                        np.sum(windows.gaussian(w_pts,2.5))
        # w_range = np.arange(int_idx[0]-2*w_pts,int_idx[0] + 2*w_pts)
        # sm = np.convolve(habs[w_range],win,mode='valid')
        # ins_range = np.arange(w_range[0] + np.ceil(w_pts/2),w_range[0] - 
        #                   np.floor(w_pts/2))

        # Repeat for the right hand side
        habs[int_idx[-1]:-1] = l_abs[int_idx[-1]:-1]

        # Smooth with exp function
        s_range = np.arange(int_idx[-1], int_idx[-1]+n_cyc*n_dec+1)
        habs[s_range] = l_abs[s_range] + \
            (habs[int_idx[-1]-1] - l_abs[int_idx[-1]-1]) * \
            np.exp(-1 * np.arange(0, n_cyc+1/n_dec, 1/n_dec))

        h_pha = np.imag(hilbert(-np.log(habs)))

        self.habs = habs
        self.labs = l_abs
        self.l_pha = l_pha
        self.h_pha = h_pha
        self.frq = frq

    def phase_plots(self, nu_lims: tuple = None) -> plt.figure:
        """Plot the phase change across the resonator profile

        Parameters
        ----------
        nu_lims : tuple, optional
            frequency limits of the plot, by default None
        Returns
        -------
        plt.figure
            matplotlib figure
        """
         
        frq_axis = self.frq

        if nu_lims is not None:
            fmin = nu_lims[0]
            fmax = nu_lims[1]
        else:
            fmin = 0
            fmax = 40 
        
        fig, ax = plt.subplots()
        ax.plot(frq_axis, self.habs, label="Interpolated fit")
        ax.plot(frq_axis, self.labs, label="Lorentzian Model")
        ax.set_xlabel('Frequency GHz')
        ax.set_ylabel('Normalised Nutation Freq.')
        ax.legend()
        ax.set_title('Resontor Profile')
        ax.set_xlim((fmin, fmax))

        return fig

    def res_prof_plot(
            self, fieldsweep=None, nu_lims: tuple = None) -> plt.figure:
        """ Plot the resonator profile

        Parameters
        ----------
        fieldsweep : _type_, optional
            Overlay the fieldsweep on the resonator profile, by default None
        nu_lims : tuple, optional
            Frequency limits on plot, by default None

        Returns
        -------
        plt.figure
            matplotlib figure of resonator profile
        """

        frq_axis = self.frq
        if nu_lims is not None:
            fmin = nu_lims[0]
            fmax = nu_lims[1]
        else:
            fmin = 0
            fmax = 40 

        if fieldsweep is not None:
            if not hasattr(fieldsweep, "fs_x"):
                fieldsweep.calc_gyro()
            data = fieldsweep.data
            data /= np.max(np.abs(data))
        
        fig, ax = plt.subplots()
        ax.plot(frq_axis, self.habs, label="Interpolated fit")
        ax.plot(frq_axis, self.labs, label="Lorentzian Model")
        if fieldsweep is not None:
            ax.plot(fieldsweep.fs_x, np.abs(data), label="Appox. Field-Sweep")
        ax.set_xlabel('Frequency GHz')
        ax.set_ylabel('Normalised Nutation Freq.')
        ax.legend()
        ax.set_title('Resonator Profile')
        ax.set_xlim((fmin, fmax))

        return fig
    
    def autofit(self, nu_lims: tuple = [33, 35]):
        """Automatically calculate resonator quality factor and center
        frequency.

        Parameters
        ----------
        nu_lims : tuple, optional
            Frequency bounds on fit, by default [33, 35]
        """
        def lorentz(f, a, q, fc):
            z = q * (f**2 - fc ** 2)/(f*fc)
            model = a*(1-z*1j) / (1 + z**2)  # Lorentzian = 1-zj/(1+z^2)
            return np.abs(model)

        fmin = nu_lims[0]
        fmax = nu_lims[1]

        nu_x = self.res_prof[:, 0]
        nu_y = self.res_prof[:, 1] * 1e3
        nu_y_norm = nu_y/nu_y.max() 

        res, pcov = curve_fit(
            lorentz, nu_x, nu_y_norm, [1, 100, (fmin+fmax)/2],
            bounds=([0.7, 0, fmin], [1.3, 1000, fmax]))
        std = np.sqrt(np.diag(pcov))

        self.params = res
        self.q = self.params[1]
        self.fc = self.params[2]
        self.params_std = std

    def calc_IF_prof(
            self, IF: float, bridgeFreq: float, nu_lims=[33, 35], type='fit'):
        # Where to interpolate
        int_idx = np.squeeze(np.argwhere(
            (nu_lims[0] < self.frq) & (self.frq < nu_lims[-1])))  
        self.IF = self.frq[int_idx] - bridgeFreq + IF
        if type == 'fit':
            self.IF_rp = self.labs[int_idx]
        elif type == 'raw':
            self.IF_rp = self.habs[int_idx]

    def _pickle_save(self, file="res_prof"):

        with open(file, 'w') as file:
            pickle.dump(self, file)

    # def _pickle_load(self, file="res_prof"):

    #     with open(file, 'r') as file:
    #         self = pickle.load(file)


