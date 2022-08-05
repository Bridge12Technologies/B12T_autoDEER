from attr import has
from matplotlib.figure import Figure
import numpy as np
from deerlab import deerload,noiselevel
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Carr_Purcell:

    def __init__(self) -> None:
        pass

    def import_from_bruker(self,file_path)->None:

        t,V = deerload(file_path, False, False)
        self.axis = t
        self.data = V
        pass 

    def import_from_numpy(self,axis:np.ndarray,spectrum:np.ndarray) -> None:
        self.axis = axis
        self.data = spectrum
        pass

    def import_from_dataclass(self,dataclass) -> None:
        self.axis = dataclass.time
        self.data = dataclass.data
        pass
    
    def fit(self,type = "mono"):

        data = np.abs(self.data)
        data /= np.max(data)

        if type == "mono":
            self.func = lambda x,a,b,e: a*np.exp(-b*x**e)
            p0 = [1,1,2]
            bounds = ([0,0,-10,0],[2,1000,10,1000])
        else:
            raise ValueError("Type must be one of: mono")
        
        self.fit_result = curve_fit(self.func,self.axis,data,p0=p0)
        pass


    def plot(self,norm=True) -> Figure:

        if norm == True:
            data = np.abs(self.data)
            data /= np.max(data)

        fig, ax = plt.subplots()
        if hasattr(self,"fit_result"):
            ax.plot(self.axis,self.func(self.axis,*self.fit_result[0]),label='fit')
            ax.plot(self.axis,data,label='data')
            ax.legend()
        else:
            ax.plot(self.axis,data,label='data')


        ax.set_xlabel('Time / us')
        ax.set_ylabel('Normalised Amplitude')
        return fig

    def find_optimal(self,shrt,averages):
        time_per_point = shrt * averages

        data = np.abs(self.data)
        data /= np.max(data)

        self.noise = noiselevel(data)
        data_snr = data/self.noise
        data_snr_avgs = data_snr/np.sqrt(averages)

        # Assume 16ns time step
        dt = 16
        # Target time
        target_time = 2 * 3600

        num_avgs = target_time / (shrt * np.floor(2 * self.axis * 1000 / 16))

        data_snr_ = data_snr_avgs * np.sqrt(num_avgs)

        self.optimal = self.axis[np.argmin(np.abs(data_snr_-20))]
        return self.optimal
