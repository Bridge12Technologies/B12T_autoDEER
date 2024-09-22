import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from autodeer.colors import primary_colors



def calculate_optimal_tau(CPanalysis, MeasTime, SNR, target_step=0.015, target_shrt=None, corr_factor=1, ci=50, full_output=True):
    """
    Calculate the optimal evolution time for a given SNR and measurement time.

    Parameters
    ----------
    CPanalysis : CPanalysis
        The analysis object for the Carr-Purcell Relaxation measurement
    MeasTime : float or array-like
        The measurement time in hours. If array-like, the optimal tau will be calculated for each value.
    SNR : float
        The desired signal-to-noise ratio
    target_step : float, optional
        The target step size for the interpolation, by default 0.015 in microseconds
    target_shrt : float, optional
        The target shortest tau value for the interpolation in seconds. If None is given then the value is take from the relaxation experiment, by default None
    corr_factor : float, optional
        Correction factor for the MNR of the DEER fit result, by default 1
    ci : int, optional
        Confidence interval for fit in perc, by default 50
    full_output : bool, optional
        If True, the function will return the lower and upper bounds of the optimal tau, by default True
    
    Returns
    -------
    float or tuple
        The optimal tau value or a tuple with the lower and upper bounds of the optimal
    """
    dataset = CPanalysis.dataset
    results = CPanalysis.fit_result
    averages = dataset.nAvgs * dataset.shots * dataset.nPcyc
    noise = results.noiselvl

    if target_shrt is None:
        target_shrt = dataset.attrs['reptime'] *1e-6 # us -> s
    nPointsInTime = lambda x: x * 3600 / target_shrt
    n_points = lambda x: x / (target_step*1e-3)
    dt=target_step

    x=np.linspace(1e-3, CPanalysis.axis.values.max(), 1000)
    y = np.linspace(0,0.2,1000)
    fit = results.evaluate(CPanalysis.fit_model, x)*results.scale
    fitUncert = results.propagate(CPanalysis.fit_model, x)
    fitUncertCi = fitUncert.ci(ci)*results.scale
    ub = CPanalysis.fit_model(x,*results.paramUncert.ci(ci)[:-1,0])*results.paramUncert.ci(ci)[-1,0]
    lb = CPanalysis.fit_model(x,*results.paramUncert.ci(ci)[:-1,1])*results.paramUncert.ci(ci)[-1,1]
    # VCi = fitUncert.ci(ci)*results.scale
    # ub = VCi[:,1]
    # lb = VCi[:,0]

    spl_fit_inverse = interp.InterpolatedUnivariateSpline(np.flip(corr_factor*fit/(noise*np.sqrt(averages)*np.sqrt(x*2/dt))),np.flip(x*2), k=3)
    spl_fit_inverse_lb = interp.InterpolatedUnivariateSpline(np.flip(corr_factor*lb/(noise*np.sqrt(averages)*np.sqrt(x*2/dt))),np.flip(x*2), k=3)
    spl_fit_inverse_ub = interp.InterpolatedUnivariateSpline(np.flip(corr_factor*ub/(noise*np.sqrt(averages)*np.sqrt(x*2/dt))),np.flip(x*2), k=3)

    optimal = spl_fit_inverse(SNR  / np.sqrt(nPointsInTime(MeasTime)))
    optimal_lb = spl_fit_inverse_lb(SNR / np.sqrt(nPointsInTime(MeasTime)))
    optimal_ub = spl_fit_inverse_ub(SNR/ np.sqrt(nPointsInTime(MeasTime)))

    if full_output:
        return optimal, optimal_lb, optimal_ub
    else:
        return optimal
    
def plot_optimal_tau(CPanlaysis, SNR,MeasTime=None, MaxMeasTime=24, labels=None, fig=None, axs=None):
    """
    Generate a plot of the optimal evolution time for a given SNR against measurement time.

    Parameters
    ----------
    CPanlaysis : CPanalysis
        The analysis object for the Carr-Purcell Relaxation measurement
    SNR : float or array-like
        The desired signal-to-noise ratio. If array-like, a line will be drawn for each value.
    MeasTime : float or array-like, optional
        Draw crosshairs at specific measurement times, by default None. There must be the same number of elements in MeasTime and SNR
    MaxMeasTime : int, optional
        The maximum measurement time for the plot in hours, by default 24
    labels : list, optional
        Labels for the SNR values, by default None
    fig : matplotlib.figure.Figure, optional
        Figure to plot on, by default None
    axs : matplotlib.axes.Axes, optional
        Axes to plot on, by default None

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    
    """
    if fig is None and axs is None:
        fig, axs = plt.subplots(1,1)
    elif axs is None:
        axs = fig.add_subplot(111)
    
    MeasTimeAxis = np.logspace(0, np.log2(MaxMeasTime), 100, base=2)
    
    if not isinstance(SNR, (list, tuple)):
        SNR = [SNR]
    
    if labels is None:
        labels = [f'SNR = {snr}' for snr in SNR]

    for i,snr in enumerate(SNR):
        optimal, optimal_lb, optimal_ub = calculate_optimal_tau(CPanlaysis, MeasTimeAxis, snr, full_output=True)
        axs.plot(MeasTimeAxis, optimal, color=primary_colors[i], label=labels[i])
        axs.fill_between(MeasTimeAxis, optimal_lb, optimal_ub, color=primary_colors[i], alpha=0.3)

    axs.set_xlim(*axs.get_xlim())
    axs.set_ylim(*axs.get_ylim())

    if MeasTime is not None:
        if not isinstance(MeasTime, (list, tuple)):
            MeasTime = [MeasTime]
        if len(MeasTime) != len(SNR):
            raise ValueError('MeasTime and SNR must have the same length')
        for i, (mt, snr) in enumerate(zip(MeasTime, SNR)):
            optimal, optimal_lb, optimal_ub = calculate_optimal_tau(CPanlaysis, mt, snr, full_output=True)
            ylim = axs.get_ylim()
            axs.vlines(mt, *ylim,ls='--', color=primary_colors[i])
            xlim = axs.get_xlim()
            axs.hlines(optimal, *xlim,ls='--', color=primary_colors[i])
            axs.fill_between(xlim, (optimal_lb,optimal_lb), (optimal_ub,optimal_ub), color=primary_colors[i], alpha=0.3)
            


    axs.set_xlabel('Measurement time / h')
    axs.set_ylabel(r'$\tau_{evo}$ / $\mu$s')
    axs.legend(ncol=2, loc = 'lower right')

    return fig


def calc_correction_factor(CPanalysis,DEER_fit_result):
    """
    Calculate the correction factor for the MNR of the DEER fit result.

    Parameters
    ----------
    CPanalysis : CPanalysis
        CPanalysis object
    DEER_fit_result : DEERanalysis
        DEER_fit_result object
    
    Returns
    -------
    float
        Correction factor
    """
    shrt = DEER_fit_result.dataset.attrs['reptime'] *1e-6 # us -> s
    nPointsInTime = lambda x: x * 3600 / shrt
    t_axis = DEER_fit_result.t
    DEER_sequence = DEER_fit_result.dataset.epr.sequence

    actual_time = DEER_sequence._estimate_time() *(DEER_fit_result.dataset.nAvgs/DEER_fit_result.dataset.averages)

    actual_MNR = DEER_fit_result.MNR

    if DEER_sequence.name == '5pDEER':
        dataset = CPanalysis.dataset
        CP_averages = dataset.nAvgs * dataset.shots * dataset.nPcyc
        noise = CPanalysis.fit_result.noiselvl
        V = lambda tau: CPanalysis.fit_result.evaluate(CPanalysis.fit_model, tau)*CPanalysis.fit_result.scale

        tau = DEER_fit_result.dataset.attrs['tau1']/1e3

        est_SNR = V(tau)/(noise*np.sqrt(CP_averages))
        est_SNR *= np.sqrt(nPointsInTime(actual_time)/t_axis.shape[0])

        correction_factor = actual_MNR/est_SNR
    
    return correction_factor

