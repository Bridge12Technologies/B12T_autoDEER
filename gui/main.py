from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog,QTableWidgetItem, QErrorMessage,QMessageBox
from PyQt6 import uic
import PyQt6.QtCore as QtCore
import PyQt6.QtGui as QtGui
from pathlib import Path
import sys, traceback

from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import autodeer as ad
import deerlab as dl
import numpy as np
from quickdeer import DEERplot
import yaml

import tools
from queue import Queue


from autoDEER_worker import autoDEERWorker
QtCore.QDir.addSearchPath('icons', 'gui/resources')



class WorkerSignals(QtCore.QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)

class Worker(QtCore.QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress

    @QtCore.pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

def fieldsweep_process(dataset):
    fsweep_analysis = ad.FieldSweepAnalysis(dataset)
    fsweep_analysis.calc_gyro()
    fsweep_analysis.fit(xtol=1e-5, lin_maxiter=100)

    return fsweep_analysis

def respro_process(dataset, freq_axis, fieldsweep=None):
    respro = ad.ResonatorProfileAnalysis(
        nuts = dataset.data.T,
        freqs = freq_axis,
        dt=2
    )
    respro.process_nutations(threshold=4)
    
    respro.fit()

    if fieldsweep is not None:
        LO_new = fieldsweep.LO + ad.optimise_spectra_position(respro, fieldsweep)
        return respro, LO_new


    return respro

def relax_process(dataset):

    if dataset.axes[0].max()>500:
        dataset.axes = [dataset.axes[0]/1e3]
    else:
        dataset.axes = [dataset.axes[0]]
    CP_data = ad.CarrPurcellAnalysis(dataset)
    CP_data.fit()

    return CP_data

class UI(QMainWindow):


    def __init__(self):
        super().__init__()
 
        # loading the ui file with uic module
        uic.loadUi("gui/gui2.ui", self)
        logo_pixmap = QtGui.QPixmap('icons:logo.png')
        logo_pixmap = logo_pixmap.scaledToHeight(60)
        self.logo.setPixmap(logo_pixmap)
        self.set_spectrometer_connected_light(0)

        self.fsweep_toolbar()
        self.respro_toolbar()
        self.relax_toolbar()
        self.create_respro_figure()
        self.create_relax_figure()
        self.advanced_mode_inputs()
        self.optimise_pulses_button()

        self.threadpool = QtCore.QThreadPool()
        self.current_results = {}
        self.current_data = {}
        self.spectromterInterface = None
        self.waitCondition = None
        self.queue = Queue()

        self.FullyAutoButton.clicked.connect(self.RunFullyAutoDEER)
        self.AdvancedAutoButton.clicked.connect(self.RunAdvanedAutoDEER)

        docs_url = QtCore.QUrl('https://jeschkelab.github.io/autoDEER/')
        github_url = QtCore.QUrl('https://github.com/JeschkeLab/autoDEER/')
        issues_url = QtCore.QUrl('https://github.com/JeschkeLab/autoDEER/issues')
        discussion_url = QtCore.QUrl('https://github.com/JeschkeLab/autoDEER/discussions')

        self.actionDocumentation.triggered.connect(lambda: QtGui.QDesktopServices.openUrl(docs_url))
        self.actionGitHub.triggered.connect(lambda: QtGui.QDesktopServices.openUrl(github_url))
        self.actionIssues.triggered.connect(lambda: QtGui.QDesktopServices.openUrl(issues_url))
        self.actionDiscussions.triggered.connect(lambda: QtGui.QDesktopServices.openUrl(discussion_url))

        self.actionLoadConfig.triggered.connect(self.load_spectrometer_config)
        self.actionConnect.triggered.connect(self.connect_spectrometer)
        self.current_folder = ''
        self.config = None
        self.connected = False

        self.LO = 0
        self.gyro = 0.0028087

    def set_spectrometer_connected_light(self, state):
        if state == 0:
            light_pixmap = QtGui.QPixmap('icons:red.png')
        elif state == 1:
            light_pixmap = QtGui.QPixmap('icons:green.png')
        elif state == 2:
            light_pixmap = QtGui.QPixmap('icons:yellow.png')
        
        light_pixmap = light_pixmap.scaledToHeight(30)
        self.Connected_Light.setPixmap(light_pixmap)
    def load_epr_file(self, store_location):

        filename, _= QFileDialog.getOpenFileName(
            self,"Select a File", self.current_folder,"Data (*.DTA *.mat)")
        
        if filename:
                path = Path(filename)
                filename_edit = str(path)

        dataset = ad.eprload(filename_edit)
        self.current_data[store_location] = dataset

    def load_spectrometer_config(self):

        filename, _= QFileDialog.getOpenFileName(
            self,"Select a File", self.current_folder,"Data (*.yaml)")
        
        if filename:
            path = Path(filename)
            filename_edit = str(path)
        else:
            return None
        

        with open(filename_edit, mode='r') as f:
            config = yaml.safe_load(f)
            self.config = config

        spectrometer = config['Spectrometer']

        if spectrometer['Manufacturer'] == 'Bruker':
            if spectrometer['AWG']:
                model = 'Bruker_AWG'
            elif spectrometer['MPFU']:
                model = 'Bruker_MPFU'
        elif spectrometer['Manufacturer'] == 'ETH':
            model = 'ETH_AWG'
        elif spectrometer['Manufacturer'] == 'Dummy':
            model = 'Dummy'
        
        if model == 'Dummy':
            from autodeer.hardware.dummy import dummyInterface
            self.spectromterInterface = dummyInterface()
        elif model == 'ETH_AWG':
            from autodeer.hardware import ETH_awg_interface
            self.spectromterInterface = ETH_awg_interface()
        elif model == 'Bruker_MPFU':
            from autodeer.hardware import BrukerMPFU
            self.spectromterInterface = BrukerMPFU(filename_edit)
        elif model == 'Bruker_AWG':
            from autodeer.hardware import BrukerAWG
            self.spectromterInterface = BrukerAWG(filename_edit)

        # Find resonators
        self.resonatorComboBox.clear()
        self.resonatorComboBox.addItems(self.config['Resonators'].keys())

        # Set LO to resonator central frequency
        key1 = list(self.config['Resonators'].keys())[0]
        self.LO = self.config['Resonators'][key1]['Center Freq']


    def connect_spectrometer(self):
        if self.spectromterInterface is None:
            QMessageBox.about(self,'ERORR!', 'A interface needs to be loaded first!')
            return None


        self.spectromterInterface.connect()
        self.connected = True
        self.set_spectrometer_connected_light(1)

    def fsweep_toolbar(self):
        upload_icon = QtGui.QIcon('icons:upload.png')
        self.Load_button.setIcon(upload_icon)
        self.Load_button.clicked.connect(lambda x: self.load_epr_file('fieldsweep'))

        refresh_icon = QtGui.QIcon('icons:refresh.png')
        self.Refresh_button.setIcon(refresh_icon)
        self.Refresh_button.clicked.connect(lambda: self.update_fieldsweep())

    def respro_toolbar(self):
        upload_icon = QtGui.QIcon('icons:upload.png')
        self.respro_Load_button.setIcon(upload_icon)
        self.respro_Load_button.clicked.connect(lambda: self.load_epr_file('respro'))

        refresh_icon = QtGui.QIcon('icons:refresh.png')
        self.respro_Refresh_button.setIcon(refresh_icon)
        self.respro_Refresh_button.clicked.connect(lambda: self.update_respro())

    def relax_toolbar(self):
        upload_icon = QtGui.QIcon('icons:upload.png')
        self.relax_Load_button.setIcon(upload_icon)
        self.relax_Load_button.clicked.connect(lambda: self.load_epr_file('relax'))

        refresh_icon = QtGui.QIcon('icons:refresh.png')
        self.relax_Refresh_button.setIcon(refresh_icon)
        self.relax_Refresh_button.clicked.connect(lambda: self.update_relax())

    def update_fieldsweep(self, dataset=None):

        if dataset is None:
            dataset = self.current_data['fieldsweep']
        else:
            self.current_data['fieldsweep'] = dataset

        worker = Worker(fieldsweep_process, dataset)
        worker.signals.result.connect(self.refresh_fieldsweep)
        
        self.threadpool.start(worker)
 
    def refresh_fieldsweep(self, fitresult):

        self.current_results['fieldsweep'] = fitresult
        self.gyro = fitresult.gyro
        if self.waitCondition is not None: # Wake up the runner thread
            self.waitCondition.wakeAll()
        
        fsweep_plot = FigureCanvas(fitresult.plot())
        self.fsweep_v_left.removeWidget(self.fsweep_plot)
        self.fsweep_v_left.addWidget(fsweep_plot,1)



        # Add fit resultss

        self.gyroSpinBox.setValue(fitresult.gyro*1e3)
        self.gxSpinBox.setValue(-0.0025 * fitresult.results.az + 2.0175)
        self.gySpinBox.setValue(fitresult.results.gy)
        self.gzSpinBox.setValue(fitresult.results.gz)
        self.AxSpinBox.setValue(fitresult.results.axy*28.0328)
        self.AySpinBox.setValue(fitresult.results.axy*28.0328)
        self.AzSpinBox.setValue(fitresult.results.az)
        self.GBSpinBox.setValue(fitresult.results.GB)
        self.BoffsetSpinBox.setValue(fitresult.results.Boffset)

        gxCI = -0.0025 *fitresult.results.azUncert.ci(95)+ 2.0175
        self.gxCI.setText(f"({gxCI[0]:.4f},{gxCI[1]:.4f})")
        self.gyCI.setText(tools.getCIstring(fitresult.results.gyUncert))
        self.gzCI.setText(tools.getCIstring(fitresult.results.gzUncert))


        # parameters = ['Boffset','az','GB']

        # analysis_rows = []
        # for param in parameters:
        #     attr = getattr(fitresult.results,param)
        #     if isinstance(attr, dl.Parameter):
        #         unit = attr.unit
        #     else:
        #         unit = None

        #     if hasattr(fitresult.results,f"{param}Uncert"):
        #         CI_tmp = getattr(fitresult.results,f"{param}Uncert").ci(95)
        #         CI =f"({CI_tmp[0]:.2f},{CI_tmp[1]:.2f})"
        #     else:
        #         CI = None
        #     analysis_rows.append({"Parameter":param, "Value":f"{attr:.2f}", "95% CI":CI, "Unit":unit})
        
        # self.fsweep_analysis_table.setRowCount(len(analysis_rows))

        # for row,e in enumerate(analysis_rows):
        #     self.fsweep_analysis_table.setItem(row, 0, QTableWidgetItem(e['Parameter']))
        #     self.fsweep_analysis_table.setItem(row, 1, QTableWidgetItem(e['Value']))
        #     self.fsweep_analysis_table.setItem(row, 2, QTableWidgetItem(e['95% CI']))
        #     self.fsweep_analysis_table.setItem(row, 3, QTableWidgetItem(e['Unit']))

    def update_respro(self, dataset=None):

        if dataset is None:
            dataset = self.current_data['respro']
        else:
            self.current_data['respro'] = dataset

        if hasattr(dataset, 'sequence'):
            f_axis = dataset.sequence.LO.value + dataset.sequence.LO.axis[0]['axis'] - 0.3 # Fix this necessary offset
        else:
            # Assuming mat file
            # TODO: Change to new data format

            f_axis = dataset.params['parvars'][2]['vec'][:,1] + dataset.params['LO']


        worker = Worker(respro_process, dataset, f_axis)
        worker.signals.result.connect(self.refresh_respro)

        self.threadpool.start(worker)

    def create_respro_figure(self):
        fig, axs  = plt.subplots(1,1,figsize=(12.5, 6.28))
        self.respro_canvas = FigureCanvas(fig)
        self.respro_v_left.addWidget(NavigationToolbar2QT(self.respro_canvas, self))
        self.respro_v_left.addWidget(self.respro_canvas)
        self.respro_ax = axs


    def refresh_respro(self, *args):

        fitresult = args[0]
        self.current_results['respro'] = fitresult

        if len(args) > 1:
            self.LO = args[1]

        if self.waitCondition is not None: # Wake up the runner thread
            self.waitCondition.wakeAll()
            
        self.respro_ax.cla()

        if 'fieldsweep'in self.current_results:
            fitresult.plot(fieldsweep=self.current_results['fieldsweep'],axs=self.respro_ax,fig=self.respro_canvas.figure);
        else:
            fitresult.plot(axs=self.respro_ax,fig=self.respro_canvas.figure)

        self.respro_canvas.draw()
        # Add fit results

        self.centreFrequencyDoubleSpinBox.setValue(fitresult.results.fc)
        self.centreFrequencyCI.setText(f"({fitresult.results.fcUncert.ci(95)[0]:.2f},{fitresult.results.fcUncert.ci(95)[1]:.2f})")
        self.qDoubleSpinBox.setValue(fitresult.results.q)
        self.qCI.setText(f"({fitresult.results.qUncert.ci(95)[0]:.2f},{fitresult.results.qUncert.ci(95)[1]:.2f})")



    def optimise_pulses_button(self):

        self.OptimisePulsesButton.clicked.connect(lambda: self.optimise_pulses())

    def optimise_pulses(self, pulses=None):
        if pulses is None:
            pump_pulse = ad.HSPulse(tp=120, init_freq=-0.25, final_freq=-0.03, flipangle=np.pi, scale=0, order1=6, order2=1, beta=10)
            exc_pulse = ad.RectPulse(tp=16, freq=0.02, flipangle=np.pi/2, scale=0)
            ref_pulse = exc_pulse.copy(flipangle=np.pi)
        else:
            pump_pulse = pulses['pump_pulse']
            ref_pulse = pulses['ref_pulse']
            exc_pulse = pulses['exc_pulse']

        pump_pulse, exc_pulse, ref_pulse = ad.optimise_pulses(self.current_results['fieldsweep'], pump_pulse, exc_pulse, ref_pulse)
        
        self.respro_ax.cla()
        ad.plot_overlap(self.current_results['fieldsweep'], pump_pulse, exc_pulse,ref_pulse, axs=self.respro_ax,fig=self.respro_canvas.figure)
        self.respro_canvas.draw()



        if self.waitCondition is not None: # Wake up the runner thread
            self.waitCondition.wakeAll()
    

    def update_relax(self, dataset=None):
        if dataset is None:
            dataset = self.current_data['relax']
        else:
            self.current_data['relax'] = dataset

        worker = Worker(relax_process, dataset)
        worker.signals.result.connect(self.refresh_relax)

        self.threadpool.start(worker)

    def create_relax_figure(self):
        fig, axs  = plt.subplots(1,1,figsize=(12.5, 6.28))
        self.relax_canvas = FigureCanvas(fig)
        self.relax_v_left.addWidget(NavigationToolbar2QT(self.relax_canvas, self))
        self.relax_v_left.addWidget(self.relax_canvas)
        self.relax_ax = axs

    def refresh_relax(self, fitresult):
        self.current_results['relax'] = fitresult

        if self.waitCondition is not None: # Wake up the runner thread
            self.waitCondition.wakeAll()

        self.relax_ax.cla()
        fitresult.plot(axs=self.relax_ax,fig=self.relax_canvas.figure)
        self.relax_canvas.draw()

        tau2hrs = fitresult.find_optimal(averages=1, SNR_target=20, target_time=2, target_shrt=0.1, target_step=0.015)
        self.DipolarEvo2hrs.setValue(tau2hrs)
        
    def advanced_mode_inputs(self):
        self.Exp_types.addItems(['5pDEER','4pDEER','3pDEER','nDEER'])
        self.ExcPulseSelect.addItems(['Auto', 'Rectangular','Chirp','HS', 'Gauss'])
        self.RefPulseSelect.addItems(['Auto', 'Rectangular','Chirp','HS', 'Gauss'])
        self.PumpPulseSelect.addItems(['Auto', 'Rectangular','Chirp','HS', 'Gauss'])

    def update_quickdeer(self, dataset=None):
        if dataset is None:
            dataset = self.current_data['relax']
        else:
            self.current_data['relax'] = dataset

        self.q_DEER.current_data['quickdeer'] = dataset
        self.q_DEER.update_exp_table()
        self.q_DEER.update_figure()
        self.q_DEER.process_deeranalysis(wait_condition = self.waitCondition)


    def RunFullyAutoDEER(self):

        if self.spectromterInterface is None or self.connected is False:
            QMessageBox.about(self,'ERORR!', 'A interface needs to be connected first!')
            return None

        userinput = {}
        userinput['MaxTime'] = self.MaxTime.value()
        userinput['sample'] = self.SampleName.text()
        userinput['Temp'] = self.TempValue.value()

        # Block the autoDEER buttons
        self.FullyAutoButton.setEnabled(False)
        self.AdvancedAutoButton.setEnabled(False)

        self.waitCondition = QtCore.QWaitCondition()
        mutex = QtCore.QMutex()

        worker = autoDEERWorker(
            self.spectromterInterface,wait=self.waitCondition,mutex=mutex,
            results=self.current_results,LO=self.LO, gyro = self.gyro,
            user_inputs=userinput )
        worker.signals.status.connect(self.msgbar.setText)
        worker.signals.fsweep_result.connect(self.update_fieldsweep)
        worker.signals.respro_result.connect(self.update_respro)
        worker.signals.optimise_pulses.connect(self.optimise_pulses)
        worker.signals.relax_result.connect(self.update_relax)
        worker.signals.quickdeer_result.connect(self.update_quickdeer)
        worker.signals.finished.connect(lambda: self.FullyAutoButton.setEnabled(True))
        worker.signals.finished.connect(lambda: self.AdvancedAutoButton.setEnabled(True))


        self.threadpool.start(worker)

    def RunAdvanedAutoDEER(self):

        if self.spectromterInterface is None or self.connected is False:
            QMessageBox.about(self,'ERORR!', 'A interface needs to be connected first!')
            return None
        
        # Block the autoDEER buttons
        self.FullyAutoButton.setEnabled(False)
        self.AdvancedAutoButton.setEnabled(False)

        self.waitCondition = QtCore.QWaitCondition()
        mutex = QtCore.QMutex()
        userinput = {}
        userinput['MaxTime'] = self.MaxTime.value()
        userinput['sample'] = self.SampleName.text()
        userinput['Temp'] = self.TempValue.value()
        userinput['ExpType'] = self.Exp_types.currentText()
        userinput['tau1'] = self.Tau1Value.value()
        userinput['tau2'] = self.Tau2Value.value()
        userinput['tau3'] = self.Tau3Value.value()
        userinput['ExcPulse'] = self.ExcPulseSelect.currentText()
        userinput['RefPulse'] = self.RefPulseSelect.currentText()
        userinput['PumpPulse'] = self.PumpPulseSelect.currentText()

        worker = autoDEERWorker(
            self.spectromterInterface,wait=self.waitCondition,mutex=mutex,
            results=self.current_results,LO=self.LO, gyro = self.gyro,
            userinput=userinput )
        worker.signals.status.connect(self.msgbar.setText)
        worker.signals.fsweep_result.connect(self.update_fieldsweep)
        worker.signals.respro_result.connect(self.update_respro)
        worker.signals.optimise_pulses.connect(self.optimise_pulses)
        worker.signals.relax_result.connect(self.update_relax)
        worker.signals.quickdeer_result.connect(self.update_quickdeer)
        worker.signals.finished.connect(lambda: self.FullyAutoButton.setEnabled(True))
        worker.signals.finished.connect(lambda: self.AdvancedAutoButton.setEnabled(True))

        self.threadpool.start(worker)

if __name__ == '__main__':
    app = QApplication([])
    window = UI()
    window.show()
    app.exec()
