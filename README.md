# autoDeer
A python package designed for the running of automated and optimised DEER/PELDOR experiments for pulsed EPR. Currently this is built as a python extension to Bruker's Xepr software. 

## Features
- Automated Parameter optimization
- Automated Running of a 2hr DEER experiment with plotting
- Saving all experiments to a .h5 file

## Dependencies
1) Numpy
2) Scipy
3) Matplotlib
4) pytest
5) h5py
6) pyvisa
7) Xepr API for python 3 (https://github.com/OE-FET/XeprAPI)
8) DeerLab (https://jeschkelab.github.io/DeerLab/)

## External Software Dependencies
1) Bruker Xepr
2) Keysight IO Libaries Suite Win/Linux (https://www.keysight.com/gb/en/lib/software-detail/computer-software/io-libraries-suite-downloads-2175637.html)

## Troubleshooting
- <Can't find any open instances of Xepr>
    This has two main causes:
    1) Xepr API hasn't be started. Please do "Processing" -> "XeprAPI" -> "Enable Xepr API"
    2) The kernal has recently lost connection, and a background kernal is connected. Please open <htop> in terminal and
    kill the process of the other python kernal

## Acknowledgements 
In developing the Bruker spectrometer control, insipration and help was gained from previous work by:
- Spinach by Ilya Kuprov et. al.
- CustomXepr by San Schott

