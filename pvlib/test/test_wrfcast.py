"""
Tests for the wrfcast.py module.
"""
import os
import xarray as xr
from pvlib import wrfcast
from pvlib.wrfcast import WRF

# Below are the testing parameters... they may be flexible
wrfout = os.path.join(
    '/Users/swardy9230/Box Sync/01_Research/01_Renewable_Analysis/',
    'Wind Resource Analysis/wrfout/19mp4lw4sw7lsm8pbl99cu',
    'wrfout_d02_2011-07-17')
start = 'July 17 2011'
end = 'Jul 18 2011'


def test_get_wrf_data():
    model = WRF()
    wrfdata = model.get_wrf_data(wrfout, start, end)
    assert type(wrfdata) == xr.Dataset


def test_process_data():
    model = WRF()
    wrfdata = model.get_processed_data(wrfout, start, end)
    assert list(wrfdata.keys()) == model.output_variables
    