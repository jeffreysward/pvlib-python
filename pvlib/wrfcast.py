'''
The 'forecast' module contains class definitions for
retreiving forecasted data from UNIDATA Thredd servers.
'''
import datetime
from netCDF4 import num2date
import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
from xml.etree.ElementTree import ParseError

from pvlib.location import Location
from pvlib.irradiance import liujordan, get_extra_radiation, disc
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

import warnings

warnings.warn(
    'The forecast module algorithms and features are highly experimental. '
    'The API may change, the functionality may be consolidated into an io '
    'module, or the module may be separated into its own package.')


class ForecastModel(object):
    """
    An object for querying and holding forecast model information for
    use within the pvlib library.

    Simplifies use of siphon library on a THREDDS server.

    Parameters
    ----------
    model_type: string
        UNIDATA category in which the model is located.
    model_name: string
        Name of the UNIDATA forecast model.
    set_type: string
        Model dataset type.

    Attributes
    ----------
    access_url: string
        URL specifying the dataset from data will be retrieved.
    base_tds_url : string
        The top level server address
    catalog_url : string
        The url path of the catalog to parse.
    data: pd.DataFrame
        Data returned from the query.
    data_format: string
        Format of the forecast data being requested from UNIDATA.
    dataset: Dataset
        Object containing information used to access forecast data.
    dataframe_variables: list
        Model variables that are present in the data.
    datasets_list: list
        List of all available datasets.
    fm_models: Dataset
        TDSCatalog object containing all available
        forecast models from UNIDATA.
    fm_models_list: list
        List of all available forecast models from UNIDATA.
    latitude: list
        A list of floats containing latitude values.
    location: Location
        A pvlib Location object containing geographic quantities.
    longitude: list
        A list of floats containing longitude values.
    lbox: boolean
        Indicates the use of a location bounding box.
    ncss: NCSS object
        NCSS
    model_name: string
        Name of the UNIDATA forecast model.
    model: Dataset
        A dictionary of Dataset object, whose keys are the name of the
        dataset's name.
    model_url: string
        The url path of the dataset to parse.
    modelvariables: list
        Common variable names that correspond to queryvariables.
    query: NCSS query object
        NCSS object used to complete the forecast data retrival.
    queryvariables: list
        Variables that are used to query the THREDDS Data Server.
    time: DatetimeIndex
        Time range.
    variables: dict
        Defines the variables to obtain from the weather
        model and how they should be renamed to common variable names.
    units: dict
        Dictionary containing the units of the standard variables
        and the model specific variables.
    vert_level: float or integer
        Vertical altitude for query data.
    """

    data_format = 'netcdf'

    units = {
        'temp_air': 'C',
        'wind_speed': 'm/s',
        'ghi': 'W/m^2',
        'ghi_raw': 'W/m^2',
        'dni': 'W/m^2',
        'dhi': 'W/m^2',
        'total_clouds': '%',
        'low_clouds': '%',
        'mid_clouds': '%',
        'high_clouds': '%'}

    def __init__(self, model_type, model_name, set_type, vert_level=None):
        self.model_type = model_type
        self.model_name = model_name
        self.set_type = set_type
        self.connected = False
        self.vert_level = vert_level

    def __repr__(self):
        return '{}, {}'.format(self.model_name, self.set_type)

    def get_data(self, latitude, longitude, start, end,
                 vert_level=None, query_variables=None,
                 close_netcdf_data=True, **kwargs):
        """
        Submits a query to the UNIDATA servers using Siphon NCSS and
        converts the netcdf data to a pandas DataFrame.

        Parameters
        ----------
        latitude: float
            The latitude value.
        longitude: float
            The longitude value.
        start: datetime or timestamp
            The start time.
        end: datetime or timestamp
            The end time.
        vert_level: None, float or integer, default None
            Vertical altitude of interest.
        query_variables: None or list, default None
            If None, uses self.variables.
        close_netcdf_data: bool, default True
            Controls if the temporary netcdf data file should be closed.
            Set to False to access the raw data.
        **kwargs:
            Additional keyword arguments are silently ignored.

        Returns
        -------
        forecast_data : DataFrame
            column names are the weather model's variable names.
        """

        if not self.connected:
            self.connect_to_catalog()

        if vert_level is not None:
            self.vert_level = vert_level

        if query_variables is None:
            self.query_variables = list(self.variables.values())
        else:
            self.query_variables = query_variables

        self.latitude = latitude
        self.longitude = longitude
        self.set_query_latlon()  # modifies self.query
        self.set_location(start, latitude, longitude)

        self.start = start
        self.end = end
        self.query.time_range(self.start, self.end)

        if self.vert_level is not None:
            self.query.vertical_level(self.vert_level)

        self.query.variables(*self.query_variables)
        self.query.accept(self.data_format)

        self.netcdf_data = self.ncss.get_data(self.query)

        # might be better to go to xarray here so that we can handle
        # higher dimensional data for more advanced applications
        self.data = self._netcdf2pandas(self.netcdf_data, self.query_variables,
                                        self.start, self.end)

        if close_netcdf_data:
            self.netcdf_data.close()

        return self.data


    def get_wrf_data(self, wrfout_file,
                 vert_level=None, query_variables=None,
                 close_netcdf_data=True, **kwargs):
        """
        Finds a local wrfout file and
        converts the netcdf data to a pandas DataFrame.

        Parameters
        ----------
        wrfout_file: str
            Location of wrfout NetCDF file.
        vert_level: None, float or integer, default None
            Vertical altitude of interest.
        query_variables: None or list, default None
            If None, uses self.variables.
        close_netcdf_data: bool, default True
            Controls if the temporary netcdf data file should be closed.
            Set to False to access the raw data.
        **kwargs:
            Additional keyword arguments are silently ignored.

        Returns
        -------
        forecast_data : DataFrame
            column names are the weather model's variable names.
        """

        if vert_level is not None:
            self.vert_level = vert_level

        if query_variables is None:
            self.query_variables = list(self.variables.values())
        else:
            self.query_variables = query_variables


        self.netcdf_data = 0

        # might be better to go to xarray here so that we can handle
        # higher dimensional data for more advanced applications
        self.data = self._netcdf2pandas(self.netcdf_data, self.query_variables,
                                        self.start, self.end)

        if close_netcdf_data:
            self.netcdf_data.close()

        return self.data

    def process_data(self, data, **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data. Most forecast models implement
        their own version of this method which also call this one.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """
        data = self.rename(data)
        return data

    def get_processed_data(self, *args, **kwargs):
        """
        Get and process forecast data.

        Parameters
        ----------
        *args: positional arguments
            Passed to get_data
        **kwargs: keyword arguments
            Passed to get_data and process_data

        Returns
        -------
        data: DataFrame
            Processed forecast data
        """
        return self.process_data(self.get_data(*args, **kwargs), **kwargs)

    def rename(self, data, variables=None):
        """
        Renames the columns according the variable mapping.

        Parameters
        ----------
        data: DataFrame
        variables: None or dict, default None
            If None, uses self.variables

        Returns
        -------
        data: DataFrame
            Renamed data.
        """
        if variables is None:
            variables = self.variables
        return data.rename(columns={y: x for x, y in variables.items()})

    def _netcdf2pandas(self, netcdf_data, query_variables, start, end):
        """
        Transforms data from netcdf to pandas DataFrame.

        Parameters
        ----------
        data: netcdf
            Data returned from UNIDATA NCSS query, or from your local forecast.
        query_variables: list
            The variables requested.
        start: Timestamp
            The start time
        end: Timestamp
            The end time

        Returns
        -------
        pd.DataFrame
        """
        # set self.time
        try:
            time_var = 'time'
            self.set_time(netcdf_data.variables[time_var])
        except KeyError:
            # which model does this dumb thing?
            time_var = 'time1'
            self.set_time(netcdf_data.variables[time_var])

        data_dict = {}
        for key, data in netcdf_data.variables.items():
            # if accounts for possibility of extra variable returned
            if key not in query_variables:
                continue
            squeezed = data[:].squeeze()
            if squeezed.ndim == 1:
                data_dict[key] = squeezed
            elif squeezed.ndim == 2:
                for num, data_level in enumerate(squeezed.T):
                    data_dict[key + '_' + str(num)] = data_level
            else:
                raise ValueError('cannot parse ndim > 2')

        data = pd.DataFrame(data_dict, index=self.time)
        # sometimes data is returned as hours since T0
        # where T0 is before start. Then the hours between
        # T0 and start are added *after* end. So sort and slice
        # to remove the garbage
        data = data.sort_index().loc[start:end]
        return data

    def set_time(self, time):
        '''
        Converts time data into a pandas date object.

        Parameters
        ----------
        time: netcdf
            Contains time information.

        Returns
        -------
        pandas.DatetimeIndex
        '''
        times = num2date(time[:].squeeze(), time.units)
        self.time = pd.DatetimeIndex(pd.Series(times), tz=self.location.tz)

    def cloud_cover_to_ghi_linear(self, cloud_cover, ghi_clear, offset=35,
                                  **kwargs):
        """
        Convert cloud cover to GHI using a linear relationship.

        0% cloud cover returns ghi_clear.

        100% cloud cover returns offset*ghi_clear.

        Parameters
        ----------
        cloud_cover: numeric
            Cloud cover in %.
        ghi_clear: numeric
            GHI under clear sky conditions.
        offset: numeric, default 35
            Determines the minimum GHI.
        kwargs
            Not used.

        Returns
        -------
        ghi: numeric
            Estimated GHI.

        References
        ----------
        Larson et. al. "Day-ahead forecasting of solar power output from
        photovoltaic plants in the American Southwest" Renewable Energy
        91, 11-20 (2016).
        """

        offset = offset / 100.
        cloud_cover = cloud_cover / 100.
        ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
        return ghi

    def cloud_cover_to_irradiance_clearsky_scaling(self, cloud_cover,
                                                   method='linear',
                                                   **kwargs):
        """
        Estimates irradiance from cloud cover in the following steps:

        1. Determine clear sky GHI using Ineichen model and
           climatological turbidity.
        2. Estimate cloudy sky GHI using a function of
           cloud_cover e.g.
           :py:meth:`~ForecastModel.cloud_cover_to_ghi_linear`
        3. Estimate cloudy sky DNI using the DISC model.
        4. Calculate DHI from DNI and GHI.

        Parameters
        ----------
        cloud_cover : Series
            Cloud cover in %.
        method : str, default 'linear'
            Method for converting cloud cover to GHI.
            'linear' is currently the only option.
        **kwargs
            Passed to the method that does the conversion

        Returns
        -------
        irrads : DataFrame
            Estimated GHI, DNI, and DHI.
        """
        solpos = self.location.get_solarposition(cloud_cover.index)
        cs = self.location.get_clearsky(cloud_cover.index, model='ineichen',
                                        solar_position=solpos)

        method = method.lower()
        if method == 'linear':
            ghi = self.cloud_cover_to_ghi_linear(cloud_cover, cs['ghi'],
                                                 **kwargs)
        else:
            raise ValueError('invalid method argument')

        dni = disc(ghi, solpos['zenith'], cloud_cover.index)['dni']
        dhi = ghi - dni * np.cos(np.radians(solpos['zenith']))

        irrads = pd.DataFrame({'ghi': ghi, 'dni': dni, 'dhi': dhi}).fillna(0)
        return irrads

    def cloud_cover_to_transmittance_linear(self, cloud_cover, offset=0.75,
                                            **kwargs):
        """
        Convert cloud cover to atmospheric transmittance using a linear
        model.

        0% cloud cover returns offset.

        100% cloud cover returns 0.

        Parameters
        ----------
        cloud_cover : numeric
            Cloud cover in %.
        offset : numeric, default 0.75
            Determines the maximum transmittance.
        kwargs
            Not used.

        Returns
        -------
        ghi : numeric
            Estimated GHI.
        """
        transmittance = ((100.0 - cloud_cover) / 100.0) * offset

        return transmittance

    def cloud_cover_to_irradiance_liujordan(self, cloud_cover, **kwargs):
        """
        Estimates irradiance from cloud cover in the following steps:

        1. Determine transmittance using a function of cloud cover e.g.
           :py:meth:`~ForecastModel.cloud_cover_to_transmittance_linear`
        2. Calculate GHI, DNI, DHI using the
           :py:func:`pvlib.irradiance.liujordan` model

        Parameters
        ----------
        cloud_cover : Series

        Returns
        -------
        irradiance : DataFrame
            Columns include ghi, dni, dhi
        """
        # in principle, get_solarposition could use the forecast
        # pressure, temp, etc., but the cloud cover forecast is not
        # accurate enough to justify using these minor corrections
        solar_position = self.location.get_solarposition(cloud_cover.index)
        dni_extra = get_extra_radiation(cloud_cover.index)
        airmass = self.location.get_airmass(cloud_cover.index)

        transmittance = self.cloud_cover_to_transmittance_linear(cloud_cover,
                                                                 **kwargs)

        irrads = liujordan(solar_position['apparent_zenith'],
                           transmittance, airmass['airmass_absolute'],
                           dni_extra=dni_extra)
        irrads = irrads.fillna(0)

        return irrads

    def cloud_cover_to_irradiance(self, cloud_cover, how='clearsky_scaling',
                                  **kwargs):
        """
        Convert cloud cover to irradiance. A wrapper method.

        Parameters
        ----------
        cloud_cover : Series
        how : str, default 'clearsky_scaling'
            Selects the method for conversion. Can be one of
            clearsky_scaling or liujordan.
        **kwargs
            Passed to the selected method.

        Returns
        -------
        irradiance : DataFrame
            Columns include ghi, dni, dhi
        """

        how = how.lower()
        if how == 'clearsky_scaling':
            irrads = self.cloud_cover_to_irradiance_clearsky_scaling(
                cloud_cover, **kwargs)
        elif how == 'liujordan':
            irrads = self.cloud_cover_to_irradiance_liujordan(
                cloud_cover, **kwargs)
        else:
            raise ValueError('invalid how argument')

        return irrads

    def dni_and_dhi_to_ghi(self, dni, dhi, zenith, **kwargs):
        """
        Calculates global horizontal irradiance.

        Parameters
        ----------
        dni : Series
            Direct normal irradiance in W m-2.
        dhi : Series
            Diffuse normal irradiance in W m-2.
        zenith : Series
        **kwargs
            Not used

        Returns
        -------
        ghi : Series (but maybe should be DataFrame)
            Global horizontal irradiance in W m-2.
        """


        ghi = dhi + dni * np.cos(np.radians(zenith))
        return ghi

    def kelvin_to_celsius(self, temperature):
        """
        Converts Kelvin to celsius.

        Parameters
        ----------
        temperature: numeric

        Returns
        -------
        temperature: numeric
        """
        return temperature - 273.15

    def uv_to_speed(self, data):
        """
        Computes wind speed from wind components.

        Parameters
        ----------
        data : DataFrame
            Must contain the columns 'wind_speed_u' and 'wind_speed_v'.

        Returns
        -------
        wind_speed : Series
        """
        wind_speed = np.sqrt(data['wind_speed_u']**2 + data['wind_speed_v']**2)

        return wind_speed

    def gust_to_speed(self, data, scaling=1/1.4):
        """
        Computes standard wind speed from gust.
        Very approximate and location dependent.

        Parameters
        ----------
        data : DataFrame
            Must contain the column 'wind_speed_gust'.

        Returns
        -------
        wind_speed : Series
        """
        wind_speed = data['wind_speed_gust'] * scaling

        return wind_speed


class WRF(ForecastModel):
    """
    Subclass of the ForecastModel class representing your own
    WRF forecast model.

    Parameters
    ----------
    set_type: string, default 'best'
        Not used

    Attributes
    ----------
    dataframe_variables: list
        Common variables present in the final set of data.
    model: string
        Name of the UNIDATA forecast model.
    model_type: string
        UNIDATA category in which the model is located.
    variables: dict
        Defines the variables to obtain from the weather
        model and how they should be renamed to common variable names.
    units: dict
        Dictionary containing the units of the standard variables
        and the model specific variables.
    """

    def __init__(self, set_type='best'):
        model_type = 'Forecast Model Data'
        model = 'WRF Forecast'

        self.variables = {
            'temp_air': 'TEMP at 2 M', # T2
            'wind_speed_u': 'U at 10 M', # U10
            'wind_speed_v': 'V at 10 M', # V10
            'total_clouds': 'CLOUD FRACTION', # CLDFRA
            'dni': 'Shortwave surface downward direct normal irradiance', # SWDDNI
            'dhi': 'Shortwave surface downward diffuse irradiance' # SWDDIF
            }

        self.output_variables = [
            'temp_air',
            'wind_speed',
            'ghi',
            'dni',
            'dhi'
            ]

        super(WRF, self).__init__(model_type, model, set_type, vert_level=None)

    def process_data(self, data, cloud_cover='total_clouds', **kwargs):
        """
        Defines the steps needed to convert raw forecast data
        into processed forecast data.

        Parameters
        ----------
        data: DataFrame
            Raw forecast data
        cloud_cover: str, default 'total_clouds'
            The type of cloud cover used to infer the irradiance.

        Returns
        -------
        data: DataFrame
            Processed forecast data.
        """
        data = super(WRF, self).process_data(data, **kwargs)
        data['temp_air'] = self.kelvin_to_celsius(data['temp_air'])
        data['wind_speed'] = self.uv_to_speed(data)
        data['ghi'] = self.dni_and_dhi_to_ghi(data['dni'], data['dhi'])
        return data[self.output_variables]
