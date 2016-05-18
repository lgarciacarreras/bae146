import numpy as np
import iris
from iris.coords import DimCoord, AuxCoord
import bae146 as bae
import atmospheric as atm
import os
import glob
from fnmatch import fnmatch

def core(filename, var=None, event=None, fsummary=None):
    
    '''
    data_cube = core(filename, var=None, var_name=None, run=None)

    Read data from BAe146.
    var: common variable names (uses var2name function to find correct var_name).
    run var2name without arguments to get list of accepted names (e.g. 'u', 'v', 'temp')
    var_name: netcdf var_name
    event: Default outputs all the data. Specific 'Run', 'Profile' or 'Sonde' times can be 
    extracted using this option. Will output from the start of the first event to the end 
    of the last one (e.g. if requesting all runs with event='Run'.
    fsummary: filename of the flight_sum* file. Default will look in the same folder as the data.
    '''

    #---------------
    # READ variable, as well as coordinate data (lon/lat/time)
    #---------------

    var_name = bae.read.var2name(var)

    if var_name == 'derived':
        cube = bae.read.derived_146(filename, var)
    else:
        cube = iris.load_cube(filename,iris.Constraint(cube_func = lambda c: c.var_name == var_name))

    t = iris.load_cube(filename, 'time')
    lat = iris.load_cube(filename, 
                        iris.Constraint(cube_func=lambda c: c.var_name == 'LAT_GIN'))
    lon = iris.load_cube(filename, 
                        iris.Constraint(cube_func=lambda c: c.var_name == 'LON_GIN'))
    alt = iris.load_cube(filename, 
                        iris.Constraint(cube_func=lambda c: c.var_name == 'ALT_GIN'))



    #------------
    # CREATE COORDINATES from metadata and add to data
    #------------

    latitude = AuxCoord(lat.data.flatten(),
                        units=lat.units, var_name=lat.var_name,
                        long_name=lat.long_name, standard_name=lat.standard_name)
    
    longitude = AuxCoord(lon.data.flatten(),
                        units=lon.units, var_name=lon.var_name,
                        long_name=lon.long_name, standard_name=lon.standard_name)

    altitude = AuxCoord(alt.data.flatten(),
                        units=alt.units, var_name=alt.var_name,
                        long_name=alt.long_name, standard_name=alt.standard_name)
 
    # for 32Hz data need to pad out 'time' - assumes data is continuous (should be) 
    if filename[-6:-3] == '1hz':
        time = DimCoord(t.data,
                        units=t.units, var_name=t.var_name, 
                        long_name=t.long_name, standard_name=t.standard_name)
    else:
        time = DimCoord(np.linspace(t.data[0], t.data[-1]+1-1/32., t.data.size*32.),
                        units=t.units, var_name=t.var_name, 
                        long_name=t.long_name, standard_name=t.standard_name)


    # recreate the cube with flattened data
    cube = iris.cube.Cube(cube.data.flatten(),
                    dim_coords_and_dims=[(time, 0)], units=cube.units,
                    aux_coords_and_dims=[(longitude, 0), (latitude, 0), (altitude, 0)],
                    var_name=cube.var_name, long_name=cube.long_name,
                    standard_name=cube.standard_name,
                    attributes=cube.attributes)

    # extract specific runs if requested
    if event:
        # if not given, look for fsummary in the same directory as the data.
        # if there is not exactly one file, abort
        if not fsummary:
            fsummary = glob.glob(os.path.dirname(filename)+'/flight-sum*txt')
            if len(fsummary) != 1:
                print('------------ ERROR ------------')
                print('Need to include fsummary as I could not automatically find a single '+
                      'flight-sum*txt in the flight data directory.'+
                      'Aborting...')
                return 
            else:
                fsummary = fsummary[0]


        start, end, event_name, height = bae.read.runtimes(fsummary=fsummary, event=event)

        # check 'event' actually produces any output
        if start == []:
            print('------------ ERROR ------------')
            print ('Event '+event+' was not recognized. Outputting the entire flight')
        else:
            print('Extracting from '+event_name[0]+' to '+event_name[-1])

            # extract from the first start to the last end
            cube = cube.extract(iris.Constraint(
                        time = lambda cell: start[0] <= cell <= end[-1]))

    return cube

def sonde(filename, var=None):
    '''
    sonde_cube = sonde(filename, var=None)

    Read dropsonde data. By default 'time' is the main coordinate,
    here it is switched to altitude, keeping time as an auxiliary coordinate

    If no var is specified, it will output everything
    '''

    # Read variable (all data if var==None)
    cubelist = iris.load(filename, var)

    # Read altitude and pressure and remove masked data (e.g. when sonde is on the ground) from the cubelist
    alt = iris.load_cube(filename, 'altitude above MSL')
    pres = iris.load_cube(filename, 'pres')
    mask = alt.data.mask

    cubelist = iris.cube.CubeList([c[~mask] for c in cubelist])

    # create pressure coordinate to be added to cube
    pressure = DimCoord(pres[~mask].data,
                        units=pres.units, var_name=pres.var_name,
                        long_name=pres.long_name, standard_name='air_pressure')

    # create altitude auxiliary coordinate to be added to cube
    altitude = AuxCoord(alt[~mask].data,
                        units=alt.units, var_name=alt.var_name,
                        long_name=alt.long_name, standard_name='altitude')
 
    # add coordinate, and demote 'time' to an AuxCoord
    for c in cubelist:
        iris.util.demote_dim_coord_to_aux_coord(c, 'time')
        c.add_dim_coord(pressure, 0)
        c.add_aux_coord(altitude, 0)
    
    # if cubelist has only one element (i.e. var has been specified), just output the cube
    if len(cubelist) > 1:
        return cubelist
    else:
        return cubelist[0]

def derived_variables(filename, var):

    if var == 'theta':
        temp = bae.read.core(filename, var='temp')
        pressure = bae.read.core(filename, var='pressure')
        pref = iris.coords.AuxCoord(1000.,
                                long_name='reference_pressure',
                                units='hPa')

        cube = temp * (pref * (pressure)**-1)** (287.03722/1005.7)
        cube.rename('potential temperature')
        
    return cube

def runtimes(fsummary=None, event=None):
    '''
    (start, end, event_name, height) = runtimes(fsummary=None, run=None)

    Extracts run times (in s from midnight) from the flight summary file (fsummary).
    'event' is a string (or list of strings) pointing either to a type of event:
    'Sonde', 'Run' or 'Profile' -> outputs a list including every sonde/run/profile
    OR specific runs: 'Profile 1' will output times for profile 1 only.
    Default outputs a list of start/end times for all 'events'.
    '''

    # initialise outputs
    start = []
    end = []
    event_name = []
    height = []

    # if no event input, output all the useful stuff
    if not event: event = ['Run','Profile','Sonde']

    # event must be a list
    if type(event) is not list: event=[event]

    # cycle through all the lines looking for the keywords
    # if present, convert times to seconds
    with open(fsummary, 'r') as f:
        for line in f:
            if any(s in line for s in event):
                # if event=Run 1, Run 10,11 etc. will also be output, so check for that first
                if event[0] == 'Run 1' and fnmatch(line[17:36].strip(), 'Run 1[0-9]'): continue

                # append start/end times, converting HHMMSS time to seconds from midnight
                start.append(
                  (np.int(line[:2])*3600) + (np.int(line[2:4])*60) + np.int(line[4:6]))

                try: 
                    end.append(
                      (np.int(line[8:10])*3600) + (np.int(line[10:12])*60) + np.int(line[12:14]))
                except ValueError:# if there is no end time (e.g. for sondes)
                    end.append(start[-1])

                event_name.append(line[17:36].strip())
                height.append(line[36:55].strip())

    return (start,end,event_name, height)

def var2name(var=None):
    '''
    var_name = var2name(var=None)
    Convert easy to remember variable names (e.g. 'u' for horizontal wind) into the appropriate var_name to be used for extraction
    A call without arguments will print out all the available names.
    '''

    names = {'u':'U_C',
             'v':'V_C',
             'w':'W_C',
             'palt':'PALT_RVS', # pressure altitude
             'alt':'ALT_GIN', # gps altitude
             'radalt':'HGT_RADR', # radar altitude
             'pressure':'PS_RVSM', # air pressure
             'temp':'TAT_DI_R', # deiced temperature
             'nditemp':'TAT_ND_R', # non deiced temperature
             'bt':'BTHEIM_C', # brightness temperature
             'tdew':'TDEW_GE',
             'swd':'SW_DN_C',
             'swu':'SW_UP_C',
             'swd_r':'RED_DN_C', # 'red dome' radiation
             'swu_r':'RED_UP_C',
            # derived variables will be calculated
             'theta':'derived'} # potential temp.

    if not var: # print out all the available names
        print "Allowed variable names are:"
        for i in names.iterkeys(): print i
        return
    else:

        try:
            name = names[var]
        except KeyError:
            print var+' is not recognized as a valid standard variable name.'
            print 'Calling the function without arguments will print the list of available variables.'
            print 'Will now try and continue using "var" as a var_name variable'
            name = var

    return name


