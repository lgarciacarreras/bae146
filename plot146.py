import numpy as np
from iris import plot as iplt
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
import bae146 as bae

def line(filename, var=None, xvar='time', event='Run',  # IMPORTANT INPUTS
        fsummary=None, fout = None):   

    '''
    line(filename, var=None, xvar='time', event='Run', [fout=None, fsummary=None])

    Plot a simple line plot of 'var' against xvar
    A label with var+event is added to each line, so plt.legend() can be called after multiple calls to line

    xvar = 'time', 'latitude' or 'longitude'.
    event = list of strings giving runs/profiles/sondes to be plotted. 
    Will plot from the start of the first one to the end of the last.
    Defaults to 'Run' (no number, so plot all runs).
    fout = filename to save figure to. Won't save the figure by default.
    '''

# Read data
    cube = bae.read.core(filename, var=var, event=event, fsummary=fsummary)

    if xvar=='time': # use iris plot as time is the dimcoord (and gives better labelling)
        iplt.plot(cube, label=var+' : '+event)
    else:
        plt.plot(cube.coord(xvar).points, cube.data, label=var+' : '+event)

    plt.xlabel(xvar)
    plt.ylabel(cube.name()+' / '+cube.units.symbol)

    # suppress scientific notation on y axis
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

    if fout:
        plt.savefig(fout)
        
def x_alt_path(filename, var=None, xvar='longitude', altvar='alt', # IMPORTANT INPUTS
        event='Run', fsummary=None,  # options for bae.read.core
        cmin=None, cmax=None, #plotting options
        fout=None):  

    '''
    Plot a line of the aircraft path (x vs altitude) coloured by 'var'

    altvar = altitude variable used ('alt','palt','radalt','pressure')
    xvar = x variable ('longitude', 'latitude')
    event = list of strings giving runs/profiles/sondes to be plotted. 
    Will plot from the start of the first one to the end of the last.
    Defaults to 'Run' (no number, so plot all runs).
    cmin/cmax = min/max for the colorbar. Default is min/max of the data.
    fout = filename to save figure to. Won't save the figure by default.
    '''

# Read data and altitude
    cube = bae.read.core(filename, var=var, event=event, fsummary=fsummary)
    alt = bae.read.core(filename, var=altvar, event=event, fsummary=fsummary)
    x = cube.coord(xvar).points

    # create coloured line
    lc = bae.plot.gen_contour_line(x, alt.data, cube.data, cmin=cmin, cmax=cmax)

    # Plot data
    fig = plt.figure()
    plt.gca().add_collection(lc)
    plt.xlim(np.floor(x.min()), np.ceil(x.max()))

    if altvar == 'pressure': # invert y axis
        plt.ylim(alt.data.max()*1.05, alt.data.min()*.9)
    else:
        plt.ylim(alt.data.min()*.95, alt.data.max()*1.1)

    plt.xlabel(cube.coord(xvar).name())
    plt.ylabel(alt.name()+' / '+alt.units.symbol)
    
    axcb = fig.colorbar(lc)
    axcb.set_label(cube.name()+' / '+cube.units.symbol)

    # save plot if fout is given
    if fout:
        plt.savefig(fout)


def x_y_path(filename, var=None, event='Run', # IMPORTANT inputs
        fsummary=None, # options for bae.read.core
        lonlim = (72,88), latlim=(22,30), cmin=None, cmax=None, # plotting options
        fout=None): 

    '''
    Plot the flight path coloured by variable var on a map.

    event = list of strings giving runs/profiles/sondes to be plotted. 
    Will plot from the start of the first one to the end of the last.
    Defaults to 'Run' (no number, so plot all runs).
    lonlim/latlim = map boundaries (defaults to northern India)
    cmin/cmax = min/max for the colorbar. Default is min/max of the data.
    fout = filename to save figure to. Won't save the figure by default.

    '''

    cube = bae.read.core(filename, var=var, event=event, fsummary=fsummary)

    # Generate map
    m = Basemap(llcrnrlon=lonlim[0], llcrnrlat=latlim[0], 
                urcrnrlon=lonlim[1], urcrnrlat=latlim[1])
    fig = plt.figure()
    m.drawcoastlines()

    # add lat/lon lines (unnecessarily complicated...)
    lon0 = np.floor(lonlim[0])
    lat0 = np.floor(latlim[0])
    lon1 = np.ceil(lonlim[1])
    lat1 = np.ceil(latlim[1])
    if np.max([lat1-lat0, lon1-lon0]) <= 15:
        dl = 1
    elif np.max([lat1-lat0, lon1-lon0]) <= 30:
        dl = 5
    else:
        dl = 10

    m.drawparallels(np.arange(lat0, lat1, dl), labels=[1,0,0,0])
    m.drawmeridians(np.arange(lon0, lon1, dl), labels=[0,1,0,1])
    
    # generate line collection which will apply the coloured line
    lc = bae.plot.gen_contour_line(cube.coord('longitude').points,
                                cube.coord('latitude').points,
                                cube.data, cmin=cmin, cmax=cmax)

    # plot line, and add colour bar
    plt.gca().add_collection(lc)
    axcb = fig.colorbar(lc)
    axcb.set_label(cube.name()+' / '+cube.units.symbol)

    # save plot if fout is given
    if fout:
        plt.savefig(fout)

def sonde_tephi(filename, winds=False, nbarbs=50, fout=None):
    '''
    Plot tephigram from a dropsonde file.
    If wind==True (default False) also add wind barbs
    nbarbs: approximate number of barbs to plot (default=50)
    Requires tephi python package
    '''
    import tephi

    # READ dropsonde data 
    temp = bae.read.sonde(filename, var='tdry')    
    tdew = bae.read.sonde(filename, var='dp')

    # remove missing (masked) data, as it creates problems with tephi
    valid = np.where((np.ma.getmaskarray(temp.data) == False) & 
                     (np.ma.getmaskarray(tdew.data) == False))
    
    # generate list of (pressure, temp) required by tephi
    tplot = zip(temp.coord('air_pressure').points[valid], temp.data[valid])
    tdplot = zip(tdew.coord('air_pressure').points[valid], tdew.data[valid])

    # repeat for winds if requested
    if winds:
        wsp = bae.read.sonde(filename, var='wspd')
        wdir = bae.read.sonde(filename, var='wdir')
        wvalid = np.where((np.ma.getmaskarray(wsp.data) == False) & 
                     (np.ma.getmaskarray(wdir.data) == False))

        windplot = zip(wsp.data[wvalid], wdir.data[wvalid], wsp.coord('air_pressure').points[wvalid])

        # limit windplot to length nbarbs
        windplot = windplot[::len(windplot)/nbarbs]

    # PLOT
    fig = plt.figure()
    tpg = tephi.Tephigram(figure=fig)
    profile = tpg.plot(tplot, color='r')
    tpg.plot(tdplot, color='b')
    if winds: profile.barbs(windplot)
     
    if fout:
        plt.savefig(fout)
    

def gen_contour_line(x, y, z, cmin=None, cmax=None):
    '''
    Generate 'line collection' which allows plotting a 2d line coloured by a third variable z
    '''

    if cmin is None: cmin = z.min()
    if cmax is None: cmax = z.max()

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('Spectral'),
                            norm=plt.Normalize(cmin,cmax))

    lc.set_array(z)
    lc.set_linewidth(3)

    return lc

