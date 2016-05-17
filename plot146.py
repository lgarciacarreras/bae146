import numpy as np
from iris import plot as iplt
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
import read_aircraft as reada
import plot_aircraft as aplot

def line(filename, var=None, xvar='time', event='Run',  # IMPORTANT INPUTS
        # options for read_146
        var_name=None, fsummary=None,
        # to save the figure
        fout = None):

    '''
    line(filename, var=None, xvar='time', event='Run', [fout=None, var_name=None, fsummary=None])

    Plot a simple line plot of 'var' against xvar
    A label with var+event is added to each line, so plt.legend() can be called after multiple calls to line

    xvar = 'time', 'latitude' or 'longitude'.
    event = list of strings giving runs/profiles/sondes to be plotted. 
    Will plot from the start of the first one to the end of the last.
    Defaults to 'Run' (no number, so plot all runs).
    fout = filename to save figure to. Won't save the figure by default.
    '''

# Read data
    cube = reada.read_146(filename, var=var, var_name=var_name, event=event, fsummary=fsummary)

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
        var_name=None, event='Run', fsummary=None,  # options for read_146
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
    cube = reada.read_146(filename, var=var, var_name=var_name, event=event, fsummary=fsummary)
    alt = reada.read_146(filename, var=altvar, event=event, fsummary=fsummary)
    x = cube.coord(xvar).points

    # create coloured line
    lc = aplot.gen_contour_line(x, alt.data, cube.data, cmin=cmin, cmax=cmax)

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
        var_name=None, fsummary=None, # options for read_146
        lonlim = (72,88), latlim=(22,30), cmin=None, cmax=None): # plotting options

    '''
    Plot the flight path coloured by variable var on a map.

    event = list of strings giving runs/profiles/sondes to be plotted. 
    Will plot from the start of the first one to the end of the last.
    Defaults to 'Run' (no number, so plot all runs).
    lonlim/latlim = map boundaries (defaults to northern India)
    cmin/cmax = min/max for the colorbar. Default is min/max of the data.
    fout = filename to save figure to. Won't save the figure by default.

    '''

    cube = reada.read_146(filename, var=var, var_name=var_name, event=event, fsummary=fsummary)

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
    lc = aplot.gen_contour_line(cube.coord('longitude').points,
                                cube.coord('latitude').points,
                                cube.data, cmin=cmin, cmax=cmax)

    # plot line, and add colour bar
    plt.gca().add_collection(lc)
    axcb = fig.colorbar(lc)
    axcb.set_label(cube.name()+' / '+cube.units.symbol)

    # save plot if fout is given
    if fout:
        plt.savefig(fout)

def sonde_tephi(filename, winds=False, minp=None, fout=None):
    '''
    Plot tephigram from a sonde.
    If wind==True (default False) also add wind barbs
    If minp is set, it determines vertical extent of plot.
    Requires tephi python package
    '''
    import tephi

    # READ data and convert into (alt, variable) tuples
    temp = reada.read_sonde(filename, var='tdry')    
    tdew = reada.read_sonde(filename, var='dp')
    pressure = reada.read_sonde(filename, var='pres')

    # remove missing (masked) data, as it creates problems with tephi
    valid = np.where((np.ma.getmaskarray(temp.data) == False) & 
                     (np.ma.getmaskarray(tdew.data) == False))
    
    tplot = zip(pressure.data[valid], temp.data[valid])
    tdplot = zip(pressure.data[valid], tdew.data[valid])

    if winds:
        wsp = reada.read_sonde(filename, var='wspd')
        wdir = reada.read_sonde(filename, var='wdir')
        wvalid = np.where((np.ma.getmaskarray(wsp.data) == False) & 
                     (np.ma.getmaskarray(wdir.data) == False))

        # need to thin out wind barbs to one every

        windplot = zip(wsp.data[wvalid], wdir.data[wvalid], pressure.data[wvalid])

    # PLOT
    fig = plt.figure()
    tpg = tephi.Tephigram(figure=fig, anchor=[(1000,-20),(300,0)])
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

