from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, ParticleFile, Variable
from scripts import convert_IndexedOutputToArray
from datetime import timedelta as delta
from glob import glob
import numpy as np
import math


def set_ofes_fieldset(ufiles):
    vfiles = [f.replace('u.nc', 'v.nc') for f in ufiles]
    vfiles = [f.replace('u_vel', 'v_vel') for f in vfiles]
    tfiles = [f.replace('u.nc', 't.nc') for f in ufiles]
    tfiles = [f.replace('u_vel', 'temp') for f in tfiles]
    filenames = {'U': ufiles, 'V': vfiles, 'temp': tfiles}
    variables = {'U': 'zu', 'V': 'zv', 'temp': 'temperature'}
    dimensions = {'lat': 'Latitude', 'lon': 'Longitude', 'time': 'Time',
                  'depth': 'Depth'}
    indices = {'lat': range(200, 500), 'lon': range(0, 500)}

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indices=indices)
    fieldset.U.data[fieldset.U.data > 1e5] = 0
    fieldset.V.data[fieldset.V.data > 1e5] = 0
    fieldset.temp.data[fieldset.temp.data == 0] = np.nan
    return fieldset


def SampleTemp(particle, fieldset, time, dt):
    particle.temp = fieldset.temp[time, particle.lon, particle.lat, particle.depth]


def Sink(particle, fieldset, time, dt):
    if particle.depth > fieldset.dwellingdepth:
        particle.depth = particle.depth + fieldset.sinkspeed * dt
    else:
        particle.depth = fieldset.dwellingdepth


def Age(particle, fieldset, time, dt):
    if particle.depth <= fieldset.dwellingdepth:
        particle.age = particle.age + math.fabs(dt)
    if particle.age > fieldset.maxage:
        particle.delete()


def DeleteParticle(particle):
    particle.delete()


def run_corefootprintparticles(outfile):
    basepath = '/Volumes/data01/OFESdata/OFES_0.1_HIND/allveldata/nest_1_*u.nc'
    files = list(reversed(glob(str(basepath))))  # Reverse the file list
    fieldset = set_ofes_fieldset(files[3:0:-1])
    fieldset.add_constant('dwellingdepth', 50.)
    fieldset.add_constant('sinkspeed', 200./86400)
    fieldset.add_constant('maxage', 30.*86400)

    corelon = [17.30]
    corelat = [-34.70]
    coredepth = [2440]

    class ForamParticle(JITParticle):
        temp = Variable('temp', dtype=np.float32, initial=np.nan)
        age = Variable('age', dtype=np.float32, initial=0.)

    pset = ParticleSet(fieldset=fieldset, pclass=ForamParticle, lon=corelon, lat=corelat,
                       depth=coredepth, time=fieldset.U.time[-1])
    pfile = ParticleFile(outfile, pset, type="indexed")
    pfile.write(pset, pset[0].time)

    kernels = pset.Kernel(AdvectionRK4) + Sink + SampleTemp + Age

    for i in range(4, len(files), 1):
        pset.execute(kernels, starttime=pset[0].time, runtime=delta(days=3),
                     dt=delta(minutes=-5), interval=delta(days=-1),
                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

        pset.add(ForamParticle(lon=corelon, lat=corelat, depth=coredepth, fieldset=fieldset))
        pfile.write(pset, pset[0].time)
        fieldset.advancetime(set_ofes_fieldset([files[i]]))


def make_plot(trajfile):
    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    class ParticleData(object):
        def __init__(self):
            self.id = []

    def load_particles_file(fname, varnames):
        T = ParticleData()
        pfile = Dataset(fname, 'r')
        T.id = pfile.variables['trajectory'][:]
        for v in varnames:
            setattr(T, v, pfile.variables[v][:])
        return T

    T = load_particles_file(trajfile, ['lon', 'lat', 'temp', 'z'])
    m = Basemap(projection='merc', llcrnrlat=-45, urcrnrlat=-25, llcrnrlon=5, urcrnrlon=35, resolution='h')
    m.drawcoastlines()
    m.drawparallels(np.arange(-50, -20, 10), labels=[True, False, False, False])
    m.drawmeridians(np.arange(0, 40, 10), labels=[False, False, False, True])

    sinks = np.where(T.z > 50.)
    dwell = np.where(T.z == 50.)
    xs, ys = m(T.lon[dwell], T.lat[dwell])
    m.scatter(xs, ys, c=T.temp[dwell], s=5)
    cbar = plt.colorbar()
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel('[$^\circ$C]')

    xs, ys = m(T.lon[sinks], T.lat[sinks])
    m.scatter(xs, ys, c='k', s=5)
    xs, ys = m(T.lon[0, 0], T.lat[0, 0])
    m.plot(xs, ys, 'om')
    plt.show()


outfile = "corefootprint_particles"
run_corefootprintparticles(outfile)
convert_IndexedOutputToArray(file_in=outfile+".nc", file_out=outfile+"_array.nc")
make_plot(outfile+"_array.nc")
