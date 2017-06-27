from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, ParticleFile, Variable
from scripts import convert_IndexedOutputToArray
from datetime import timedelta as delta
from progressbar import ProgressBar
import numpy as np
from os import path
import math


def set_ofes_fieldset(snapshots):
    ufiles = [path.join(path.dirname(__file__), "ofesdata", "uvel{:05d}.nc".format(s)) for s in snapshots]
    vfiles = [path.join(path.dirname(__file__), "ofesdata", "vvel{:05d}.nc".format(s)) for s in snapshots]
    tfiles = [path.join(path.dirname(__file__), "ofesdata", "temp{:05d}.nc".format(s)) for s in snapshots]
    filenames = {'U': ufiles, 'V': vfiles, 'temp': tfiles}
    variables = {'U': 'uvel', 'V': 'vvel', 'temp': 'temp'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time', 'depth': 'lev'}

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    fieldset.U.data /= 100.  # convert from cm/s to m/s
    fieldset.V.data /= 100.  # convert from cm/s to m/s
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
    snapshots = range(3165, 3289)
    fieldset = set_ofes_fieldset(snapshots[-4:-1])
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

    pbar = ProgressBar()
    for s in pbar(range(len(snapshots)-5, -1, -1)):
        pset.execute(kernels, starttime=pset[0].time, runtime=delta(days=3),
                     dt=delta(minutes=-5), interval=delta(days=-1),
                     recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})

        pset.add(ForamParticle(lon=corelon, lat=corelat, depth=coredepth, fieldset=fieldset))
        pfile.write(pset, pset[0].time)
        fieldset.advancetime(set_ofes_fieldset([snapshots[s]]))


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
    m = Basemap(projection='merc', llcrnrlat=-40, urcrnrlat=-27.5, llcrnrlon=10, urcrnrlon=32.5, resolution='h')
    m.drawcoastlines()
    m.fillcontinents(color='burlywood')
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
