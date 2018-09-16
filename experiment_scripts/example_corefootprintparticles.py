from parcels import (FieldSet, ParticleSet, JITParticle, AdvectionRK4_3D,
                     ErrorCode, ParticleFile, Variable)
from datetime import timedelta as delta
import numpy as np
from os import path
import math


def set_ofes_fieldset(snapshots):
    ufiles = [path.join(path.dirname(__file__), "ofesdata", "uvel{:05d}.nc".format(s)) for s in snapshots]
    vfiles = [path.join(path.dirname(__file__), "ofesdata", "vvel{:05d}.nc".format(s)) for s in snapshots]
    wfiles = [path.join(path.dirname(__file__), "ofesdata", "wvel{:05d}.nc".format(s)) for s in snapshots]
    tfiles = [path.join(path.dirname(__file__), "ofesdata", "temp{:05d}.nc".format(s)) for s in snapshots]
    filenames = {'U': ufiles, 'V': vfiles, 'W': wfiles, 'temp': tfiles}
    variables = {'U': 'uvel', 'V': 'vvel', 'W': 'wvel', 'temp': 'temp'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time', 'depth': 'lev'}

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    fieldset.U.set_scaling_factor(0.01)  # convert from cm/s to m/s
    fieldset.V.set_scaling_factor(0.01)  # convert from cm/s to m/s
    fieldset.W.set_scaling_factor(0.01)  # convert from cm/s to m/s
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


def DeleteParticle(particle, fieldset, time, dt):
    particle.delete()


def run_corefootprintparticles(outfile):
    snapshots = range(3165, 3288)
    fieldset = set_ofes_fieldset(snapshots)
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
                       depth=coredepth, time=fieldset.U.time[-1],
                       repeatdt=delta(days=3))  # the new argument 'repeatdt' means no need to call pset.add() anymore in for-loop
    pfile = ParticleFile(outfile, pset, outputdt=delta(days=1))  # `interval` argument has changed to `outputdt`

    kernels = pset.Kernel(AdvectionRK4_3D) + Sink + SampleTemp + Age

    pset.execute(kernels, dt=delta(minutes=-5), output_file=pfile,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})


def make_plot(trajfile):
    from netCDF4 import Dataset
    from parcels.plotting import create_parcelsfig_axis

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

    plt, fig, ax, cartopy = create_parcelsfig_axis(spherical=True)

    sinks = np.where(T.z > 50.)
    dwell = np.where(T.z == 50.)
    plt.scatter(T.lon[dwell], T.lat[dwell], c=T.temp[dwell], s=5)
    cbar = plt.colorbar()
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_xlabel('[$^\circ$C]')

    plt.scatter(T.lon[sinks], T.lat[sinks], c='k', s=5)
    plt.plot(T.lon[0, 0], T.lat[0, 0], 'om')
    plt.show()


outfile = "corefootprint_particles"
run_corefootprintparticles(outfile)
make_plot(outfile+".nc")
