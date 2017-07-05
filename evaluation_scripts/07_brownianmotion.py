from parcels import FieldSet, ParticleSet, JITParticle, random
from netCDF4 import Dataset
import numpy as np
import math
from datetime import timedelta as delta
import matplotlib.pyplot as plt


def two_dim_brownian_flat(particle, fieldset, time, dt):
    # Kernel for simple Brownian particle diffusion in zonal and meridional direction.

    particle.lat += random.normalvariate(0, 1)*math.sqrt(2*dt*fieldset.Kh_meridional)
    particle.lon += random.normalvariate(0, 1)*math.sqrt(2*dt*fieldset.Kh_zonal)


def brownian_fieldset(xdim=200, ydim=200):  # Define a flat fieldset of zeros
    dimensions = {'lon': np.linspace(-30000, 30000, xdim, dtype=np.float32),
                  'lat': np.linspace(-30000, 30000, ydim, dtype=np.float32)}

    data = {'U': np.zeros((xdim, ydim), dtype=np.float32),
            'V': np.zeros((xdim, ydim), dtype=np.float32)}

    return FieldSet.from_data(data, dimensions, mesh='flat')


def run_brownian(fieldset, npart, outfilename):

    # Set diffusion constants.
    fieldset.Kh_meridional = 100.
    fieldset.Kh_zonal = 100.

    # Set random seed
    random.seed(123456)

    pset = ParticleSet.from_line(fieldset=fieldset, size=npart, pclass=JITParticle,
                                 start=(0., 0.), finish=(0., 0.))

    pset.execute(two_dim_brownian_flat, runtime=delta(days=1), dt=delta(minutes=5))
    pset.ParticleFile(name=outfilename).write(pset, pset[0].time)


def make_plot(outfile):
    pfile = Dataset(outfile + ".nc", 'r')
    lon = np.transpose(pfile.variables['lon'][:, -1]) / 1000.
    lat = np.transpose(pfile.variables['lat'][:, -1]) / 1000.

    binedges = [np.arange(-15, 15, 0.5), np.arange(-15, 15, 0.5)]
    plt.hist2d(lon, lat, binedges, normed=True)
    plt.colorbar()

    X, Y = np.meshgrid(np.linspace(-15, 15, 50), np.linspace(-15, 15, 50))
    mu, sigma = 0, np.sqrt(2 * 100 * 86400) / 1000
    G = np.exp(-((X - mu) ** 2 + (Y - mu) ** 2) / 2.0 / sigma ** 2) / (sigma * math.sqrt(2 * math.pi))
    plt.contour(X, Y, G)

    plt.xlabel('Zonal distance [km]')
    plt.ylabel('Meridional distance [km]')
    plt.title('(g) Brownian motion with a uniform $K_h$')
    plt.show()


if __name__ == "__main__":
    outfilename = "07_brownianmotion"
    fieldset = brownian_fieldset()
    run_brownian(fieldset, 100000, outfilename)
    make_plot(outfilename)
