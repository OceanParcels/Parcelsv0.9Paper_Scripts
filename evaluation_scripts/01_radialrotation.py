from parcels import FieldSet, ParticleSet, JITParticle
from parcels import AdvectionRK4
from netCDF4 import Dataset
import numpy as np
from datetime import timedelta as delta
import math
import matplotlib.pyplot as plt


def radialrotation_fieldset(xdim, ydim):
    # Coordinates of the test fieldset (on A-grid in deg)
    a = b = 20000  # domain size
    lon = np.linspace(-a/2, a/2, xdim, dtype=np.float32)
    lat = np.linspace(-b/2, b/2, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional) on A-grid
    U = np.zeros((lon.size, lat.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size), dtype=np.float32)
    R = np.zeros((lon.size, lat.size), dtype=np.float32)

    omega = 2 * math.pi / delta(days=1).total_seconds()
    for i in range(lon.size):
        for j in range(lat.size):
            r = np.sqrt(lon[i]**2 + lat[j]**2)
            phi = np.arctan2(lat[j], lon[i])
            U[j, i] = -omega * r * math.sin(phi)
            V[j, i] = omega * r * math.cos(phi)
            R[j, i] = r

    data = {'U': U, 'V': V, 'P': R}
    dimensions = {'lon': lon, 'lat': lat}
    return FieldSet.from_data(data, dimensions, mesh='flat')


def run_radialrotation(fieldset, outfilename):
    # Define a ParticleSet
    pset = ParticleSet.from_line(fieldset, size=4, pclass=JITParticle,
                                 start=(0, 1000), finish=(0, 4000))

    # Advect the particles for 24h
    outfile = pset.ParticleFile(name=outfilename, outputdt=delta(hours=1))
    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(minutes=5),
                 output_file=outfile)


def make_plot(fieldset, outfile):
    pfile = Dataset(outfile + ".nc", 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    print np.max(abs(lat[:, -1] - lat[:, 0]))
    print np.max(abs(lon[:, -1]))

    plt.contour(fieldset.P.lon/1000, fieldset.P.lat/1000, -fieldset.P.data[0, :, :],
                levels=np.flip(-lat[:, 0], 0), colors=('k',), linewidths=(1,))
    plt.plot(np.transpose(lon)/1000, np.transpose(lat)/1000, '.-', linewidth=0.5)
    plt.xlabel('Zonal distance [km]')
    plt.ylabel('Meridional distance [km]')
    plt.title('(a) Radial rotation with known period')
    plt.axis((-5, 5, -5, 5))
    plt.show()


if __name__ == "__main__":
    outfilename = "01_radialrotation"
    fieldset = radialrotation_fieldset(200, 200)
    run_radialrotation(fieldset, outfilename)
    make_plot(fieldset, outfilename)
