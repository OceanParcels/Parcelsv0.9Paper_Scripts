from parcels import FieldSet, ParticleSet, JITParticle, Variable
from parcels import AdvectionRK4
from netCDF4 import Dataset
import numpy as np
from datetime import timedelta as delta
import math
import matplotlib.pyplot as plt


def stommelgyre_fieldset(xdim, ydim):
    """Simulate a periodic current along a western boundary, with significantly
    larger velocities along the western edge than the rest of the region

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    # Some constants
    A = 100
    eps = 0.05
    a = b = 10000

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, xdim, dtype=np.float32)
    lat = np.linspace(0, b, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional), and Psi (streamfunction) all on A-grid
    U = np.zeros((lon.size, lat.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size), dtype=np.float32)
    Psi = np.zeros((lon.size, lat.size), dtype=np.float32)

    [x, y] = np.mgrid[:lon.size, :lat.size]
    l1 = (-1 + math.sqrt(1 + 4 * math.pi**2 * eps**2)) / (2 * eps)
    l2 = (-1 - math.sqrt(1 + 4 * math.pi**2 * eps**2)) / (2 * eps)
    c1 = (1 - math.exp(l2)) / (math.exp(l2) - math.exp(l1))
    c2 = -(1 + c1)
    for i in range(lon.size):
        for j in range(lat.size):
            xi = lon[i] / a
            yi = lat[j] / b
            Psi[i, j] = A * (c1*math.exp(l1*xi) + c2*math.exp(l2*xi) + 1) * math.sin(math.pi * yi)
    for i in range(lon.size-2):
        for j in range(lat.size):
            V[i+1, j] = (Psi[i+2, j] - Psi[i, j]) / (2 * a / xdim)
    for i in range(lon.size):
        for j in range(lat.size-2):
            U[i, j+1] = -(Psi[i, j+2] - Psi[i, j]) / (2 * b / ydim)

    data = {'U': U, 'V': V, 'Psi': Psi}
    dimensions = {'lon': lon, 'lat': lat}
    return FieldSet.from_data(data, dimensions, mesh='flat')


def UpdatePsi(particle, fieldset, time, dt):
    particle.psi = fieldset.Psi[time, particle.lon, particle.lat, particle.depth]


def run_stommelgyre(fieldset, outfilename):
    class MyParticle(JITParticle):
        psi = Variable('psi', dtype=np.float32, initial=fieldset.Psi)

    pset = ParticleSet.from_line(fieldset, size=4, pclass=MyParticle,
                                 start=(100, 5000), finish=(1000, 5000))

    outfile = pset.ParticleFile(name=outfilename)
    pset.execute(AdvectionRK4 + pset.Kernel(UpdatePsi), runtime=delta(days=50),
                 dt=delta(minutes=5), interval=delta(hours=24), output_file=outfile)


def make_plot(fieldset, outfile):
    pfile = Dataset(outfile + ".nc", 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    psi = pfile.variables['psi']
    print np.max([abs(psi[i, :] - psi[i, 0]) for i in range(psi.shape[0])])

    plt.contour(fieldset.Psi.lon, fieldset.Psi.lat, fieldset.Psi.data[0, :, :],
                levels=psi[:, 0], colors=('k',), linewidths=(1,))
    plt.plot(np.transpose(lon), np.transpose(lat), '.-', linewidth=0.5)
    plt.xlabel('Zonal distance [km]')
    plt.ylabel('Meridional distance [km]')
    plt.show()


if __name__ == "__main__":
    outfilename = "05_stommelgyre_particles"
    fieldset = stommelgyre_fieldset(200, 200)
    run_stommelgyre(fieldset, outfilename)
    make_plot(fieldset, outfilename)
