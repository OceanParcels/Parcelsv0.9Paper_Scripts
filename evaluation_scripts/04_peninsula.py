from parcels import FieldSet, ParticleSet, JITParticle, Variable
from parcels import AdvectionRK4
from netCDF4 import Dataset
import numpy as np
from datetime import timedelta as delta
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def peninsula_fieldset(xdim, ydim):
    """Construct a fieldset encapsulating the flow field around an
    idealised peninsula.

    :param xdim: Horizontal dimension of the generated fieldset
    :param xdim: Vertical dimension of the generated fieldset

    The original test description can be found in Fig. 2.2.3 in:
    North, E. W., Gallego, A., Petitgas, P. (Eds). 2009. Manual of
    recommended practices for modelling physical - biological
    interactions during fish early life.
    ICES Cooperative Research Report No. 295. 111 pp.
    http://archimer.ifremer.fr/doc/00157/26792/24888.pdf

    Note that the problem is defined on an A-grid while NEMO
    normally returns C-grids. However, to avoid accuracy
    problems with interpolation from A-grid to C-grid, we
    return NetCDF files that are on an A-grid.
    """
    # Generate the original test setup on A-grid in km
    dx = 100. / xdim / 2.
    dy = 50. / ydim / 2.
    La = np.linspace(dx, 100.-dx, xdim, dtype=np.float32)
    Wa = np.linspace(dy, 50.-dy, ydim, dtype=np.float32)

    u0 = 1
    x0 = 50.
    R = 0.32 * 50.

    # Create the fields
    x, y = np.meshgrid(La, Wa, sparse=True, indexing='ij')
    Psi = u0*R**2*y/((x-x0)**2+y**2)-u0*y
    U = u0-u0*R**2*((x-x0)**2-y**2)/(((x-x0)**2+y**2)**2)
    V = -2*u0*R**2*((x-x0)*y)/(((x-x0)**2+y**2)**2)

    # Set land points to NaN
    I = Psi >= 0.
    U[I] = np.nan
    V[I] = np.nan

    # Convert from km to lat/lon
    lon = La / 1.852 / 60.
    lat = Wa / 1.852 / 60.

    data = {'U': U, 'V': V, 'Psi': Psi}
    dimensions = {'lon': lon, 'lat': lat}
    return FieldSet.from_data(data, dimensions)


def UpdatePsi(particle, fieldset, time, dt):
    particle.psi = fieldset.Psi[time, particle.lon, particle.lat, particle.depth]


def run_pensinsula(fieldset, npart, outfilename):

    class MyParticle(JITParticle):
        psi = Variable('psi', dtype=np.float32, initial=fieldset.Psi)

    # Initialise particles
    x = 3. * (1. / 1.852 / 60)
    y = (fieldset.U.lat[0] + x, fieldset.U.lat[-1] - x)
    pset = ParticleSet.from_line(fieldset, size=npart, pclass=MyParticle,
                                 start=(x, y[0]), finish=(x, y[1]))

    # Advect the particles for 24h
    outfile = pset.ParticleFile(name=outfilename)
    pset.execute(AdvectionRK4 + pset.Kernel(UpdatePsi),
                 runtime=delta(hours=24), dt=delta(minutes=5),
                 interval=delta(hours=1), output_file=outfile)


def make_plot(fieldset, outfile):
    pfile = Dataset(outfile + ".nc", 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    psi = np.ma.filled(pfile.variables['psi'])
    print np.max([abs(psi[i, :] - psi[i, 0]) for i in range(psi.shape[0])])

    levels = np.insert(np.flip(psi[:, 1], 0), 0, [-50])
    plt.gca().patch.set_color('.25')
    cm = LinearSegmentedColormap.from_list('test', [(1, 1, 1), (1, 1, 1)], N=1)
    plt.contourf(fieldset.Psi.lon, fieldset.Psi.lat, fieldset.Psi.data[0, :, :],
                 levels=[-50, 0], cmap=cm)
    plt.contour(fieldset.Psi.lon, fieldset.Psi.lat, fieldset.Psi.data[0, :, :],
                levels=levels, colors=('k',), linewidths=(1,))
    plt.plot(np.transpose(lon), np.transpose(lat), '.-', linewidth=0.5)
    plt.xlabel('Longitude [degrees]')
    plt.ylabel('Latitude [degrees]')
    plt.title('(d) Steady-state flow around a peninsula')
    plt.show()


if __name__ == "__main__":
    outfilename = "04_peninsula_particles"
    fieldset = peninsula_fieldset(100, 50)
    run_pensinsula(fieldset, 20, outfilename)
    make_plot(fieldset, outfilename)
