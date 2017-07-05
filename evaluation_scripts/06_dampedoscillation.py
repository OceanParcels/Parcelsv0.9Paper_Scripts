from parcels import FieldSet, ParticleSet, JITParticle
from parcels import AdvectionRK4
from netCDF4 import Dataset
import numpy as np
from datetime import timedelta as delta
import matplotlib.pyplot as plt


# Define some constants.
u_g = .04  # Geostrophic current
u_0 = .3  # Initial speed in x dirrection. v_0 = 0
gamma = 1./delta(days=2.89).total_seconds()  # Dissipitave effects due to viscousity.
gamma_g = 1./delta(days=28.9).total_seconds()
f = 1.e-4  # Coriolis parameter.


def dampedoscillation_fieldset(xdim, ydim):
    """Simulate an ocean that accelerates subject to Coriolis force
    and dissipative effects, upon which a geostrophic current is
    superimposed.

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    time = np.arange(0., 4. * 86400., 60.*5., dtype=np.float64)
    lon = np.linspace(-20000, 20000, xdim, dtype=np.float32)
    lat = np.linspace(-20000, 20000, ydim, dtype=np.float32)

    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    for t in range(time.size):
        U[:, :, t] = u_g*np.exp(-gamma_g*time[t]) + (u_0-u_g)*np.exp(-gamma*time[t])*np.cos(f*time[t])
        V[:, :, t] = -(u_0-u_g)*np.exp(-gamma*time[t])*np.sin(f*time[t])

    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat, 'time': time}
    return FieldSet.from_data(data, dimensions, mesh='flat')


def true_values(t, x_0, y_0):  # Calculate the expected values for particles at the endtime, given their start location.
    x = x_0 + (u_g/gamma_g)*(1-np.exp(-gamma_g*t)) + f*((u_0-u_g)/(f**2 + gamma**2))*((gamma/f) + np.exp(-gamma*t)*(np.sin(f*t) - (gamma/f)*np.cos(f*t)))
    y = y_0 - ((u_0-u_g)/(f**2+gamma**2))*f*(1 - np.exp(-gamma*t)*(np.cos(f*t) + (gamma/f)*np.sin(f*t)))
    return x, y


def run_dampedoscillation(fieldset, start_lon, start_lat, outfilename):
    pset = ParticleSet(fieldset, pclass=JITParticle, lon=start_lon, lat=start_lat)

    outfile = pset.ParticleFile(name=outfilename)
    pset.execute(AdvectionRK4, runtime=delta(days=4), dt=delta(minutes=5),
                 interval=delta(hours=1), output_file=outfile)

    x, y = true_values(pset[0].time, start_lon, start_lat)
    print (x - pset[0].lon)/1000, (y - pset[0].lat)/1000


def make_plot(start_lon, start_lat, outfile):
    pfile = Dataset(outfile + ".nc", 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']

    truetimes = range(0, 4 * 86400, 360)
    x = np.zeros((2, len(truetimes)))
    for ind, t in enumerate(truetimes):
        x[:, ind] = true_values(t, start_lon, start_lat)
    plt.plot(x[0, :] / 1000., x[1, :] / 1000., 'k--')
    plt.plot(np.transpose(lon) / 1000, np.transpose(lat) / 1000, '.-', linewidth=0.5)
    plt.xlabel('Zonal distance [km]')
    plt.ylabel('Meridional distance [km]')
    plt.title('(f) Damped inertial oscillation on a geostrophic flow')
    plt.show()


if __name__ == "__main__":
    outfilename = "06_dampedoscillation"
    start_lon = [0.]  # Define the start longitude and latitude for the particle.
    start_lat = [0.]
    fieldset = dampedoscillation_fieldset(2, 2)
    run_dampedoscillation(fieldset, start_lon, start_lat, outfilename)
    make_plot(start_lon, start_lat, outfilename)
