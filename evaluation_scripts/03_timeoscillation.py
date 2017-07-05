from parcels import FieldSet, ParticleSet, JITParticle
from parcels import AdvectionRK4
from netCDF4 import Dataset
import numpy as np
from datetime import timedelta as delta
import matplotlib.pyplot as plt


omega = 2 * np.pi / 86400.
A = 0.1


def timeoscillation_fieldset(xdim, ydim):
    time = np.arange(0., 4. * 86400., 60.*5., dtype=np.float64)
    lon = np.linspace(-20000, 20000, xdim, dtype=np.float32)
    lat = np.linspace(0, 40000, ydim, dtype=np.float32)

    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    for t in range(time.size):
        U[:, :, t] = A * np.cos(omega*time[t])
        V[:, :, t] = A

    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat, 'time': time}
    return FieldSet.from_data(data, dimensions, mesh='flat')


def run_timeoscillation(fieldset, outfilename):

    pset = ParticleSet.from_line(fieldset, pclass=JITParticle, size=20,
                                 start=(-10000, 0), finish=(10000, 0))

    outfile = pset.ParticleFile(name=outfilename)
    pset.execute(AdvectionRK4, runtime=delta(days=4), dt=delta(minutes=5),
                 interval=delta(hours=3), output_file=outfile)


def make_plot(outfile):
    pfile = Dataset(outfile + ".nc", 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']
    time = pfile.variables['time']

    max_xerr = 0
    max_yerr = 0
    for i, x in enumerate(lon[:, 0]):
        x_true = (x + A / omega * np.sin(omega * time[0, :]))
        y_true = (time[0, :] * A)
        max_xerr = max((max_xerr, max(abs(x_true - lon[i, :]))))
        max_yerr = max((max_yerr, max(abs(y_true - lat[i, :]))))
        plt.plot(x_true / 1000, y_true / 1000, 'k--', linewidth=1.0)
    print max_xerr, max_yerr
    plt.plot(np.transpose(lon) / 1000, np.transpose(lat) / 1000, '.-',
             linewidth=0.5)
    plt.xlabel('Zonal distance [km]')
    plt.ylabel('Meridional distance [km]')
    plt.title('(c) Advection due to a time-oscillating zonal flow')
    plt.show()


if __name__ == "__main__":
    outfilename = "03_timeoscillation"
    fieldset = timeoscillation_fieldset(2, 2)
    run_timeoscillation(fieldset, outfilename)
    make_plot(outfilename)
