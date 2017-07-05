from parcels import FieldSet, ParticleSet, JITParticle
from parcels import AdvectionRK4
from netCDF4 import Dataset
import numpy as np
from datetime import timedelta as delta
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def longitudinalshear_fieldset(xdim, ydim):
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    data = {'U': np.ones((lon.size, lat.size), dtype=np.float32),
            'V': np.zeros((lon.size, lat.size), dtype=np.float32)}
    dimensions = {'lon': lon, 'lat': lat}
    return FieldSet.from_data(data, dimensions, mesh='spherical')


def run_longitudinalshear(fieldset, npart, outfilename):

    pset = ParticleSet.from_line(fieldset, size=npart, pclass=JITParticle,
                                 start=(0, -30), finish=(0, 60))

    outfile = pset.ParticleFile(name=outfilename)
    pset.execute(AdvectionRK4, runtime=delta(days=57), dt=delta(minutes=5),
                 interval=delta(days=1), output_file=outfile)


def make_plot(outfile):
    pfile = Dataset(outfile + ".nc", 'r')
    lon = pfile.variables['lon']
    lat = pfile.variables['lat']

    fig, ax1 = plt.subplots()
    plt.xlabel('Longitude [degrees]')
    plt.ylabel('Latitude [degrees]')
    plt.title('(b) Longitudinal shear flow')

    ax2 = fig.add_axes([0.52, 0.15, 0.4, 0.4])

    ax1.plot(np.transpose(lon), np.transpose(lat), '.-', linewidth=0.5)

    m = Basemap(projection='ortho', lat_0=45, lon_0=10, resolution='l', ax=ax2)
    m.drawmeridians(np.arange(0, 360, 45))
    m.drawparallels(np.arange(-90, 90, 30))
    xs, ys = m(np.transpose(lon), np.transpose(lat))
    m.plot(xs, ys)

    plt.show()


if __name__ == "__main__":
    outfilename = "02_longitudinalshear"
    fieldset = longitudinalshear_fieldset(360, 180)
    run_longitudinalshear(fieldset, 31, outfilename)
    make_plot(outfilename)
