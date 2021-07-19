import xarray
import numpy as np
import netCDF4 as nc4
import py3dep
from shapely.geometry import Polygon
import rioxarray



def main(flight_nc_file,mintime,maxtime,datenow,fov,buff):


    df = nc4.Dataset(flight_nc_file,'r')
    time = df['Time'][:]
    lon = df['LON'][:]
    lat = df['LAT'][:]
    alt = df['ALT'][:,:]
    roll = df['ROLL'][:,:]
    pitch = df['PITCH'][:,:]
    df.close()

    mintime = (mintime - datenow).seconds 
    maxtime = (maxtime - datenow).seconds 

    mintime= int(np.interp(mintime,time,x))
    maxtime= int(np.interp(maxtime,time,x))

    
    minlon = min(lon[mintime ] ,lon[maxtime ])
    maxlon = max(lon[mintime ] ,lon[maxtime ] )
    minlat = min(lat[mintime ] ,lat[maxtime ])
    maxlat = max(lat[mintime ] ,lat[maxtime ]) 
    
    max_roll = max(np.nanmax(roll[mintime:maxtime,:]) * np.pi/180,np.nanmin(roll[mintime:maxtime,:]) * np.pi/180) 
    min_roll = min(np.nanmax(roll[mintime:maxtime,:]) * np.pi/180,np.nanmin(roll[mintime:maxtime,:]) * np.pi/180) 
 
    max_pitch = max(np.nanmax(pitch[mintime:maxtime,:]) * np.pi/180,np.nanmin(pitch[mintime:maxtime,:]) * np.pi/180)
    min_pitch = min(np.nanmax(pitch[mintime:maxtime,:]) * np.pi/180,np.nanmin(pitch[mintime:maxtime,:]) * np.pi/180 )
    fov = fov * np.pi / 180
    buff = 0.00001*buff / 1.11 
    maxalt = np.nanmax(alt[mintime:maxtime,:])

    max_angle = max(abs(max_roll),abs(max_pitch))
    min_angle = max((min_roll),(min_pitch))

    xmax = 0.00001*(maxalt * np.cos(abs(max_angle))  )  / 1.11
    xmin = 0.00001*(maxalt * np.cos(abs(min_angle))  )  / 1.11
    emax = 0.00001*(maxalt * np.cos(abs(max_angle)) * np.sin(fov/2) / (np.sin(abs(max_angle) + 0.5*np.pi)) ) / 1.11
    emin = 0.00001*(maxalt * np.cos(abs(min_angle)) * np.sin(fov/2) / (np.sin(abs(min_angle) + 0.5*np.pi)) ) / 1.11
  
    lon_min = minlon - xmin - emin - buff 
    lat_min = minlat - xmin - emin - buff 
    lon_max = maxlon + xmax + emax + buff 
    lat_max = maxlat + xmax + emax + buff 
   
    cord = [(lon_min,lat_min),(lon_min,lat_max),(lon_max,lat_max),(lon_max,lat_min)]
    geom = Polygon(cord)

    dem = py3dep.get_map("DEM", geom, resolution=10, geo_crs="epsg:4326", crs="epsg:4326")
    dem.to_netcdf('dem.nc')

    df = xarray.open_dataset("dem.nc")
    
    elev = df.variables['elevation']
    
    x = df.variables['elevation'].attrs['transform']
    
    y = list(x)
    
    trans = rasterio.transform.Affine(y[0],y[1],y[2],y[3],y[4],y[5])
    
    new_dataset = rasterio.open('dem.tiff', 'w', driver='GTiff',
                                height = elev.shape[0], width = elev.shape[1],
                                count=1, dtype=str(elev.dtype),
                                crs='+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0',
                                transform=trans)
    
    new_dataset.write(elev, 1)
    new_dataset.close()

    return('dem.tiff')
