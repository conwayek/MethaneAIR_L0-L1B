import numpy as np
import netCDF4 as nc4
import os
import pysplat
from skimage.measure import block_reduce

#o2path = '/n/wofsy_lab/econway/MethaneAIR/level1/RF01/EKC_V3/CH4_NATIVE/' 
#o2path = os.getcwd()


#list_files = []
#for file in os.listdir(o2path):
#    if file.endswith(".nc"):
#        list_files.append(os.path.join(o2path, file))
#
#nfiles = len(list_files)

#for i in range(nfiles):


def main(list_files,l1AggDataDir):
    name = str(list_files).split('.nc')[0]
    name = name.split('/')[-1]
    l1_outfile = os.path.join(l1AggDataDir,str(name)+'.nc')
    f = nc4.Dataset(list_files,'r')
    y1 = f.groups['Band1']
    y2 = f.groups['Geolocation']

    wvl = y1['Wavelength'][:,:,:] 
    rad_err = y1['RadianceUncertainty'][:,:,:]
    rad_flags = y1['RadianceFlag'][:,:,:]
    rad = y1['Radiance'][:,:,:] 

    corner_lon = y2['CornerLongitude'][:,:,:] 
    corner_lat = y2['CornerLatitude'][:,:,:] 
    

    lon = y2['Longitude'][:,:] 
    lat = y2['Latitude'][:,:] 
    sza = y2['SolarZenithAngle'][:,:] 
    saa = y2['SolarAzimuthAngle'][:,:] 
    aza = y2['RelativeAzimuthAngle'][:,:] 
    vza = y2['ViewingZenithAngle'][:,:] 
    vaa = y2['ViewingAzimuthAngle'][:,:] 
    surfalt = y2['SurfaceAltitude'][:,:] 
    ac_lon = y2['Aircraft_Longitude'][:]
    ac_lat = y2['Aircraft_Latitude'][:]
    ac_alt_surf = y2['Aircraft_AltitudeAboveSurface'][:]
    ac_surf_alt = y2['Aircraft_SurfaceAltitude'][:]
    ac_pix_bore = y2['Aircraft_PixelBore'][:,:,:]
    ac_pos = y2['Aircraft_Pos'][:,:]


    #############################

    obsalt = y2['ObservationAltitude'][:] 
    time = y2['Time'][:] 

    #############################
    # NOW WE WRITE THE NEW FILE TO DESK: CH4 FIRST
    #############################
    xtrk_aggfac = 15
    atrk_aggfac = 3

    norm_1d = block_reduce(np.ones(obsalt.shape),block_size=(atrk_aggfac,),func=np.mean)
    obsalt_new = block_reduce(obsalt,block_size=(atrk_aggfac,),func=np.mean) ; obsalt_new = obsalt_new / norm_1d
    time_new = block_reduce(time,block_size=(atrk_aggfac,),func=np.mean) ; time_new = time_new / norm_1d
    ac_alt_surf_new = block_reduce(ac_alt_surf,block_size=(atrk_aggfac,),func=np.mean) ; ac_alt_surf_new = ac_alt_surf_new / norm_1d
    ac_surf_alt_new = block_reduce(ac_surf_alt,block_size=(atrk_aggfac,),func=np.mean) ; ac_surf_alt_new = ac_surf_alt_new / norm_1d
    ac_lon_new= block_reduce(ac_lon,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lon_new = ac_lon_new / norm_1d
    ac_lat_new= block_reduce(ac_lat,block_size=(atrk_aggfac,),func=np.mean) ;  ac_lat_new = ac_lat_new / norm_1d

    norm_2d = block_reduce(np.ones(lon.shape),block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean)
    lon_new = block_reduce(lon,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lon_new = lon_new / norm_2d
    lat_new = block_reduce(lat,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; lat_new = lat_new / norm_2d
    saa_new=block_reduce(saa,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  saa_new = saa_new / norm_2d
    sza_new=block_reduce(sza,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  sza_new = sza_new / norm_2d
    vza_new=block_reduce(vza,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vza_new = vza_new / norm_2d
    vaa_new=block_reduce(vaa,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  vaa_new = vaa_new / norm_2d
    aza_new=block_reduce(aza,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  aza_new = aza_new / norm_2d
    surfalt_new=block_reduce(surfalt,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  surfalt_new = surfalt_new / norm_2d

    norm_3d = block_reduce(np.ones(ac_pix_bore.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
    ac_pix_bore_new=block_reduce(ac_pix_bore,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ;  ac_pix_bore_new = ac_pix_bore_new / norm_3d

    norm_2d = block_reduce(np.ones(ac_pos.shape),(1,atrk_aggfac),func=np.mean)
    ac_pos_new=block_reduce(ac_pos,block_size=(1,atrk_aggfac),func=np.mean) ;  ac_pos_new = ac_pos_new / norm_2d

    lon_new = lon_new.transpose((1,0))
    lat_new = lat_new.transpose((1,0))
    saa_new=saa_new.transpose((1,0))
    sza_new=sza_new.transpose((1,0))
    vza_new=vza_new.transpose((1,0))
    vaa_new=vaa_new.transpose((1,0))
    aza_new=aza_new.transpose((1,0))
    surfalt_new=surfalt_new.transpose((1,0))
    ac_pos_new=ac_pos_new.transpose((1,0))
    ac_pix_bore_new=ac_pix_bore_new.transpose((2,1,0))




    norm_3d = block_reduce(np.ones(corner_lat.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
    corner_lat_new = block_reduce(corner_lat,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lat_new = corner_lat_new / norm_3d
    corner_lon_new = block_reduce(corner_lon,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; corner_lon_new = corner_lon_new / norm_3d

    corner_lat_new = corner_lat_new.transpose((2,1,0))
    corner_lon_new = corner_lon_new.transpose((2,1,0)) 

    norm_3d = block_reduce(np.ones(rad.shape),block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
    valpix = np.zeros(rad.shape)
    idv = np.logical_and(np.isfinite(rad),rad>0.0)
    valpix[idv] = 1.0
    valpix_agg = block_reduce(valpix,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean)
    valpix_agg = valpix_agg / norm_3d

    rad_new = block_reduce(rad,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_new = rad_new / norm_3d
    rad_err_new = block_reduce(rad_err,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_err_new = rad_err_new / norm_3d 
    rad_err_new = rad_err_new / np.sqrt(xtrk_aggfac*atrk_aggfac)
    rad_new[valpix_agg <0.99999999999999999999] = np.nan
    rad_err_new[valpix_agg <0.99999999999999999999] = np.nan

    wvl_new = block_reduce(wvl,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; wvl_new = wvl_new / norm_3d
    rad_flags_new = block_reduce(rad_flags,block_size=(1,atrk_aggfac, xtrk_aggfac),func=np.mean) ; rad_flags_new = rad_flags_new / norm_3d


    rad_new = rad_new.transpose((2,1,0))
    rad_err_new = rad_err_new.transpose((2,1,0))
    wvl_new = wvl_new.transpose((2,1,0))
    rad_flags_new = rad_flags_new.transpose((2,1,0))


    l1 = pysplat.level1(l1_outfile,lon_new,lat_new,obsalt_new,time_new,ac_lon_new,ac_lat_new,ac_pos_new,ac_surf_alt_new,ac_alt_surf_new,ac_pix_bore_new,optbenchT=None,clon=corner_lon_new,clat=corner_lat_new)
    l1.set_2d_geofield('SurfaceAltitude', surfalt_new)
    l1.set_2d_geofield('SolarZenithAngle', sza_new)
    l1.set_2d_geofield('SolarAzimuthAngle', saa_new)
    l1.set_2d_geofield('ViewingZenithAngle', vza_new)
    l1.set_2d_geofield('ViewingAzimuthAngle', vaa_new)
    l1.set_2d_geofield('RelativeAzimuthAngle', aza_new)
    l1.add_radiance_band(wvl_new,rad_new,rad_err=rad_err_new,rad_flag=rad_flags_new)
    l1.close()


    f.close()

