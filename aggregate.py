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


def main(list_files,l1AggDataDir,xfac,yfac,av=False):

    xfac = int(xfac)
    yfac = int(yfac)

    name = str(list_files).split('.nc')[0]
    name = name.split('/')[-1]
    l1_outfile = os.path.join(l1AggDataDir,str(name)+'.nc')
    f = nc4.Dataset(list_files,'r')
    y1 = f.groups['Band1']
    y2 = f.groups['Geolocation']
    if(av==False):
        y3 = f.groups['SupportingData']
 
        akaze_used = y3['AkazeUsed'][:]
        avionics_used = y3['AvionicsUsed'][:]
        optimized_used = y3['OptimizedUsed'][:]
        
        ac_roll = y3['AircraftRoll'][:] 
        ac_heading = y3['AircraftHeading'][:] 
        ac_pitch = y3['AircraftPitch'][:] 
        if(optimized_used == 1):
            akaze_lon = y3['AkazeLongitude'][:,:] 
            akaze_lat = y3['AkazeLatitude'][:,:] 
            akaze_msi = y3['AkazeMSIImage'][:,:] 
            akaze_clon = y3['AkazeCornerLongitude'][:,:,:] 
            akaze_clat = y3['AkazeCornerLatitude'][:,:,:] 
            akaze_surfalt = y3['AkazeSurfaceAltitude'][:,:] 
            akaze_Distance_Akaze_Reprojected = y3['DistanceAkazeReprojected'][:,:] 
            akaze_Optimization_Convergence_Fail = y3['OptimizationConvergenceFail'][:] 
            akaze_Reprojection_Fit_Flag = y3['ReprojectionFitFlag'][:] 
            # Add avionics variables
            av_lon = y3['AvionicsLongitude'][:,:] 
            av_lat = y3['AvionicsLatitude'][:,:] 
            av_clon = y3['AvionicsCornerLongitude'][:,:,:] 
            av_clat = y3['AvionicsCornerLatitude'][:,:,:] 
            av_sza = y3['AvionicsSolarZenithAngle'][:,:] 
            av_saa = y3['AvionicsSolarAzimuthAngle'][:,:] 
            av_aza = y3['AvionicsRelativeAzimuthAngle'][:,:] 
            av_vza = y3['AvionicsViewingZenithAngle'][:,:] 
            av_vaa = y3['AvionicsViewingAzimuthAngle'][:,:] 
            av_surfalt = y3['AvionicsSurfaceAltitude'][:,:] 
            av_ac_lon = y3['AvionicsAircraftLongitude'][:]
            av_ac_lat = y3['AvionicsAircraftLatitude'][:]
            av_ac_alt_surf = y3['AvionicsAircraftAltitudeAboveSurface'][:]
            av_ac_surf_alt = y3['AvionicsAircraftSurfaceAltitude'][:]
            av_ac_pix_bore = y3['AvionicsAircraftPixelBore'][:,:,:]
            av_ac_pos = y3['AvionicsAircraftPos'][:,:]
            av_obsalt = y3['AvionicsObservationAltitude'][:] 

        if(akaze_used == 1):
            # Only add avionics variables lon/lat/clon/clat/surfalt
            akaze_msi = y3['AkazeMSIImage'][:,:] 
            av_lon = y3['AvionicsLongitude'][:,:] 
            av_lat = y3['AvionicsLatitude'][:,:] 
            av_clon = y3['AvionicsCornerLongitude'][:,:,:] 
            av_clat = y3['AvionicsCornerLatitude'][:,:,:] 
            av_surfalt = y3['AvionicsSurfaceAltitude'][:,:] 
            # Add all optimized variables
            op_lon = y3['OptimizedLongitude'][:,:] 
            op_lat = y3['OptimizedLatitude'][:,:] 
            op_clon = y3['OptimizedCornerLongitude'][:,:,:] 
            op_clat = y3['OptimizedCornerLatitude'][:,:,:] 
            op_sza = y3['OptimizedSolarZenithAngle'][:,:] 
            op_saa = y3['OptimizedSolarAzimuthAngle'][:,:] 
            op_aza = y3['OptimizedRelativeAzimuthAngle'][:,:] 
            op_vza = y3['OptimizedViewingZenithAngle'][:,:] 
            op_vaa = y3['OptimizedViewingAzimuthAngle'][:,:] 
            op_surfalt = y3['OptimizedSurfaceAltitude'][:,:] 
            op_ac_lon = y3['OptimizedAircraftLongitude'][:]
            op_ac_lat = y3['OptimizedAircraftLatitude'][:]
            op_ac_alt_surf = y3['OptimizedAircraftAltitudeAboveSurface'][:]
            op_ac_surf_alt = y3['OptimizedAircraftSurfaceAltitude'][:]
            op_ac_pix_bore = y3['OptimizedAircraftPixelBore'][:,:,:]
            op_ac_pos = y3['OptimizedAircraftPos'][:,:]
            op_obsalt = y3['OptimizedObservationAltitude'][:] 
            akaze_Distance_Akaze_Reprojected = y3['DistanceAkazeReprojected'][:,:] 
            akaze_Optimization_Convergence_Fail = y3['OptimizationConvergenceFail'][:] 
            akaze_Reprojection_Fit_Flag = y3['ReprojectionFitFlag'][:] 
 

    else:
        akaze_msi = None

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
    ac_lon = y2['AircraftLongitude'][:]
    ac_lat = y2['AircraftLatitude'][:]
    ac_alt_surf = y2['AircraftAltitudeAboveSurface'][:]
    ac_surf_alt = y2['AircraftSurfaceAltitude'][:]
    ac_pix_bore = y2['AircraftPixelBore'][:,:,:]
    ac_pos = y2['AircraftPos'][:,:]


    #############################

    obsalt = y2['ObservationAltitude'][:] 
    time = y2['Time'][:] 

    #############################
    # NOW WE WRITE THE NEW FILE TO DESK: CH4 FIRST
    #############################
    xtrk_aggfac = xfac
    atrk_aggfac = yfac

    norm_1d = block_reduce(np.ones(obsalt.shape),block_size=(atrk_aggfac,),func=np.mean)
    if(av==False):
         ac_roll = block_reduce(ac_roll,block_size=(atrk_aggfac,),func=np.mean) ; ac_roll = ac_roll / norm_1d
         ac_pitch = block_reduce(ac_pitch,block_size=(atrk_aggfac,),func=np.mean) ; ac_pitch = ac_pitch / norm_1d
         ac_heading = block_reduce(ac_heading,block_size=(atrk_aggfac,),func=np.mean) ; ac_heading = ac_heading / norm_1d
         if(akaze_used==1):
            op_ac_lon  = block_reduce(op_ac_lon,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lon = op_ac_lon / norm_1d
            op_ac_lat  = block_reduce(op_ac_lat,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_lat = op_ac_lat / norm_1d
            op_ac_alt_surf  = block_reduce(op_ac_alt_surf,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_alt_surf = op_ac_alt_surf/ norm_1d
            op_ac_surf_alt  = block_reduce(op_ac_surf_alt,block_size=(atrk_aggfac,),func=np.mean) ; op_ac_surf_alt = op_ac_surf_alt/ norm_1d
            op_obsalt  = block_reduce(op_obsalt,block_size=(atrk_aggfac,),func=np.mean) ; op_obsalt = op_obsalt / norm_1d
         elif(optimized_used==1):
            av_ac_lon = block_reduce(av_ac_lon,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lon  =av_ac_lon/ norm_1d
            av_ac_lat = block_reduce(av_ac_lat,block_size=(atrk_aggfac,),func=np.mean) ;   av_ac_lat  =av_ac_lat/ norm_1d
            av_ac_alt_surf = block_reduce(av_ac_alt_surf,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_alt_surf=av_ac_alt_surf    / norm_1d
            av_ac_surf_alt = block_reduce(av_ac_surf_alt,block_size=(atrk_aggfac,),func=np.mean) ; av_ac_surf_alt=av_ac_surf_alt    / norm_1d
            av_obsalt = block_reduce(av_obsalt,block_size=(atrk_aggfac,),func=np.mean) ;av_obsalt =av_obsalt    / norm_1d

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

    if(av==False):
        if(optimized_used==1):
            akaze_lon = block_reduce(akaze_lon,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_lon = akaze_lon / norm_2d
            akaze_lat = block_reduce(akaze_lat,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_lat = akaze_lat / norm_2d
            akaze_msi = block_reduce(akaze_msi,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi = akaze_msi / norm_2d
            akaze_surfalt = block_reduce(akaze_surfalt,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_surfalt = akaze_surfalt / norm_2d 
            akaze_Distance_Akaze_Reprojected = block_reduce(akaze_Distance_Akaze_Reprojected,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected = akaze_Distance_Akaze_Reprojected / norm_2d

            av_lon = block_reduce(av_lon,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon=av_lon   / norm_2d
            av_lat = block_reduce(av_lat,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat=av_lat   / norm_2d
            av_sza = block_reduce(av_sza,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_sza=av_sza   / norm_2d
            av_saa = block_reduce(av_saa,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_saa=av_saa   / norm_2d
            av_aza = block_reduce(av_aza,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_aza=av_aza   / norm_2d
            av_vza = block_reduce(av_vza,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vza=av_vza   / norm_2d
            av_vaa = block_reduce(av_vaa,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_vaa=av_vaa   / norm_2d
            av_surfalt = block_reduce(av_surfalt,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt=av_surfalt/norm_2d 
            av_lon = av_lon.transpose(1,0) 
            av_lat = av_lat.transpose(1,0) 
            av_sza = av_sza.transpose(1,0) 
            av_saa = av_saa.transpose(1,0) 
            av_aza = av_aza.transpose(1,0) 
            av_vza = av_vza.transpose(1,0) 
            av_vaa = av_vaa.transpose(1,0) 
            av_surfalt = av_surfalt.transpose(1,0) 


            norm_3d = block_reduce(np.ones(corner_lat.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
            akaze_clon = block_reduce(akaze_clon,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clon = akaze_clon / norm_3d 
            akaze_clat = block_reduce(akaze_clat,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_clat = akaze_clat / norm_3d 
            

            akaze_lon = akaze_lon.transpose(1,0)
            akaze_lat = akaze_lat.transpose(1,0)
            akaze_msi = akaze_msi.transpose(1,0)
            akaze_surfalt = akaze_surfalt.transpose(1,0)
            akaze_Distance_Akaze_Reprojected = akaze_Distance_Akaze_Reprojected.transpose(1,0)
            akaze_clon = akaze_clon.transpose((2,1,0))
            akaze_clat = akaze_clat.transpose((2,1,0))

            norm_3d = block_reduce(np.ones(ac_pix_bore.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
            av_ac_pix_bore = block_reduce(av_ac_pix_bore,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_ac_pix_bore = av_ac_pix_bore/norm_3d 

            norm_2d = block_reduce(np.ones(ac_pos.shape),(1,atrk_aggfac),func=np.mean)
            av_ac_pos = block_reduce(av_ac_pos,block_size=(1,atrk_aggfac),func=np.mean) ;  av_ac_pos = av_ac_pos / norm_2d 

            av_ac_pos=av_ac_pos.transpose((1,0))
            av_ac_pix_bore=av_ac_pix_bore.transpose((2,1,0))

        if(akaze_used==1):
            akaze_msi = block_reduce(akaze_msi,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_msi = akaze_msi / norm_2d
            akaze_Distance_Akaze_Reprojected = block_reduce(akaze_Distance_Akaze_Reprojected,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; akaze_Distance_Akaze_Reprojected = akaze_Distance_Akaze_Reprojected / norm_2d
            av_lon = block_reduce(av_lon,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lon=av_lon   / norm_2d
            av_lat = block_reduce(av_lat,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_lat=av_lat   / norm_2d
            av_surfalt = block_reduce(av_surfalt,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_surfalt=av_surfalt/norm_2d 
            av_lon = av_lon.transpose(1,0) 
            av_lat = av_lat.transpose(1,0) 
            av_surfalt = av_surfalt.transpose(1,0) 

            op_lon = block_reduce(op_lon,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lon=op_lon  / norm_2d
            op_lat = block_reduce(op_lat,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_lat=op_lat  / norm_2d
            op_sza = block_reduce(op_sza,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_sza=op_sza  / norm_2d
            op_saa = block_reduce(op_saa,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_saa=op_saa  / norm_2d
            op_aza = block_reduce(op_aza,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_aza=op_aza  / norm_2d
            op_vza = block_reduce(op_vza,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vza=op_vza  / norm_2d
            op_vaa = block_reduce(op_vaa,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ;  op_vaa=op_vaa  / norm_2d
            op_surfalt = block_reduce(op_surfalt,block_size=(atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_surfalt =op_surfalt / norm_2d

            op_lon = op_lon.transpose(1,0) 
            op_lat = op_lat.transpose(1,0) 
            op_sza = op_sza.transpose(1,0) 
            op_saa = op_saa.transpose(1,0) 
            op_aza = op_aza.transpose(1,0) 
            op_vza = op_vza.transpose(1,0) 
            op_vaa = op_vaa.transpose(1,0) 
            op_surfalt = op_surfalt.transpose(1,0) 
            akaze_msi = akaze_msi.transpose(1,0)
            akaze_Distance_Akaze_Reprojected = akaze_Distance_Akaze_Reprojected.transpose(1,0)



            norm_3d = block_reduce(np.ones(corner_lat.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)

            av_clon = block_reduce(av_clon,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clon = av_clon / norm_3d 
            av_clat = block_reduce(av_clat,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; av_clat = av_clat / norm_3d 
            op_clon = block_reduce(op_clon,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clon = op_clon / norm_3d 
            op_clat = block_reduce(op_clat,(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_clat = op_clat / norm_3d 
            op_clon = op_clon.transpose((2,1,0))
            op_clat = op_clat.transpose((2,1,0))
            av_clon = av_clon.transpose((2,1,0))
            av_clat = av_clat.transpose((2,1,0))

            norm_3d = block_reduce(np.ones(ac_pix_bore.shape),(1,atrk_aggfac,xtrk_aggfac),func=np.mean)
            op_ac_pix_bore = block_reduce(op_ac_pix_bore,block_size=(1,atrk_aggfac,xtrk_aggfac),func=np.mean) ; op_ac_pix_bore = op_ac_pix_bore/norm_3d 

            norm_2d = block_reduce(np.ones(ac_pos.shape),(1,atrk_aggfac),func=np.mean)
            op_ac_pos = block_reduce(op_ac_pos,block_size=(1,atrk_aggfac),func=np.mean) ;  op_ac_pos = op_ac_pos / norm_2d 

            op_ac_pos=op_ac_pos.transpose((1,0))
            op_ac_pix_bore=op_ac_pix_bore.transpose((2,1,0))

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


    
    l1 = pysplat.level1_AIR(l1_outfile,lon_new,lat_new,obsalt_new,time_new,ac_lon_new,ac_lat_new,ac_pos_new,ac_surf_alt_new,ac_alt_surf_new,ac_pix_bore_new,optbenchT=None,clon=corner_lon_new,clat=corner_lat_new,akaze_msi_image=akaze_msi)
 
    l1.set_2d_geofield('SurfaceAltitude', surfalt_new)
    l1.set_2d_geofield('SolarZenithAngle', sza_new)
    l1.set_2d_geofield('SolarAzimuthAngle', saa_new)
    l1.set_2d_geofield('ViewingZenithAngle', vza_new)
    l1.set_2d_geofield('ViewingAzimuthAngle', vaa_new)
    l1.set_2d_geofield('RelativeAzimuthAngle', aza_new)
    l1.add_radiance_band(wvl_new,rad_new,rad_err=rad_err_new,rad_flag=rad_flags_new)


    if(av==False):
        if(optimized_used==1):
              l1.set_supportfield('AkazeLongitude',akaze_lon)
              l1.set_supportfield('AkazeLatitude',akaze_lat)
              l1.set_supportfield('AkazeSurfaceAltitude',akaze_surfalt)
              l1.set_supportfield('AkazeCornerLatitude',akaze_clat)
              l1.set_supportfield('AkazeCornerLongitude',akaze_clon)

              l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt)
              l1.set_supportfield('AvionicsSolarZenithAngle',av_sza)
              l1.set_supportfield('AvionicsSolarAzimuthAngle',av_saa)
              l1.set_supportfield('AvionicsViewingZenithAngle',av_vza)
              l1.set_supportfield('AvionicsViewingAzimuthAngle',av_vaa)
              l1.set_supportfield('AvionicsRelativeAzimuthAngle',av_aza)
              l1.set_supportfield('AvionicsAircraftLongitude',av_ac_lon)
              l1.set_supportfield('AvionicsAircraftLatitude',av_ac_lat)
              l1.set_supportfield('AvionicsAircraftAltitudeAboveSurface',av_ac_alt_surf)
              l1.set_supportfield('AvionicsAircraftSurfaceAltitude',av_ac_surf_alt)
              l1.set_supportfield('AvionicsAircraftPixelBore',av_ac_pix_bore)
              l1.set_supportfield('AvionicsAircraftPos',av_ac_pos)
              l1.set_supportfield('AvionicsLongitude',av_lon)
              l1.set_supportfield('AvionicsLatitude',av_lat)
              l1.set_supportfield('AvionicsCornerLongitude',av_clon)
              l1.set_supportfield('AvionicsCornerLatitude',av_clat)
              l1.set_supportfield('AvionicsObservationAltitude',av_obsalt)

              l1.set_1d_flag(True,None,None)


        elif(akaze_used==1):
              l1.set_supportfield('OptimizedSolarZenithAngle', op_sza)
              l1.set_supportfield('OptimizedSolarAzimuthAngle', op_saa)
              l1.set_supportfield('OptimizedViewingZenithAngle', op_vza)
              l1.set_supportfield('OptimizedViewingAzimuthAngle', op_vaa)
              l1.set_supportfield('OptimizedRelativeAzimuthAngle', op_aza)
              l1.set_supportfield('OptimizedSurfaceAltitude',op_surfalt)
              l1.set_supportfield('OptimizedAircraftLongitude',op_ac_lon)
              l1.set_supportfield('OptimizedAircraftLatitude',op_ac_lat)
              l1.set_supportfield('OptimizedAircraftAltitudeAboveSurface',op_ac_alt_surf)
              l1.set_supportfield('OptimizedAircraftSurfaceAltitude',op_ac_surf_alt)
              l1.set_supportfield('OptimizedAircraftPixelBore',op_ac_pix_bore)
              l1.set_supportfield('OptimizedAircraftPos',op_ac_pos)
              l1.set_supportfield('OptimizedLongitude',op_lon)
              l1.set_supportfield('OptimizedLatitude',op_lat)
              l1.set_supportfield('OptimizedCornerLongitude',op_clon)
              l1.set_supportfield('OptimizedCornerLatitude',op_clat)
              l1.set_supportfield('OptimizedObservationAltitude',op_obsalt)

              l1.set_supportfield('AvionicsLongitude',av_lon)
              l1.set_supportfield('AvionicsLatitude',av_lat)
              l1.set_supportfield('AvionicsCornerLongitude',av_clon)
              l1.set_supportfield('AvionicsCornerLatitude',av_clat)
              l1.set_supportfield('AvionicsSurfaceAltitude',av_surfalt)

              l1.set_1d_flag(None,True,True)
        else:
              l1.set_1d_flag(None,None,True)
        l1.add_akaze(ac_roll,ac_pitch,ac_heading,akaze_Distance_Akaze_Reprojected,akaze_Optimization_Convergence_Fail,akaze_Reprojection_Fit_Flag)

    l1.close()

    f.close()

