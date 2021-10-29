
Orthorectification_Avionics_NC_Airglow <- function(

  # Customize Inputs-----------------------------------------------------------

  # Netcdf flight data file
  file_flightnc = '/n/holylfs04/LABS/wofsy_lab/Lab/econway/DATA/MethaneAIR21rf10s.nc',

  # Digital Elevation Map created using DEM_elevatr.R
  file_dem = "/n/home10/benmergui/MethaneSAT_LOSST/MethaneSAT_LOSST_Cloud/dat/dem.tif",

  # Level 1 file. Must be a NetCDF 
  file_L1 = "/n/holylfs04/LABS/wofsy_lab/Lab/MethaneAIR/level1/RF10/O2Avionics/MethaneAIR_L1B_O2_20210816T230134_20210816T230204_20210818T131157.nc",
  L1_var_time = "GEOS_5_tau",

  # Directory to store output 
  dir_output = "/n/holyscratch01/wofsy_lab/benmergui/Orthorectification_Avionics/RF10_70km/",

  # Heights above WGS84 to orthorectify against
  celestial_sphere_height_km = 70,
  # Default Inputs-------------------------------------------------------------

  # Framerate in seconds
  framerate = 0.1,

  # Number of across track pixels
  points_x = 1280,

  # Variable to select longitude and latitude from flight data
  var_lat                 = "GGLAT",
  var_lon                 = "GGLON",
  var_pitch               = "PITCH",
  var_roll                = "ROLL",
  var_heading             = "THDG",
  var_alt_geoid           = "GEOPTH",
  var_geoid_height_WGS84  = "GGEOIDHT",

  # Earth Orientation Parameter (EOP) file 
  file_eop = "/n/holylfs04/LABS/wofsy_lab/Lab/econway/DATA/EOP-Last5Years_celestrak_20210723.txt",

  # Directory with user defined functions
  dir_lib = "/n/holylfs04/LABS/wofsy_lab/Lab/econway/DATA/0User_Functions/",

  # Instrument angular field of view
  FOV = 33.7,

  # Instrument focual length (mm)
  f = 16
  
) {

  # Dependencies---------------------------------------------------------------
  library(orientlib)
  library(dplyr)
  library(raster)
  library(sp)
  library(ncdf4)
  library(lubridate)
  library(stringr)
  library(pracma)
  library(astrolibR)
  library(parallel)
  library(foreach)
  library(doParallel)
  library(abind)
  library(flexclust)

  # User defined functions (includes loading R packages) 
  scripts <- list.files(dir_lib, pattern = "\\.R$", full.names = TRUE, ignore.case = TRUE)
  for(tick in 1:length(scripts)) {
    source(scripts[tick])
  }

  # Special!
  #source("./intersect_ellipsoid.R")

  # Load Inputs----------------------------------------------------------------

  # flight data
  flight_nc <- ncdf4::nc_open(file_flightnc)

  # Digital elevation map
  dem <- raster::raster(file_dem)

  # Earth Orientation Parameters
  eop <- read.table(file_eop, header = TRUE)

  # L1 File
  file_L1_split <- strsplit(file_L1, "/")[[1]]
  file_L1_strip <- file_L1_split[length(file_L1_split)]
  L1_nc <- nc_open(file_L1)

  # Process Inputs-------------------------------------------------------------

  # * Flight Data--------------------------------------------------------------

  # Initialize flight data frame
  flightdate <- ncatt_get(flight_nc, 0)$FlightDate
  flightdate_string <- paste0(substr(flightdate, 7, 10), "-",
                              substr(flightdate, 1, 2), "-",
                              substr(flightdate, 4, 5))

  #  Get the variables 
  time                <- as.vector(ncvar_get(flight_nc, "Time"))
  lat                 <- as.vector(ncvar_get(flight_nc, var_lat))
  lon                 <- as.vector(ncvar_get(flight_nc, var_lon))
  pitch               <- as.vector(ncvar_get(flight_nc, var_pitch))
  roll                <- as.vector(ncvar_get(flight_nc, var_roll))
  heading             <- as.vector(ncvar_get(flight_nc, var_heading))
  geoid               <- as.vector(ncvar_get(flight_nc, var_alt_geoid))
  geoid_height_WGS84  <- as.vector(ncvar_get(flight_nc, var_geoid_height_WGS84))

  # The longest length in the system:
  len_full <- max(
    length(time),
    length(lat),
    length(lon),
    length(pitch),
    length(roll),
    length(heading),
    length(geoid),
    length(geoid_height_WGS84)
  ) 
    
  # Interpolate variables down to the highest frequency 
  lat_full                = splinefun(
                              x = seq(from = 1, to = len_full, length = length(lat)), y = lat
                            )(1:len_full)
  lon_full                = splinefun(
                              x = seq(from = 1, to = len_full, length = length(lon)), y = lon
                            )(1:len_full)
  pitch_full              = splinefun(
                              x = seq(from = 1, to = len_full, length = length(pitch)), y = pitch
                            )(1:len_full)
  roll_full               = splinefun(
                              x = seq(from = 1, to = len_full, length = length(roll)), y = roll
                            )(1:len_full)
  # Use a constant approx for heading to deal with 360 crossing. 
  heading_full            = approx(
                              x     = seq(from = 1, to = len_full, length = length(heading)), 
                              y     = heading,
                              xout  = 1:len_full,
                              method = "constant"
                            )$y 
  geoid_full              = splinefun(
                              x = seq(from = 1, to = len_full, length = length(geoid)), y = geoid
                            )(1:len_full)
  geoid_height_WGS84_full = splinefun(
                              x = seq(from = 1, to = len_full, length = length(geoid_height_WGS84)), y = geoid_height_WGS84
                            )(1:len_full)
 
  # Generate a dataframe of the flight data
  flight_df <- data.frame(
    time = ymd_hms(paste(flightdate_string, "00:00:00 UTC")) +
      rep(seconds(ncvar_get(flight_nc, "Time")), each = len_full / length(time)) +
      rep((0:((len_full / length(time)) - 1))/(len_full / length(time)), length(time)),
    geoid_height_WGS84  = geoid_height_WGS84_full, 
    alt_geoid           = geoid_full,
    lat                 = lat_full,
    lon                 = lon_full,
    pitch               = pitch_full,
    roll                = roll_full,
    heading             = heading_full
  )

  # * Level 1 retrieval data---------------------------------------------------

  # Generate a time object from the L1 data
  L1_time <- ymd_hms("1985-01-01 00:00:00 UTC") + 60 * 60 * ncvar_get(L1_nc, L1_var_time) %>% seconds() 
  # Add a timestep for the last pixel end
  L1_time <- c(L1_time, L1_time[length(L1_time)] + seconds(framerate))

  # Pull flight_df during the time interval
  flight_df <- flight_df[(flight_df$time >= min(L1_time) - seconds(10)) & 
    (flight_df$time <= max(L1_time) + seconds(1)), ]
  # Calculate time variables
  flight_df$MJD_UTC         <- utc2mjd(flight_df$time)

  # Interpolate to the L1 times 
  flight_df_L1 <- data.frame(
    time                = L1_time,
    geoid_height_WGS84  = approx(x = flight_df$time, y = flight_df$geoid_height_WGS84,  xout = L1_time)$y,
    alt_geoid           = approx(x = flight_df$time, y = flight_df$alt_geoid,           xout = L1_time)$y,
    lat                 = approx(x = flight_df$time, y = flight_df$lat,                 xout = L1_time)$y,
    lon                 = approx(x = flight_df$time, y = flight_df$lon,                 xout = L1_time)$y,
    pitch               = approx(x = flight_df$time, y = flight_df$pitch,               xout = L1_time)$y,
    roll                = approx(x = flight_df$time, y = flight_df$roll,                xout = L1_time)$y,
    heading             = approx(x = flight_df$time, y = flight_df$heading,             xout = L1_time)$y,
    MJD_UTC             = approx(x = flight_df$time, y = flight_df$MJD_UTC,             xout = L1_time)$y
  )

  # * Earth Orientation Parameters---------------------------------------------

  const.Arcs      = 3600*180/pi         # Arcseconds per radian
  IERS_L1 = data.frame(
    x_pole  = splinefun(x = eop$MJD, y = eop$x)(flight_df_L1$MJD_UTC)         / const.Arcs,
    y_pole  = splinefun(x = eop$MJD, y = eop$y)(flight_df_L1$MJD_UTC)         / const.Arcs,
    UT1_UTC = splinefun(x = eop$MJD, y = eop$UT1.UTC)(flight_df_L1$MJD_UTC),
    LOD     = splinefun(x = eop$MJD, y = eop$LOD)(flight_df_L1$MJD_UTC),
    dpsi    = splinefun(x = eop$MJD, y = eop$dPsi)(flight_df_L1$MJD_UTC)      / const.Arcs,
    deps    = splinefun(x = eop$MJD, y = eop$dEpsilon)(flight_df_L1$MJD_UTC)  / const.Arcs,
    dx_pole = splinefun(x = eop$MJD, y = eop$dX)(flight_df_L1$MJD_UTC)        / const.Arcs,
    dy_pole = splinefun(x = eop$MJD, y = eop$dY)(flight_df_L1$MJD_UTC)        / const.Arcs,
    TAI_UTC = splinefun(x = eop$MJD, y = eop$DAT)(flight_df_L1$MJD_UTC)
  )

  # * Solar Ephemeris----------------------------------------------------------

  # Calcualte ECI coordinates of the sun
  sun_eci <- 149597870691 * solar_ephemeris_AA(flight_df_L1$MJD_UTC + 2400000.5 )
  flight_df_L1$sun_eci_x <- sun_eci$x
  flight_df_L1$sun_eci_y <- sun_eci$y
  flight_df_L1$sun_eci_z <- sun_eci$z

  # Calculate ECEF coordinates of the sun
  sun_ecef <- eci2ecef(
    mjd = flight_df_L1$MJD_UTC,
    eci = sun_eci,
    eop = IERS_L1
  )
  flight_df_L1$sun_ecef_x <- sun_ecef[,1]
  flight_df_L1$sun_ecef_y <- sun_ecef[,2]
  flight_df_L1$sun_ecef_z <- sun_ecef[,3]

  # Calculate the sub-solar point
  sun_geodetic <- ecef2geodetic_WGS84(ecef = sun_ecef)
  flight_df_L1$sun_lon <- sun_geodetic$lon
  flight_df_L1$sun_lat <- sun_geodetic$lat

  # * Aircraft location--------------------------------------------------------
  
  # Calcualte ECI coordinates of the aircraft
  aircraft_eci <- geodetic2eci_WGS84(
      lat = flight_df_L1$lat,
      lon = flight_df_L1$lon,
      alt = flight_df_L1$alt_geoid + flight_df_L1$geoid_height_WGS84,
      eop = IERS_L1,
      mjd = flight_df_L1$MJD_UTC
  )
  flight_df_L1$aircraft_eci_x <- aircraft_eci[,1]
  flight_df_L1$aircraft_eci_y <- aircraft_eci[,2]
  flight_df_L1$aircraft_eci_z <- aircraft_eci[,3]

  # Calculate ECEF coordinates of the aircraft 
  aircraft_ecef <- eci2ecef(
    mjd = flight_df_L1$MJD_UTC, 
    eci = aircraft_eci, 
    eop = IERS_L1
  )
  flight_df_L1$aircraft_ecef_x <- aircraft_ecef[,1]
  flight_df_L1$aircraft_ecef_y <- aircraft_ecef[,2]
  flight_df_L1$aircraft_ecef_z <- aircraft_ecef[,3]

  aircraft_geodetic <-  ecef2geodetic_WGS84(ecef = aircraft_ecef)
  flight_df_L1$aircraft_lon       <- aircraft_geodetic$lon
  flight_df_L1$aircraft_lat       <- aircraft_geodetic$lat
  flight_df_L1$aircraft_alt_WGS84 <- aircraft_geodetic$alt

  # Orthorectification---------------------------------------------------------

  # Find the tangent plane
  tangentplane <- geodetic2tangentplane_ecefWGS84(
    lon     = flight_df_L1$lon, 
    lat     = flight_df_L1$lat
  )
  xhat <- tangentplane$yt_hat
  yhat <- tangentplane$xt_hat
  zhat <- - tangentplane$zt_hat

  # Find the rotation to the tangent plane 
  rotmat_basis <- lapply(1:nrow(xhat), function(x) {
    unitvectors2matrix(xhat  = xhat[x,], yhat  = yhat[x,], zhat  = zhat[x,])
  })

  # Find the rotation to instrument pointing
  rotmat_pointing <- headingpitchroll2matrix(
    heading = (pi/180) * flight_df_L1$heading,
    pitch   = (pi/180) * flight_df_L1$pitch,
    roll    = (pi/180) * flight_df_L1$roll
  )

  # Apply a reflection about the xy plane
  rotmat_pointing <- lapply(rotmat_pointing, function(x) {
    x %*% rbind(c(1,0,0), c(0,1,0), c(0,0,-1))
  })

  # Calculate the instrument boresight in ecef
  boresight_ecef <- lapply(1:length(rotmat_pointing), function(x) { 
    (rotmat_basis[[x]] %*% rotmat_pointing[[x]] %*% c(0,0,1)) %>%
    as.vector
  })

  # Find the axis around wichto rotate to get the individual pixel bores
  rotationaxis_ecef <- lapply(1:length(rotmat_pointing), function(x) {
    (rotmat_basis[[x]] %*% rotmat_pointing[[x]] %*% c(1,0,0)) %>%
    as.vector
  })

  # Find the pixel angles
  pixel_angs <- (180/pi) * 2 * atan2(
    seq(from = - f * tan((pi/180) * FOV/2), to = f * tan((pi/180) * FOV/2), length = 2 * points_x + 1),
    (2*f)
  )

  # Perform the orthorectification
  orthorectified_lon <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_lat <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_alt <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_vza <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_vaa <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_sza <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_saa <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))

  for(tick in 1:nrow(flight_df_L1)) {

    pixel_bores <- sapply(pixel_angs, function(x) {
      vecrotation(v = boresight_ecef[[tick]], n = as.vector(rotationaxis_ecef[[tick]]), theta_deg = x)
    }) %>% t()

    # Simply intersect with an ellipsoid inflated to the celestial sphere
    orthorectified <- sapply(1:nrow(pixel_bores), function(x) { 
      intersect_ellipsoid(
        pos_ecef          = aircraft_ecef[tick,],
        pixel_bore        = pixel_bores[x,],
        elevation         = celestial_sphere_height_km * 1000,
        latitude          = aircraft_geodetic$lat[tick],
        celestial_sphere  = TRUE,
        output            = "geodetic") %>% unlist
      }) %>% t %>% as.data.frame
    orthorectified_lon[, tick] <- orthorectified$lon
    orthorectified_lat[, tick] <- orthorectified$lat
    orthorectified_alt[, tick] <- orthorectified$alt

    # Find viewing angles
    spacecraft_zenaz  <- zenithazimuth(lon              = orthorectified$lon,
                                       lat              = orthorectified$lat,
                                       alt              = orthorectified$alt,
                                       lightsource_ecef = aircraft_ecef[tick, ])
    orthorectified_vza[, tick]       <- spacecraft_zenaz$zenith
    orthorectified_vaa[, tick]       <- spacecraft_zenaz$azimuth

    # Find the solar angles 
    solar_zenaz       <- zenithazimuth(lon              = orthorectified$lon,
                                       lat              = orthorectified$lat,
                                       alt              = orthorectified$alt,
                                       lightsource_ecef = sun_ecef[tick, ])
    orthorectified_sza[, tick]       <- solar_zenaz$zenith
    orthorectified_saa[, tick]       <- solar_zenaz$azimuth
  }

  # Assemble Output------------------------------------------------------------

  # * Extract from Orthorectification Object-----------------------------------

  # Identify Corner Indicies
  indices_left   <- seq(from = 1,  to = nrow(orthorectified_lon) - 2, by = 2)
  indices_center <- seq(from = 2,  to = nrow(orthorectified_lon) - 1, by = 2)
  indices_right  <- seq(from = 3,  to = nrow(orthorectified_lon),     by = 2)
  indices_lower  <- seq(from = 1,  to = ncol(orthorectified_lon) - 1, by = 1)
  indices_upper  <- seq(from = 2,  to = ncol(orthorectified_lon),     by = 1)

  # Organize to the relevant points
  # Longitude/Latitudes of corners
  # Lon lower-left
  lon_ll <- orthorectified_lon[indices_left,   indices_lower]
  # Lon lower-right
  lon_lr <- orthorectified_lon[indices_right,  indices_lower]
  # Lon upper-left
  lon_ul <- orthorectified_lon[indices_left,   indices_upper]
  # Lon upper-right
  lon_ur <- orthorectified_lon[indices_right,  indices_upper]
  # Combine
  lonc <- abind(lon_ll, lon_lr, lon_ul, lon_ur, along = 3)
  # Center
  lon <- (lon_ll + lon_lr + lon_ul + lon_ur) / 4

  # Lat lower-left
  lat_ll <- orthorectified_lat[indices_left,   indices_lower]
  # Lat lower-right
  lat_lr <- orthorectified_lat[indices_right,  indices_lower]
  # Lat upper-left
  lat_ul <- orthorectified_lat[indices_left,   indices_upper]
  # Lat upper-right
  lat_ur <- orthorectified_lat[indices_right,  indices_upper]
  # Combine
  latc <- abind(lat_ll, lat_lr, lat_ul, lat_ur, along = 3)
  # Center
  lat <- (lat_ll + lat_lr + lat_ul + lat_ur) / 4

  # Viewing Zenmith Angle
  vza <- orthorectified_vza[indices_center,   indices_lower]
  # Viewing Azimuth Angle
  vaa <- orthorectified_vaa[indices_center,   indices_lower]
  # Solar Zenith Angle
  sza <- orthorectified_sza[indices_center,   indices_lower]
  # Solar Azimuth Angle
  saa <- orthorectified_saa[indices_center,   indices_lower]

  # * Externally Calculated Variables------------------------------------------

  # Digital Elevation Map Height
  dem_height <-
    matrix(nrow = nrow(lat),
           ncol = ncol(lat),
           data = raster::extract(dem,
           SpatialPoints(list(x = as.vector(lon), y = as.vector(lat))))
    )

  # DEM Beneath Aircraft
  aircraft_dem <- raster::extract(dem, 
    SpatialPoints(list(x = flight_df_L1$lon, y = flight_df_L1$lat)))

  # ECI of WGS84 Ellipsoid Below Aircraft
  WGS84_subsat_eci <- geodetic2eci_WGS84(
    lat = flight_df_L1$lat,
    lon = flight_df_L1$lon,
    alt = 0, 
    eop = IERS_L1, 
    mjd = flight_df_L1$MJD_UTC
  )

  # Radius to WGS84 Ellipsoid
  WGS84_subsat_alt <- df3norm(WGS84_subsat_eci)

  # Aircraft altitude in ECI (From Center of Earth)
  aircraft_alt_eci <- df3norm(aircraft_eci)

  # Aircraft Altitude above WGS84 Ellipsoid
  aircraft_alt_wgs84 <- aircraft_alt_eci - WGS84_subsat_alt

  # Aircraft Altitude Above DEM
  aircraft_alt_dem <- aircraft_alt_eci - WGS84_subsat_alt - aircraft_dem

  # Calculate the scan and boresight vectors 
  scan_eci      <- array(dim = c(nrow(lon), ncol(lon), 3))
  boresight_eci <- array(dim = c(nrow(lon), ncol(lon), 3))
  for(tick in 1:ncol(lon)) { 
    scan_eci[,tick,] <- geodetic2eci_WGS84(
          lon = lon[,tick],
          lat = lat[,tick],
          alt = celestial_sphere_height_km * 1000,
          eop = do.call("rbind", replicate(nrow(lon), IERS_L1[tick,], simplify = FALSE)),
          mjd = rep(flight_df_L1$MJD_UTC[tick], nrow(lon))
    )
    boresight_eci[,tick,] <- scan_eci[,tick,] - rowrep(n = nrow(lon), vec = aircraft_eci[tick,])
    boresight_eci[,tick,] <- boresight_eci[,tick,] / df3norm(boresight_eci[,tick,])
  }

  # Write NetCDF---------------------------------------------------------------

  # Make dimensions
  xvals <- 1:points_x
  yvals <- ncvar_get(L1_nc, L1_var_time)
  zvals <- 1:3
  cvals <- 1:4
  nx <- length(xvals)
  ny <- length(yvals)
  nz <- 3
  nc <- 4
  xdim <- ncdim_def( 'nrow', 'acrosstrack',  xvals )
  ydim <- ncdim_def( 'nframe', 'time',         yvals )
  zdim <- ncdim_def( 'z', 'vector_components', zvals )
  cdim <- ncdim_def( 'c', 'corner',       cvals )
  # Make var
  mv <- 99999 # missing value
  var_lon       <- ncvar_def('Longitude',                     'degrees',        list(xdim,ydim),      mv)
  var_lat       <- ncvar_def('Latitude',                      'degrees',        list(xdim,ydim),      mv)
  var_pos       <- ncvar_def('AircraftPos',                  'm ECI',          list(ydim, zdim),     mv)
  var_lon_air   <- ncvar_def('AircraftLongitude',            'degrees',        list(ydim),           mv)
  var_lat_air   <- ncvar_def('AircraftLatitude',             'degrees',        list(ydim),           mv)
  var_dem_air   <- ncvar_def('AircraftSurfaceAltitude',      'm WGS84',        list(ydim),           mv)
  var_alt_dem   <- ncvar_def('AircraftAltitudeAboveSurface', 'm',              list(ydim),           mv)
  var_alt_WGS84 <- ncvar_def('AircraftAltitudeAboveWGS84',   'm',              list(ydim),           mv)
  var_bor       <- ncvar_def('AircraftPixelBore',            'm ECI',          list(xdim,ydim,zdim), mv)
  var_vza       <- ncvar_def('ViewingZenithAngle',            'degrees',        list(xdim,ydim),      mv)
  var_vaa       <- ncvar_def('ViewingAzimuthAngle',           'degrees bearing',list(xdim,ydim),      mv)
  var_sza       <- ncvar_def('SolarZenithAngle',              'degrees',        list(xdim,ydim),      mv)
  var_saa       <- ncvar_def('SolarAzimuthAngle',             'degrees bearing',list(xdim,ydim),      mv)
  var_dem       <- ncvar_def('SurfaceAltitude',               'm WGS84',        list(xdim,ydim),      mv)
  var_lonc      <- ncvar_def('CornerLongitude',               'degrees',        list(xdim,ydim,cdim), mv)
  var_latc      <- ncvar_def('CornerLatitude',                'degrees',        list(xdim,ydim,cdim), mv)
  # Make new output file
  output_fname <- paste0(dir_output, "/", file_L1_strip)
  ncid_new <- nc_create(output_fname, list(var_lon, var_lat, var_pos, var_lon_air, var_lat_air, var_dem_air, var_alt_dem, var_alt_WGS84, var_bor, var_vza, var_vaa, var_sza, var_saa, var_dem, var_lonc, var_latc))
  # Fill the file with data
  ncvar_put(ncid_new, var_lon,             lon,                                                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_lat,             lat,                                                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_pos,             aircraft_eci[1:(nrow(aircraft_eci) - 1),],             start=c(1,1),   count=c(ny,nz))
  ncvar_put(ncid_new, var_lon_air,         aircraft_geodetic$lon[1:(nrow(aircraft_geodetic) - 1)], start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_lat_air,         aircraft_geodetic$lat[1:(nrow(aircraft_geodetic) - 1)], start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_dem_air,         aircraft_dem[1:(length(aircraft_dem) - 1)],           start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_alt_dem,         aircraft_alt_dem[1:(length(aircraft_dem) - 1)],       start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_alt_WGS84,       aircraft_alt_wgs84[1:(length(aircraft_dem) - 1)],     start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_bor,             boresight_eci,                                        start=c(1,1,1), count=c(nx,ny,nz))
  ncvar_put(ncid_new, var_vza,             vza,                                                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_vaa,             vaa,                                                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_sza,             sza,                                             start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_saa,             saa,                                             start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_dem,             dem_height,                                           start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_lonc,            lonc,                                                 start=c(1,1,1), count=c(nx,ny,nc))
  ncvar_put(ncid_new, var_latc,            latc,                                                 start=c(1,1,1), count=c(nx,ny,nc))
  # Close our new output file
  nc_close(ncid_new)

}


