
  args <- commandArgs(TRUE)

  # Correction Factors---------------------------------------------------------
  slope_lon_relative = as.double(args[1])
  slope_lat_relative = as.double(args[2])
  intercept_lon_relative = as.double(args[3])
  intercept_lat_relative = as.double(args[4])
  slope_lon_absolute = as.double(args[5])
  slope_lat_absolute = as.double(args[6])
  intercept_lon_absolute = as.double(args[7])
  intercept_lat_absolute = as.double(args[8])

  # Customize Inputs-----------------------------------------------------------

  # Netcdf flight data file
  #file_flightnc = '../../Inputs/MethaneAIRrf01_hrt.nc',
  file_flightnc = args[9]

  # Digital Elevation Map created using DEM_elevatr.R
  file_dem = args[10]

  # Level 1 file. Must be a NetCDF
  file_L1_O2 = args[11]  
  L1_var_time = "GEOS_5_tau"

  # Directory to store output
  dir_output = args[12]

  # Aggregation factor for DEM (higher number speeds computation, lower number more accurate)
  dem_aggrregation_optimize = 8 # Perform the optimization using the cheap version
  dem_aggrregation_orthorectification = 1

  # Number of across-track points to calculate orthorectification
  points_sample_optimize = 3

  # Number of along-track points to calculate orthorectification.
  # Will be evenly spaced throughout the file
  times_sample = 3

  # Report things
  verbose = TRUE

  # Reprojection RMSE Flag threshold in meters
  reprojection_max_flag = 25

  # Default Inputs-------------------------------------------------------------

  # Framerate in seconds
  framerate = 0.1

  # Number of across track pixels
  points_x = 1280

  # Variable to select longitude and latitude from flight data
  var_lat                 =  args[15]
  var_lon                 =  args[16]
  var_pitch               =  args[17]
  var_roll                =  args[18]
  var_heading             =  args[19]
  var_alt_geoid           =  args[20]
  var_geoid_height_WGS84  =  args[21]

  # ECEF x, y, z, roll, pich, true heading
  bounds_lower = c(-400, -400, -400, -10,-10,-10)
  bounds_upper = c( 400,  400,  400,  10, 10, 10)
  boundary_fail_expansionfactor = 2

  # Earth Orientation Parameter (EOP) file
  file_eop = args[13]

  # Directory with user defined functions
  dir_lib = args[14]

  # Instrument angular field of view
  FOV = 33.7

  # Instrument focual length (mm)
  f = 16

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

  # User defined functions
  scripts <- list.files(dir_lib, pattern = "\\.r$", full.names = TRUE, ignore.case = TRUE)
  for(tick in 1:length(scripts)) {
    source(scripts[tick])
  }

  # Load Inputs----------------------------------------------------------------

  # flight data
  flight_nc <- nc_open(file_flightnc)

  # Digital elevation map
  #dem <- readRDS(file_dem)
  dem <- raster::raster(file_dem)

  # Earth Orientation Parameters
  eop <- read.table(file_eop, header = TRUE)

  # L1 File
  file_L1_O2_split <- strsplit(file_L1_O2, "/")[[1]]
  file_L1_O2_strip <- file_L1_O2_split[length(file_L1_O2_split)]
  L1_nc_O2 <- nc_open(file_L1_O2)

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

  # * Level 1 retrieval data-----------------------------------------------------

  # Generate a time object from the L1 data
  L1_time <- ymd_hms("1985-01-01 00:00:00 UTC") + 60 * 60 * ncvar_get(L1_nc_O2, L1_var_time) %>% seconds()
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

  # * Earth Orientation Parameters-----------------------------------------------

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

  # * Solar Ephemeris-----------------------------------------------------------

  # Calcualte ECI coordinates of the sun
  sun_eci <- 149597870691 * solar_ephemeris_AA(flight_df_L1$MJD_UTC + 2400000.5)
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

  # * Aircraft location---------------------------------------------------------
 
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
 
  # * Digital Elevation Map----------------------------------------------------

  # Find the centroid of the flight_df_L1 object
  centroid <- data.frame(lon = mean(flight_df_L1$lon), lat = mean(flight_df_L1$lat))

  # Calculate the expanded target area for isolating the DEM for orthorectificaiton
  target_demarea_polygon <- centroids2sfcpolygons(
    lon = centroid$lon,
    lat = centroid$lat,
    buffer_m = 15000)

  # Generate the local DEM for orthorectification
  dem_ortho <-  dem %>%
    raster::crop(extent(st_bbox(target_demarea_polygon[[1]])[c(1,3,2,4)])) %>%
    raster::aggregate(dem_aggrregation_optimize)

  # Convert the DEM to a data frame
  crs(dem_ortho)  <- "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"
  r.pts           <- rasterToPoints(dem_ortho, spatial=TRUE)
  target_ortho_df <- spTransform(r.pts, CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")) %>% as.data.frame()
  colnames(target_ortho_df) <- c("alt", "lon", "lat")
  dem_ecef_df <- geodetic2ecef_WGS84(
    lon = target_ortho_df$lon,
    lat = target_ortho_df$lat,
    alt = target_ortho_df$alt) %>% as.data.frame()
  dem_ecef_df$lon <- target_ortho_df$lon
  dem_ecef_df$lat <- target_ortho_df$lat
  dem_ecef_df$alt <- target_ortho_df$alt

  # Orthorectification---------------------------------------------------------
  # Calculate the registered orhtorectification by computing the avionics-only orthorectification 
  # then applying the supplied correction factors                                                 

  # * Avionics-only------------------------------------------------------------
  if(verbose) {message("Computing Avionics Only Orthorectification")}

  # Set sample times to truncate the corners
  sample_times <- seq(from = 1, to = nrow(flight_df_L1) - 1, length = times_sample) %>% round
  flight_df_Sample <- flight_df_L1[sample_times, ]

  # Find the tangent plane
  tangentplane <- geodetic2tangentplane_ecefWGS84(
    lon     = flight_df_Sample$lon,
    lat     = flight_df_Sample$lat
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
    heading = (pi/180) * flight_df_Sample$heading,
    pitch   = (pi/180) * flight_df_Sample$pitch,
    roll    = (pi/180) * flight_df_Sample$roll
  )

  # zhatprime = rotmat_basis[[1]] %*% rotmat_pointing[[1]] %*% c(0,0,1)

  # Perform the orthorectification
  lonmap_avionics <- matrix(nrow = points_sample_optimize, ncol = nrow(flight_df_Sample))
  latmap_avionics <- matrix(nrow = points_sample_optimize, ncol = nrow(flight_df_Sample))
  for(tick in 1:length(sample_times)) {
    sample.tick <- sample_times[tick]
    # Perform the orthorectification
    orthorectified <- orthorectification(
      dem_ecef_df     = dem_ecef_df,
      pos_ecef        = aircraft_ecef[sample.tick,],
      rotm_basis      = rotmat_basis[[tick]],
      rotm_pointing   = rotmat_pointing[[tick]],
      FOV             = FOV,
      f               = f,
      pixels          = points_sample_optimize,
      output          = "geodetic")
    lonmap_avionics[, tick] <- orthorectified$lon
    latmap_avionics[, tick] <- orthorectified$lat
  }

  # Register O2
  lonmap_registered_O2 <- (lonmap_avionics- intercept_lon_absolute)/slope_lon_absolute
  latmap_registered_O2 <- (latmap_avionics- intercept_lat_absolute)/slope_lat_absolute

  # * Optimize-----------------------------------------------------------------

  if(verbose) {message("Optimizing Avionics")}

  boundary_flag <- TRUE
  while(boundary_flag) {
    orthocorrection <- optim(par =  c(0,0,0,0,0,0), fn = reprojection_error_o2,
      dem_ecef_df             = dem_ecef_df,
      FOV                     = FOV,
      f                       = f,
      pos_ecef                = aircraft_ecef[sample_times, ],
      heading                 = flight_df_Sample$heading,
      pitch                   = flight_df_Sample$pitch,
      roll                    = flight_df_Sample$roll,
      lonmap_registered_O2    = lonmap_registered_O2,
      latmap_registered_O2    = latmap_registered_O2,
      lower                   = bounds_lower,
      upper                   = bounds_upper,
      method                  = "L-BFGS-B"
    )
    if(any(orthocorrection$par == bounds_lower | orthocorrection$par == bounds_upper)) {
      bounds_lower <- bounds_lower * boundary_fail_expansionfactor
      bounds_upper <- bounds_upper * boundary_fail_expansionfactor
    } else {
      boundary_flag <- FALSE
    }
  }


  # * Optimized Orthorectification---------------------------------------------
  
  corrections <- orthocorrection$par
  if(orthocorrection$convergence == 1 & verbose) {cat("\n!!Optimization did not converge!!")}

  # Set up the DEM Again-------------------------------------------------------

  # Generate the local DEM for orthorectification
  dem_ortho <-  dem %>%
    raster::crop(extent(st_bbox(target_demarea_polygon[[1]])[c(1,3,2,4)])) %>%
    raster::aggregate(dem_aggrregation_orthorectification)

  # Convert the DEM to a data frame
  crs(dem_ortho)  <- "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"
  r.pts           <- rasterToPoints(dem_ortho, spatial=TRUE)
  target_ortho_df <- spTransform(r.pts, CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")) %>% as.data.frame()
  colnames(target_ortho_df) <- c("alt", "lon", "lat")
  dem_ecef_df <- geodetic2ecef_WGS84(
    lon = target_ortho_df$lon,
    lat = target_ortho_df$lat,
    alt = target_ortho_df$alt) %>% as.data.frame()
  dem_ecef_df$lon <- target_ortho_df$lon
  dem_ecef_df$lat <- target_ortho_df$lat
  dem_ecef_df$alt <- target_ortho_df$alt

  # Orthorectify the O2 Channel------------------------------------------------

  # Find the tangent plane
  aircraft_ecef <- aircraft_ecef + rowrep(vec = corrections[1:3], n = nrow(aircraft_ecef))
  aircraft_geodetic <- ecef2geodetic_WGS84(aircraft_ecef)
  tangentplane <- geodetic2tangentplane_ecefWGS84(
    lon     = aircraft_geodetic$lon,
    lat     = aircraft_geodetic$lat
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
    heading = (pi/180) * (flight_df_L1$heading + corrections[4]),
    pitch   = (pi/180) * (flight_df_L1$pitch   + corrections[5]),
    roll    = (pi/180) * (flight_df_L1$roll    + corrections[6])
  )
    heading =  (flight_df_L1$heading + corrections[4])
    pitch   =  (flight_df_L1$pitch   + corrections[5])
    roll    =  (flight_df_L1$roll    + corrections[6])

  # Perform the orthorectification
  orthorectified_lon <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_lat <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_alt <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_vza <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_vaa <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_sza <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))
  orthorectified_saa <- matrix(nrow = 2 * points_x + 1, ncol = nrow(flight_df_L1))

  if(verbose) {cat("\nCalculating Orthorectification")}
  for(tick in 1:nrow(flight_df_L1)) {

    if(verbose) {(cat(paste("\n", tick, "of", nrow(flight_df_L1))))}

    # Perform the orthorectification
    orthorectified <- orthorectification(
      dem_ecef_df     = dem_ecef_df,
      pos_ecef        = aircraft_ecef[tick,],
      rotm_basis      = rotmat_basis[[tick]],
      rotm_pointing   = rotmat_pointing[[tick]],
      FOV             = FOV,
      f               = f,
      pixels          = 2 * points_x + 1,
      output          = "geodetic")
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
          alt = dem_height[,tick],
          eop = do.call("rbind", replicate(nrow(lon), IERS_L1[tick,], simplify = FALSE)),
          mjd = rep(flight_df_L1$MJD_UTC[tick], nrow(lon))
    )
    boresight_eci[,tick,] <- scan_eci[,tick,] - rowrep(n = nrow(lon), vec = aircraft_eci[tick,])
    boresight_eci[,tick,] <- boresight_eci[,tick,] / df3norm(boresight_eci[,tick,])
  }

  # AKAZE----------------------------------------------------------------------

  # Full resolution AKAZE
  lon_avionics <- ncvar_get(L1_nc_O2, "Longitude")
  lat_avionics <- ncvar_get(L1_nc_O2, "Latitude")
  lon_avionics_c <- ncvar_get(L1_nc_O2, "CornerLongitude")
  lat_avionics_c <- ncvar_get(L1_nc_O2, "CornerLatitude")

  # Register O2
  lon_registered_O2 <- (lon_avionics- intercept_lon_absolute)/slope_lon_absolute
  lat_registered_O2 <- (lat_avionics- intercept_lat_absolute)/slope_lat_absolute

  lon_ll_registered_O2 <- (lon_avionics_c[,,1] - intercept_lon_absolute)/slope_lon_absolute
  lon_lr_registered_O2 <- (lon_avionics_c[,,2] - intercept_lon_absolute)/slope_lon_absolute
  lon_ul_registered_O2 <- (lon_avionics_c[,,3] - intercept_lon_absolute)/slope_lon_absolute
  lon_ur_registered_O2 <- (lon_avionics_c[,,4] - intercept_lon_absolute)/slope_lon_absolute
  lonc_registered      <- abind(lon_ll_registered_O2, lon_lr_registered_O2, lon_ul_registered_O2, lon_ur_registered_O2, along = 3)

  lat_ll_registered_O2 <- (lat_avionics_c[,,1] - intercept_lat_absolute)/slope_lat_absolute
  lat_lr_registered_O2 <- (lat_avionics_c[,,2] - intercept_lat_absolute)/slope_lat_absolute
  lat_ul_registered_O2 <- (lat_avionics_c[,,3] - intercept_lat_absolute)/slope_lat_absolute
  lat_ur_registered_O2 <- (lat_avionics_c[,,4] - intercept_lat_absolute)/slope_lat_absolute
  latc_registered     <- abind(lat_ll_registered_O2, lat_lr_registered_O2, lat_ul_registered_O2, lat_ur_registered_O2, along = 3)

  # Digital Elevation Map Height
  dem_height_registered <-
    matrix(nrow = nrow(lon_registered_O2),
           ncol = ncol(lon_registered_O2),
           data = raster::extract(dem,
           SpatialPoints(list(x = as.vector(lon_registered_O2), y = as.vector(lat_registered_O2))))
    )

  # Calculate the Error--------------------------------------------------------
  dists <- spDists(
              x = cbind(c(as.vector(lon)),
                        c(as.vector(lat))),
              y = cbind(c(as.vector(lon_registered_O2)),
                        c(as.vector(lat_registered_O2))),
              longlat = TRUE, diagonal = TRUE) %>% '*'(1000)
  distmap <- matrix(nrow = nrow(lon), ncol = ncol(lon), data = dists, byrow = TRUE)

  # Write NetCDF---------------------------------------------------------------

  # Make dimensions
  xvals <- 1:points_x
  yvals <- ncvar_get(L1_nc_O2, L1_var_time)
  zvals <- 1:3
  cvals <- 1:4
  nx <- length(xvals)
  ny <- length(yvals)
  nz <- 3
  nc <- 4
  flagdim <- ncdim_def( 'flag', 'flag',  1 )
  xdim <- ncdim_def( 'nrow', 'acrosstrack',  xvals )
  ydim <- ncdim_def( 'nframe', 'time',         yvals )
  zdim <- ncdim_def( 'z', 'vector_components', zvals )
  cdim <- ncdim_def( 'c', 'corner',       cvals )
  # Make var
  mv <- 99999 # missing value
  var_lon       <- ncvar_def('Longitude',                     'degrees',        list(xdim,ydim),      mv)
  var_lat       <- ncvar_def('Latitude',                      'degrees',        list(xdim,ydim),      mv)
  var_lon_reg   <- ncvar_def('LongitudeRegistered',          'degrees',        list(xdim,ydim),      mv)
  var_lat_reg   <- ncvar_def('LatitudeRegistered',           'degrees',        list(xdim,ydim),      mv)
  var_dist      <- ncvar_def('DistanceRegisteredReprojected','m',             list(xdim,ydim),      mv)
  var_pos       <- ncvar_def('AircraftPos',                  'm ECI',          list(ydim, zdim),     mv)
  var_lon_air   <- ncvar_def('AircraftLongitude',            'degrees',        list(ydim),           mv)
  var_lat_air   <- ncvar_def('AircraftLatitude',             'degrees',        list(ydim),           mv)
  var_hdg_air   <- ncvar_def('AircraftHeading',              'degrees',        list(ydim),           mv)
  var_roll_air  <- ncvar_def('AircraftRoll',                 'degrees',        list(ydim),           mv)
  var_pitch_air <- ncvar_def('AircraftPitch',                'degrees',        list(ydim),           mv)
  var_dem_air   <- ncvar_def('AircraftSurfaceAltitude',      'm WGS84',        list(ydim),           mv)
  var_alt_dem   <- ncvar_def('AircraftAltitudeAboveSurface', 'm',              list(ydim),           mv)
  var_alt_WGS84 <- ncvar_def('AircraftAltitudeAboveWGS84',   'm',              list(ydim),           mv)
  var_bor       <- ncvar_def('AircraftPixelBore',            'm ECI',          list(xdim,ydim,zdim), mv)
  var_vza       <- ncvar_def('ViewingZenithAngle',            'degrees',        list(xdim,ydim),      mv)
  var_vaa       <- ncvar_def('ViewingAzimuthAngle',           'degrees bearing',list(xdim,ydim),      mv)
  var_sza       <- ncvar_def('SolarZenithAngle',              'degrees',        list(xdim,ydim),      mv)
  var_saa       <- ncvar_def('SolarAzimuthAngle',             'degrees bearing',list(xdim,ydim),      mv)
  var_dem       <- ncvar_def('SurfaceAltitude',               'm WGS84',        list(xdim,ydim),      mv)
  var_dem_reg   <- ncvar_def('SurfaceAltitudeRegistered',    'm WGS84',        list(xdim,ydim),      mv)
  var_lonc      <- ncvar_def('CornerLongitude',               'degrees',        list(xdim,ydim,cdim), mv)
  var_latc      <- ncvar_def('CornerLatitude',                'degrees',        list(xdim,ydim,cdim), mv)
  var_lonc_reg  <- ncvar_def('CornerLongitudeRegistered',    'degrees',        list(xdim,ydim,cdim), mv)
  var_latc_reg  <- ncvar_def('CornerLatitudeRegistered',     'degrees',        list(xdim,ydim,cdim), mv)
  var_conflag   <- ncvar_def('OptimizationConvergenceFail', 'flag',           list(flagdim), mv)
  var_regflag   <- ncvar_def('ReprojectionFitFlag',         'flag',           list(flagdim), mv)

  # Make new output file
  output_fname <- paste0(dir_output, "/", file_L1_O2_strip)
  ncid_new <- nc_create(output_fname, list(var_lon, var_lat, var_lon_reg, var_lat_reg, var_dist, var_pos, var_lon_air, var_lat_air, var_hdg_air, var_roll_air, var_pitch_air, var_dem_air, var_alt_dem, var_alt_WGS84, var_bor, var_vza, var_vaa, var_sza, var_saa, var_dem, var_dem_reg, var_lonc, var_latc, var_lonc_reg, var_latc_reg, var_conflag, var_regflag))

  # Fill the file with data
  ncvar_put(ncid_new, var_lon,             lon[,1:ny],                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_lat,             lat[,1:ny],                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_lon_reg,         lon_registered_O2[,1:ny],    start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_lat_reg,         lat_registered_O2[,1:ny],    start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_dist,            distmap[,1:ny],              start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_pos,             aircraft_eci[1:ny,],         start=c(1,1),   count=c(ny,nz))
  ncvar_put(ncid_new, var_lon_air,         aircraft_geodetic$lon[1:ny], start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_lat_air,         aircraft_geodetic$lat[1:ny], start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_hdg_air,         heading[1:ny], start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_roll_air,        roll[1:ny],    start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_pitch_air,       pitch[1:ny],   start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_dem_air,         aircraft_dem[1:ny],          start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_alt_dem,         aircraft_alt_dem[1:ny],      start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_alt_WGS84,       aircraft_alt_wgs84[1:ny],    start=c(1),     count=c(ny))
  ncvar_put(ncid_new, var_bor,             boresight_eci[,1:ny,],       start=c(1,1,1), count=c(nx,ny,nz))
  ncvar_put(ncid_new, var_vza,             vza[,1:ny],                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_vaa,             vaa[,1:ny],                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_sza,             sza[,1:ny],                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_saa,             saa[,1:ny],                  start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_dem,             dem_height[,1:ny],           start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_dem_reg,         dem_height_registered[,1:ny],start=c(1,1),   count=c(nx,ny))
  ncvar_put(ncid_new, var_lonc,            lonc[,1:ny,],                start=c(1,1,1), count=c(nx,ny,nc))
  ncvar_put(ncid_new, var_lonc_reg,        lonc_registered[,1:ny,],     start=c(1,1,1), count=c(nx,ny,nc))
  ncvar_put(ncid_new, var_latc,            latc[,1:ny,],                start=c(1,1,1), count=c(nx,ny,nc))
  ncvar_put(ncid_new, var_latc_reg,        latc_registered[,1:ny,],     start=c(1,1,1), count=c(nx,ny,nc))
  ncvar_put(ncid_new, var_conflag,         as.numeric(orthocorrection$convergence == 1), start=c(1), count=c(1))
  ncvar_put(ncid_new, var_regflag,         as.numeric(max(distmap) > reprojection_max_flag), start=c(1),      count=c(1))

  # Close our new output file
  nc_close(ncid_new)
