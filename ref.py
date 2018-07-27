#coding=utf-8
### pyhdf http://pysclint.sourceforge.net/pyhdf/pyhdf.SD.html
import math
import config as CF
from numpy import array
import numpy as np
from pyhdf.SD import SD, SDC
from scipy import ndimage

# read HDF file and initialization
def Init1KMMODISData(inHdfFile):
    print 'hdf init'

    latitude  = CF.SDSObject()
    longitude = CF.SDSObject()
    refArr = CF.SDSObject()
    sunZ = CF.SDSObject()
    senZ = CF.SDSObject()
    sunA = CF.SDSObject()
    senA = CF.SDSObject()
    #hdfDSArr = [CF.SDSObject()] * 7

    hdf = SD(inHdfFile, SDC.READ)

    lat = hdf.select('Latitude')
    latitude.data = np.array(lat[:,:])
    latitude.Nl = latitude.data.shape[0]
    latitude.Np = latitude.data.shape[1]

    lon = hdf.select('Longitude')
    longitude.data = array(lon[:,:])
    longitude.Nl = longitude.data.shape[0]
    longitude.Np = longitude.data.shape[1]

    sunz = hdf.select('SolarZenith')
    sunZ.data = array(sunz[:,:]) * 0.01
    sunZ.Nl = sunZ.data.shape[0]
    sunZ.Np = sunZ.data.shape[1]

    senz = hdf.select('SensorZenith')
    senZ.data = array(senz[:,:]) * 0.01
    senZ.Nl = senZ.data.shape[0]
    senZ.Np = senZ.data.shape[1]

    suna = hdf.select('SolarAzimuth')
    sunA.data = array(suna[:,:]) * 0.01
    sunA.Nl = sunA.data.shape[0]
    sunA.Np = sunA.data.shape[1]

    sena = hdf.select('SensorAzimuth')
    senA.data = array(sena[:,:]) * 0.01
    senA.Nl = senA.data.shape[0]
    senA.Np = senA.data.shape[1]

    # EV_1KM_RefSB  EV_250_Aggr1km_RefSB
    refArr.data = np.zeros((CF.Nbands, 2030,1354))
    ref = hdf.select('EV_250_Aggr1km_RefSB')
    #refArr.data = ((array(ref[:,:,:]) - CF.refOffset) * CF.refScale ) / np.cos( sunZ.data * CF.DEG2RAD)
    refArr.data[:2, :, :] = array(ref[:, :, :])
    refArr.Nl = refArr.data.shape[1]
    refArr.Np = refArr.data.shape[2]

    # EV_500_Aggr1km_RefSB
    ref = hdf.select('EV_500_Aggr1km_RefSB')
    refArr.data[2:7, :, :] = array(ref[:, :, :])

    hdf.end()
    return latitude, longitude, refArr, sunZ, senZ, sunA, senA
    #return array(lat[:,:])

# read low resolution HDF file
def ReadDEM(inHdfFile):

    hdfDS = CF.SDSObject()

    hdf = SD(inHdfFile, SDC.READ)

    lat = hdf.select('Elevation')
    hdfDS.data = array(lat[:,:])
    hdfDS.Nl = hdfDS.data.shape[0]
    hdfDS.Np = hdfDS.data.shape[1]

    hdf.end()

    return hdfDS

# get altitude from DEM file
def GetHeightFromDEM(latitude, longitude, inHdfDEMFile):

    hdfDSHeigh = CF.SDSObject()

    hdfDSHeigh.Nl = latitude.Nl
    hdfDSHeigh.Np = latitude.Np
    hdfDSHeigh.data = np.zeros((latitude.Nl, latitude.Np))
    print latitude.Np, latitude.Nl

    dem = ReadDEM(inHdfDEMFile)
    dem.data.resize(dem.Nl*dem.Np)

    for i in range(0, latitude.Nl):
        #print i
        for j in range(0, latitude.Np):
            hdfDSHeigh.data[i,j] = interp_dem(latitude.data[i,j], longitude.data[i,j], dem)
            #print hdfDSHeigh.data[i,j]

    return hdfDSHeigh

# write to HDF file
def WriteToHDF(outHDFFile, refResultArr, processStatus):

    hdfFile = SD(outHDFFile, SDC.WRITE | SDC.CREATE | SDC.TRUNC)

    # Assign a few attributes at the file level
    hdfFile.author = 'author'
    hdfFile.priority = 2
    bandT = 0

    for i in range(0, CF.Nbands):
        if(not processStatus[i]):
            continue

        #print refResultArr[::bandT].shape, refResultArr[::bandT].shape[0], refResultArr[::bandT].shape[1]

        # Create a dataset named 'd1' to hold a 3x3 float array.
        #d1 = hdfFile.create(CF.refBandNames[i], SDC.UINT16, (refResultArr.shape[0], refResultArr.shape[1]))
        d1 = hdfFile.create(CF.refBandNames[i], SDC.FLOAT32, (refResultArr.shape[0], refResultArr.shape[1]))

        # Set some attributs on 'd1'
        d1.description = 'simple atmosphere correction for ' + CF.refBandNames[i]
        d1.units = 'Watts/m^2/micrometer/steradian'

        # Name 'd1' dimensions and assign them attributes.
        dim1 = d1.dim(0)
        dim2 = d1.dim(1)

        # Assign values to 'd1'
        d1[:] = refResultArr[:,:,bandT]

        #print refResultArr[:,:,bandT].shape

        bandT += 1

        d1.endaccess()

    hdfFile.end()


# interpolation dem
def interp_dem(lat, lon, dem):

    fractrow = (90.0 - lat) * dem.Nl / 180.0;
    demrow1 = int ( math.floor(fractrow) )
    demrow2 = demrow1 + 1;
    if (demrow1 < 0):
        demrow1 = demrow2 + 1;
    if (demrow2 > dem.Nl - 1):
        demrow2 = demrow1 - 1;
    t = (fractrow - demrow1) / (demrow2 - demrow1);

    fractcol = (lon + 180.0) * dem.Np / 360.0;
    demcol1 = int(math.floor(fractcol));
    demcol2 = demcol1 + 1;
    if (demcol1 < 0):
        demcol1 = demcol2 + 1;
    if (demcol2 > dem.Np - 1):
        demcol2 = demcol1 - 1;
    u = (fractcol - demcol1) / (demcol2 - demcol1);

    #print demrow1, demcol1

    height11 = (dem.data)[demrow1 * dem.Np + demcol1];
    height12 = (dem.data)[demrow1 * dem.Np + demcol2];
    height21 = (dem.data)[demrow2 * dem.Np + demcol1];
    height22 = (dem.data)[demrow2 * dem.Np + demcol2];
    height = (int) (t * u * height22 + t * (1.0 - u) * height21 + (1.0 - t) * u * height12 + (1.0 - t) * (1.0 - u) * height11);
    if (height < 0):
        height = 0;
    return height;

def csalbrEffect(tau):
    a = [-.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857];

    xx = np.ones(tau.size) * a[0]
    xftau = np.ones(tau.size);
    for i in range(1, 6):
        xftau *= tau;
        xx += a[i] * xftau;

    return (3.0 * tau - ( (np.exp(-tau) * (1.0 - tau) + tau * tau * (xx - np.log(tau))) / 2.0 ) * (4.0 + 2.0 * tau) + 2.0 * np.exp(-tau)) / (4.0 + 3.0 * tau);
    #return (np.exp(-tau) * (1.0 - tau) + tau * tau * (xx - np.log(tau))) / 2.0
    #return xftau

# core do it
def chand(phi, muv, mus, taur, trup, rhoray, trdown, processStatus):
    xfd = 0.958725775;
    xbeta2 = 0.5;
    pl = [0] * 5;
    #fs01, fs02, fs0, fs1, fs2;
    as0 = [0.33243832, 0.16285370, -0.30924818, -0.10324388, 0.11493334, -6.777104e-02, 1.577425e-03, -1.240906e-02, 3.241678e-02, -3.503695e-02];
    as1 = [0.19666292, -5.439061e-02];
    as2 = [0.14545937, -2.910845e-02];
    #phios, xcos1, xcos2, xcos3;
    #xph1, xph2, xph3, xitm1, xitm2;
    #xlntaur, xitot1, xitot2, xitot3;
    #i, ib;

    phios = phi + 180.0;
    xcos1 = 1.0;
    xcos2 = math.cos(phios * CF.DEG2RAD);
    xcos3 = math.cos(2.0 * phios * CF.DEG2RAD);
    xph1 = 1.0 + (3.0 * mus * mus - 1.0) * (3.0 * muv * muv - 1.0) * xfd / 8.0;
    xph2 = - xfd * xbeta2 * 1.5 * mus * muv * math.sqrt(1.0 - mus * mus) * math.sqrt(1.0 - muv * muv);
    xph3 = xfd * xbeta2 * 0.375 * (1.0 - mus * mus) * (1.0 - muv * muv);

    pl[0] = 1.0;
    pl[1] = mus + muv;
    pl[2] = mus * muv;
    pl[3] = mus * mus + muv * muv;
    pl[4] = mus * mus * muv * muv;

    fs01 = fs02 = 0.0;
    for i in range(0, 5):
        fs01 = fs01 + ((pl[i] * as0[i]));
        fs02 = fs02 + ((pl[i] * as0[5 + i]));

    for ib in range(0, CF.Nbands):
        if (processStatus[ib]):
            xlntaur = math.log(taur[ib]);
            fs0 = fs01 + fs02 * xlntaur;
            fs1 = as1[0] + xlntaur * as1[1];
            fs2 = as2[0] + xlntaur * as2[1];
            trdown[ib] = math.exp(-taur[ib] / mus);
            trup[ib] = math.exp(-taur[ib] / muv);
            xitm1 = (1.0 - trdown[ib] * trup[ib]) / 4.0 / (mus + muv);
            xitm2 = (1.0 - trdown[ib]) * (1.0 - trup[ib]);
            xitot1 = xph1 * (xitm1 + xitm2 * fs0);
            xitot2 = xph2 * (xitm1 + xitm2 * fs1);
            xitot3 = xph3 * (xitm1 + xitm2 * fs2);
            rhoray[ib] = xitot1 * xcos1 + xitot2 * xcos2 * 2.0 + xitot3 * xcos3 * 2.0;

# Cos (Sun zenith angle) cos (observed zenith angle) The difference between the observed azimuth angle and the solar azimuth. Altitude Back Hemispherical reflection Molecular scattering Water vapor absorption Ozone absorption
def getatmvariables(mus, muv, phi, height, processStatus):

    aH2O = [-5.60723, -5.25251, 0, 0, -6.29824, -7.70944, -3.91877, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bH2O = [0.820175, 0.725159, 0, 0, 0.865732, 0.966947, 0.745342, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    aO3 = [0.0715289, 0, 0.00743232, 0.089691, 0, 0, 0, 0.001, 0.00383, 0.0225, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263]
    taur0 = [0.05100, 0.01631, 0.19325, 0.09536, 0.00366, 0.00123, 0.00043, 0.3139, 0.2375, 0.1596, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155]

    tOG = [0] * CF.Nbands
    TtotraytH2O = [0] * CF.Nbands
    sphalb = [0] * CF.Nbands
    rhoray = [0] * CF.Nbands

    taur = [0] * CF.Nbands
    trup = [0] * CF.Nbands
    trdown = [0] * CF.Nbands
    sphalb0 = (np.arange(CF.MAXNUMSPHALBVALUES)) * CF.TAUSTEP4SPHALB
    sphalb0 = csalbrEffect(sphalb0)

    m = 1.0 / mus + 1.0 / muv;
    if (m > CF.MAXAIRMASS):
        return -1;

    psurfratio = math.exp(-height / float(CF.SCALEHEIGHT));
    for ib in range(0,  CF.Nbands):
        if (processStatus[ib]):
            taur[ib] = taur0[ib] * psurfratio;

    chand(phi, muv, mus, taur, trup, rhoray, trdown, processStatus);

    for ib in range(0, CF.Nbands):
        if (not processStatus[ib]):
            continue;

        if (taur[ib] / CF.TAUSTEP4SPHALB >= CF.MAXNUMSPHALBVALUES):
            sphalb[ib] = -1.0;
            continue;

        sphalb[ib] = sphalb0[int((taur[ib] / CF.TAUSTEP4SPHALB + 0.5))];
        Ttotrayu = ((2 / 3. + muv) + (2 / 3. - muv) * trup[ib]) / (4 / 3. + taur[ib]);
        Ttotrayd = ((2 / 3. + mus) + (2 / 3. - mus) * trdown[ib]) / (4 / 3. + taur[ib]);
        tO3 = tO2 = tH2O = 1.0;
        if (aO3[ib] != 0):
            tO3 = math.exp(-m * CF.UO3 * aO3[ib]);
        if (bH2O[ib] != 0):
            tH2O = math.exp(-math.exp(aH2O[ib] + bH2O[ib] * math.log(m * CF.UH2O)));

        TtotraytH2O[ib] = Ttotrayu * Ttotrayd * tH2O;
        tOG[ib] = tO3 * tO2;

    return sphalb, rhoray, TtotraytH2O, tOG


# atmosphere effect
def CalAtmoEffect(sunZ, senZ, sunA, senA, height, processStatus):

    phi = sunA.data - senA.data
    sphalb = np.zeros((latitude.Nl, latitude.Np, CF.Nbands), dtype=np.float32)
    rhoray = np.zeros((latitude.Nl, latitude.Np, CF.Nbands), dtype=np.float32)
    TtotraytH2O = np.zeros((latitude.Nl, latitude.Np, CF.Nbands), dtype=np.float32)
    tOG = np.zeros((latitude.Nl, latitude.Np, CF.Nbands), dtype=np.float32)

    # 
    for i in range(0, latitude.Nl):
        for j in range(0, latitude.Np):
           # if(math.cos(sunZ.data[i,j]* CF.DEG2RAD) < 0):
             #   print math.cos(sunZ.data[i,j]* CF.DEG2RAD), sunZ.data[i,j], i, j
            sphalb[i,j], rhoray[i,j], TtotraytH2O[i,j], tOG[i,j] = getatmvariables(math.cos(sunZ.data[i,j]* CF.DEG2RAD), math.cos(senZ.data[i,j]* CF.DEG2RAD), phi[i,j], height.data[i,j], processStatus)
            #print height.data[i,j], sphalb, rhoray, TtotraytH2O, tOG
    return sphalb, rhoray, TtotraytH2O, tOG

# calculation reflection
def CalRef(refArr, sphalb, rhoray, TtotraytH2O, tOG, processStatus):

    refFinalArr = np.zeros((refArr.Nl, refArr.Np, sum(processStatus)), dtype=np.float32)

    ibT = 0
    sphalb = ndimage.zoom(sphalb, zoom=(5, 4.9963, 1), order=1)
    rhoray = ndimage.zoom(rhoray, zoom=(5, 4.9963, 1), order=1)
    TtotraytH2O = ndimage.zoom(TtotraytH2O, zoom=(5, 4.9963, 1), order=1)
    tOG = ndimage.zoom(tOG, zoom=(5, 4.9963, 1), order=1)

    sunZ.data = ndimage.zoom(sunZ.data, zoom=(5, 4.9963), order=1)

    for ib in range(0, CF.Nbands):
        if (not processStatus[ib]):
            #print ib, processStatus[ib]
            continue;

        print ' Processing band:', ibT
        refArrNew = ((refArr.data[ib,:,:] - CF.refOffset[ib]) * CF.refScale[ib]) / np.cos(sunZ.data * CF.DEG2RAD)

        #print refArrNew[0,0], refArr.data[ib,0,0], sphalb[0,0,ib], rhoray[0,0,ib], TtotraytH2O[0,0,ib], tOG[0,0,ib]

        #refFinalArr[:,:,ibT] = ( (refArrNew / tOG[:,:,ib] - rhoray[:,:,ib] ) / TtotraytH2O[:,:,ib] ) /\
    #( 1 + (refFinalArr[:,:,ibT] / tOG[:,:,ib] - rhoray[:,:,ib] ) / TtotraytH2O[:,:,ib]) * sphalb[:,:,ib]

        corr_refl = (refArrNew / tOG[:,:,ib] - rhoray[:,:,ib] ) / TtotraytH2O[:,:,ib]
        #print corr_refl[0,0]
        corr_refl /= (1.0 + corr_refl * sphalb[:,:,ib]);
        #print corr_refl[0, 0]

        refFinalArr[:, :, ibT] = corr_refl

        #print refFinalArr[0,0,ibT]
        ibT += 1

    return refFinalArr


if __name__ == '__main__':

    # Read 1 km data
    inHdf1KMFile= 'MYD021KM.A2014015.1935.006.2014016162209.hdf'
    latitude, longitude, refArr, sunZ, senZ, sunA, senA = Init1KMMODISData(inHdf1KMFile)

    # Processing height
    inHdfDEMFile = 'tbase.hdf'
    height = GetHeightFromDEM(latitude, longitude, inHdfDEMFile)

    # Define which bands need to be calculated
    processStatus = [0]*16
    processStatus[0:3] = [1,0,1,1]
    # Calculate the scattering of hemispheric molecules, the absorption of water vapor and ozone
    sphalb, rhoray, TtotraytH2O, tOG = CalAtmoEffect(sunZ, senZ, sunA, senA, height, processStatus)
    #print sphalb[0,0,0], rhoray[0,0,0], TtotraytH2O[0,0,0], tOG[0,0,0]

    # Sampling resolution
    print 'band need to be processed ', processStatus
    refResultArr = CalRef(refArr, sphalb, rhoray, TtotraytH2O, tOG, processStatus)
    #print refArr.data[0,0,0], refResultArr[0,0,0]

    #　write result
    print 'write to HDF'
    outHDFFile = 'ref.hdf'
    WriteToHDF(outHDFFile, refResultArr, processStatus)

    print 'Completed!'