#coding=utf-8
import numpy as np

MAXNAMELENGTH = 200
Nbands = 16
DEG2RAD	= 0.0174532925199		#/* PI/180 */
UO3	= 0.319
UH2O = 2.93
REFLMIN = -0.01
REFLMAX = 1.6
ANCPATH	 = "."
DEMFILENAME = "tbase.hdf"
DEMSDSNAME = "Elevation"
REFSDS = "SOLZ"
MISSING	= -2
SATURATED = -3
MAXSOLZ = 86.5
MAXAIRMASS = 18
SCALEHEIGHT = 8000
FILL_INT16=	-32768
NUM1KMCOLPERSCAN = 1354
NUM1KMROWPERSCAN = 10
TAUSTEP4SPHALB = 0.0001
MAXNUMSPHALBVALUES = 4000		#/* with no aerosol taur <= 0.4 in all bands everywhere */

(BAND1, BAND2, BAND3, BAND4, BAND5, BAND6, BAND7, BAND8, BAND9, BAND10, BAND11, BAND12, BAND13, BAND14, BAND15, BAND16, SOLZ, SENZ, SOLA, SENA, LON, LAT, Nitems) = [0]*23

#refOffset = [316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722, 316.9722]
#refScale = [1.706693E-5, 9.6923795E-6, 6.2189442E-6, 4.826482E-6, 3.816887E-6, 2.2438335E-6, 1.0825778E-6, 2.2938268E-6, 8.3540584E-7, 2.168378E-6, 1.918186E-6, 2.0680825E-5, 3.229738E-5, 2.4188545E-5, 2.3625807E-5]
# 0 4.7466518E-5

refOffset = [0]* Nbands
refScale = [4.7466518E-5,2.603254E-5, 3.5012934E-5,3.070953E-5,3.183692E-5,3.4649496E-5,2.7887398E-5]

refBandNames = ['BAND1', 'BAND2', 'BAND3', 'BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9', 'BAND10', 'BAND11', 'BAND12', 'BAND13', 'BAND14', 'BAND15', 'BAND16']

# hdf object
class SDSObject:
    def __init__(self):
        self.name = ""
        self.filename = ""
        self.file_id = ""
        self.id = ""
        self.index = ""
        self.num_type = ""
        self.rank = ""
        self.n_attr = ""
        self.Nl = 0
        self.Np = 0
        self.plane = ""
        self.Nplanes = ""
        self.rowsperscan = ""
        self.start = ""
        self.edges = ""
        self.dim_sizes = ""
        self.data = np.array(0)
        self.fillvalue = ""
        self.factor = ""
        self.offset = ""
