# bb_astromodels

Some additional models for astromodels.

## Install

git clone https://github.com/BjoernBiltzinger/bb_astromodels.git \n
cd bb_astromodels \n
python setup.py install or python setup.py develop \n

=> \n
\t from bb_astromodels.xray.absorption import absori, integrate_absori

## References

The implementations in this repository depend on work done by many other people:

### General

Depends on astromodels: https://github.com/threeML/astromodels \n
Model code taken from XSPEC code: https://heasarc.gsfc.nasa.gov/xanadu/xspec/ \n

### Models:

Absori: https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node232.html; Magdziarz & Zdziarski (1995, MNRAS 273, 837) \n

Integrated Absori/xabs: https://var.sron.nl/SPEX-doc/manualv3.05/manualse85.html; Steenbrugge et al. (2003)\n


