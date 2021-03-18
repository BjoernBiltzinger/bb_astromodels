# bb_astromodels

Some additional models for astromodels.

## Install

git clone https://github.com/BjoernBiltzinger/bb_astromodels.git \
cd bb_astromodels \
python setup.py install or python setup.py develop 

=> \
```python
from bb_astromodels import Absori, Integrate_Absori
```
## References

The implementations in this repository depend on work done by many other people:

### General

Depends on astromodels: https://github.com/threeML/astromodels \
Model code taken from XSPEC code: https://heasarc.gsfc.nasa.gov/xanadu/xspec/ 

### Models

Absori: https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node232.html; Magdziarz & Zdziarski (1995, MNRAS 273, 837) 

Integrated Absori/xabs: https://var.sron.nl/SPEX-doc/manualv3.05/manualse85.html; Steenbrugge et al. (2003)


