import os
import sys
from functools import lru_cache, wraps
from collections import OrderedDict

import astropy.units as astropy_units
import numpy as np
import six
from astromodels.functions.function import Function1D, FunctionMeta
from astromodels.utils import configuration
from astropy.io import fits
from numba import njit
from scipy.interpolate import interp1d

from bb_astromodels.utils.cache import cache_array_method
from bb_astromodels.utils.data_files import _get_data_file_path
from bb_astromodels.utils.numba_functions import calc_ion_spec_numba


class Absori(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Ionized medium absorption (absori implementation from xspec),
        f(E) = exp(- NH * sigma(E, temp, xi, gamma, abundance))
        contributed by Björn Biltzinger
    parameters :
        NH :
            desc : absorbing column density in units of 1e22 particles per cm^2
            initial value : 1.0
            is_normalization : False
            transformation : log10
            min : 1e-4
            max : 1e4
            delta : 0.1

        redshift :
            desc : the redshift of the source
            initial value : 0.
            is_normalization : False
            min : 0
            max : 15
            delta : 0.1
            fix: True

        temp :
            desc : temperture of the gas in K
            initial value : 10000.0
            is_normalization : False
            transformation : log10
            min : 1e2
            max : 1e9
            delta : 0.1

        xi :
            desc : absorber ionization state =L/nR^2
            initial value : 1.0
            is_normalization : False
            transformation : log10
            min : 0.1
            max : 1e3
            delta : 0.1

        gamma :
            desc : photon index of the ionizing spectrum
            initial value : 2.0
            is_normalization : False
            min : 0
            max : 5
            delta : 0.1

        abundance:
            desc : Abundance of elements >He and <Fe compared to base model (log10(n/n_basemodel))
            initial value : 0.0
            is_normalization : False
            min : -10
            max : 10
            delta : 0.1

        fe_abundance :
            desc : Abundance of Fe compared to base model (log10(nFe/nFe_basemodel))
            initial value : 0.0
            is_normalization : False
            min : -10
            max : 10
            delta : 0.1

    """

    def _setup(self):
        self._fixed_units = (
            astropy_units.keV,
            astropy_units.dimensionless_unscaled,
        )

        self._cache_ion_spec = OrderedDict()
        self._cache_num = OrderedDict()

        # the elements in this model
        self._absori_elements = [
            "H",
            "He",
            "C",
            "N",
            "O",
            "Ne",
            "Mg",
            "Si",
            "S",
            "Fe",
        ]

        # load database for absori
        (
            self._ion,
            self._sigma,
            self._atomicnumber,
            self._base_energy,
        ) = self._load_sigma()

        self._max_atomicnumber = int(np.max(self._atomicnumber))

        self._sigma = self._sigma.T

        self._base_energy = np.array(self._base_energy, dtype=float)

        # precalc two masks needed in the calc
        # valid entries of the num matrix
        self._mask_valid = np.zeros_like(self._ion[:, :, 0], dtype=bool)
        for i, n in enumerate(self._atomicnumber):
            self._mask_valid[i, :n] = True

        self._mask_2 = np.zeros_like(self._ion[:, :, 0], dtype=bool)
        for i, n in enumerate(self._atomicnumber - 1):
            self._mask_2[i, n] = True

        # build the interpolation of sigma
        self._interp_sigma = interp1d(self._base_energy, self._sigma, axis=0)

        # precalc the "deltaE" per ebin in the base energy
        self._deltaE = np.zeros(len(self._base_energy))
        self._deltaE[0] = self._base_energy[1] - self._base_energy[0]
        self._deltaE[-1] = self._base_energy[-1] - self._base_energy[-2]
        self._deltaE[1:-1] = (
            self._base_energy[2:] - self._base_energy[0:-2]
        ) / 2

        # load abundance
        self._abundance = self._load_abundance()

    def _load_sigma(self):
        """
        Load the base data for absori.
        Not the most efficient way but only needed
        in the precalc.
        """
        ion = np.zeros((10, 26, 10))
        sigma = np.zeros((10, 26, 721))
        atomicnumber = np.empty(10, dtype=int)

        with fits.open(
            _get_data_file_path(os.path.join("ionized", "mansig.fits"))
        ) as f:
            znumber = f["SIGMAS"].data["Z"]
            ionnumber = f["SIGMAS"].data["ION"]
            sigmadata = f["SIGMAS"].data["SIGMA"]
            iondata = f["SIGMAS"].data["IONDATA"]

            energy = f["ENERGIES"].data["ENERGY"]

        currentZ = -1
        iZ = -1
        iIon = -1
        for i in range(len(znumber)):
            if znumber[i] != currentZ:
                iZ += 1
                atomicnumber[iZ] = znumber[i]
                currentZ = znumber[i]
                iIon = -1
            iIon += 1
            for k in range(10):
                ion[iZ, iIon, k] = iondata[i][k]

            # change units of coef

            ion[iZ][iIon][1] *= 1.0e10
            ion[iZ][iIon][3] *= 1.0e04
            ion[iZ][iIon][4] *= 1.0e-04
            ion[iZ][iIon][6] *= 1.0e-04

            for k in range(721):
                sigma[iZ][iIon][k] = sigmadata[i][k] / 6.6e-27

        return ion, sigma, atomicnumber, energy

    def _load_abundance(self, model="angr"):
        """
        Load the base abundance for the given model.
        Only needed in the precalc.
        """
        with open(
            _get_data_file_path(os.path.join("abundance", "abundances.dat"))
        ) as f:
            rows = f.readlines()
            ele = np.array(rows[0].split(" "), dtype=str)
            ele = ele[ele != ""][1:]
            # get rid of \n at the end
            ele[-1] = ele[-1][:2]
            vals = np.zeros((7, len(ele)))
            keys = []
            for i, row in enumerate(rows[1:8]):
                l = np.array(row.split(" "), dtype=str)
                l = l[l != ""]
                # get rid of \n at the end
                if l[-1][-2:] == "\n":
                    l[-1] = l[-1][:2]
                if l[-1] == "\n":
                    l = l[:-1]
                vals[i] = np.array(l[1:], dtype=float)
                keys.append(l[0][:-1])
            keys = np.array(keys)
        vals_all = np.zeros(len(self._absori_elements))
        for i, element in enumerate(self._absori_elements):
            assert (
                element in ele
            ), f"{element} not a valid element. Valid elements: {ele}"

            idx = np.argwhere(ele == element)[0, 0]

            assert (
                model in keys
            ), f"{model} not a valid name. Valid names: {keys}"

            idy = np.argwhere(keys == model)[0, 0]

            vals_all[i] = vals[idy, idx]

        return vals_all

    def _set_units(self, x_unit, y_unit):

        self.NH.unit = astropy_units.cm ** (-2)
        self.redshift.unit = astropy_units.dimensionless_unscaled
        self.temp.unit = astropy_units.K
        self.gamma.unit = astropy_units.dimensionless_unscaled
        self.xi.unit = astropy_units.dimensionless_unscaled
        self.abundance.unit = astropy_units.dimensionless_unscaled
        self.fe_abundance.unit = astropy_units.dimensionless_unscaled

    def evaluate(
        self, x, NH, redshift, temp, xi, gamma, abundance, fe_abundance
    ):
        # calc energies with z
        e = x * (1 + redshift)
        # calc opacity
        opacity = self._calc_opacity(
            e, temp, xi, gamma, abundance, fe_abundance
        )

        return np.exp(-NH * opacity)

    # @cache_array_method(maxsize=1)
    def _calc_opacity(self, e, temp, xi, gamma, abundance, fe_abundance):
        """
        Calculate the opacity for the given parameters and energies
        """

        # calc the ionizing spectrum
        # spec = self._calc_ion_spec(gamma)

        # get the num matrix
        num = self._calc_num(gamma, temp, xi)

        # get abundance TODO check this
        ab = np.copy(self._abundance)
        ab[2:-1] *= 10 ** abundance  # for elements>He
        ab[-1] *= 10 ** fe_abundance  # for iron

        # weight num by abundance
        num *= ab
        # interpolate sigma for the given e values
        sigma = self._interpolate_sigma(e)

        # multiply together and sum
        return np.sum(num * sigma, axis=(1, 2)) * 6.6e-5

    # @cache_array_method(maxsize=1000)
    def _interpolate_sigma(self, ekev):
        """
        Interpolate sigma for the e values
        """
        e = 1000 * ekev

        sigma = np.zeros((len(e), self._sigma.shape[1], self._sigma.shape[2]))

        # we have to split in three parts. e>max(base_energy)
        # and e<min(base_energy) and rest
        mask1 = e > self._base_energy[-1]
        mask2 = e < self._base_energy[0]

        mask3 = (~mask1) * (~mask2)
        # for mask true use simple interpolation between
        # the base energy values

        sigma[mask3] = self._interp_sigma(e[mask3])

        # for mask false extend the sigma at the highest energy base value with
        # a powerlaw with slope -3

        sigma[mask1] = self._sigma[720]
        sigma[mask1] *= np.expand_dims(
            np.power((e[mask1] / self._base_energy[-1]), -3.0), axis=(1, 2)
        )

        sigma[mask2] = self._sigma[0]

        return sigma

    #    @lru_cache(maxsize=4)
    def _calc_ion_spec(self, gamma):
        """
        Calc the F(E)*deltaE at the grid energies of the base energies.
        """
        try:

            value = self._cache_ion_spec[gamma]
            self._cache_ion_spec.move_to_end(gamma)

        except KeyError:

            value = calc_ion_spec_numba(gamma, self._base_energy, self._deltaE)

            if len(self._cache_ion_spec) > 4:

                self._cache_ion_spec.popitem(False)

            self._cache_ion_spec[gamma] = value

        return value

    #   @lru_cache(maxsize=4)
    def _calc_num(self, gamma, temp, xi):
        """
        Calc the num matrix. I don't really understand most of this. I copied the code
        from xspec and vectrorized most of the calc for speed. Tested to give the same result
        like xspec.
        """
        key = f"{gamma}_{temp}_{xi}"

        try:

            value = self._cache_num[key]
            self._cache_num.move_to_end(key)

        except KeyError:

            spec = self._calc_ion_spec(gamma)
            # transform temp to units of 10**4 K
            t4 = 0.0001 * temp
            tfact = 1.033e-3 / np.sqrt(t4)

            # log of xi
            if xi <= 0:
                xil = -100.0
            else:
                xil = np.log(xi)

            num = np.zeros((self._max_atomicnumber, len(self._atomicnumber)))

            # loop over all types of atoms in the model
            e1 = np.exp(-self._ion[:, :, 4] / t4)
            e2 = np.exp(-self._ion[:, :, 6] / t4)
            arec = self._ion[:, :, 1] * np.power(
                t4, -self._ion[:, :, 2]
            ) + self._ion[:, :, 3] * np.power(t4, -1.5) * e1 * (
                1.0 + self._ion[:, :, 5] * e2
            )
            z2 = self._atomicnumber ** 2
            y = 15.8 * z2 / t4
            arec2 = tfact * z2 * (1.735 + np.log(y) + 1 / (6.0 * y))
            arec[self._mask_2] = arec2

            intgral = np.sum(self._sigma.T * spec, axis=2)

            ratio = np.zeros_like(arec)

            ratio[arec != 0] = np.log(
                3.2749e-6 * intgral[arec != 0] / arec[arec != 0]
            )
            # ratio = np.log(3.2749e-6*intgral/arec)
            # ratio[arec == 0] = 0
            ratcumsum = np.cumsum(ratio, axis=1)

            mul = ratcumsum + (np.arange(1, self._max_atomicnumber + 1)) * xil
            mul[~self._mask_valid] = -(10 ** 99)
            mult = np.max(mul, axis=1)
            mul = (mul.T - mult).T
            emul = np.exp(mul)
            emul[~self._mask_valid] = 0

            s = np.sum(emul, axis=1)

            s += np.exp(-mult)
            num[0] = -mult - np.log(s)
            for j in range(1, 26):
                num[j] = num[j - 1] + ratio[:, j - 1] + xil

            num = np.exp(num)
            num[~self._mask_valid.T] = 0

            value = num

            if len(self._cache_num) > 4:

                self._cache_num.popitem(False)

            self._cache_num[key] = value

        return value


class Integrate_Absori(Absori, metaclass=FunctionMeta):
    r"""
    description :
        Integrate ionized medium absorption (absori implementation from xspec) over redshift
        contributed by Björn Biltzinger
    parameters :
        n0 :
            desc : local particle density of IGM in units of cm^-3
            initial value : 1e-4
            is_normalization : False
            transformation : log10
            min : 1e-9
            max : 1
            delta : 0.1

        delta :
            desc: delta parameter for the density evolution with z => n(r)=n0*(1+z)^delta
            initial value : 0
            is_normalization : False
            min : -10
            max : 10
            delta : 0.1
            fix: True

        redshift :
            desc : the redshift of the source
            initial value : 0.
            is_normalization : False
            min : 0
            max : 15
            delta : 0.1
            fix: True

        temp :
            desc : temperture of the IGM in K
            initial value : 10000.0
            is_normalization : False
            transformation : log10
            min : 1e2
            max : 1e9
            delta : 0.1

        xi :
            desc : absorber ionization state of the IGM =L/nR^2
            initial value : 1.0
            is_normalization : False
            transformation : log10
            min : 0.1
            max : 1e3
            delta : 0.1

        gamma :
            desc : photon index of the ionizing spectrum
            initial value : 2.0
            is_normalization : False
            min : 0
            max : 5
            delta : 0.1

        abundance:
            desc : Abundance of elements >He and <Fe compared to base model (log10(n/n_basemodel))
            initial value : 0.0
            is_normalization : False
            min : -10
            max : 10
            delta : 0.1

        fe_abundance :
            desc : Abundance of Fe compared to base model (log10(nFe/nFe_basemodel))
            initial value : 0.0
            is_normalization : False
            min : -10
            max : 10
            delta : 0.1

    """

    def _setup(self):
        super(Integrate_Absori, self)._setup()
        self._omegam = 0.307
        self._omegal = 0.693
        self._h0 = 67.7
        self._cmpermpc = 3.08568e24
        self._c = 2.99792458e5
        self._last_x_sum = 0
        self._last_z = None
        # self._2last_x_sum = 0
        # self._2last_z = None
        # self._3last_x_sum = 0
        # self._3last_z = None
        self._last_sigma_all = None
        # self._last_sigma_all2 = None
        # self._last_sigma_all3 = None
        self._xsec_precalc = None
        # self._xsec_precalc2 = None
        # self._xsec_precalc3 = None
        self._last_gamma = None
        self._last_temp = None
        self._last_xi = None
        # if self.redshift.fixed == True:
        #    # precalc the sigma interpolation
        #    nz = int(self.redshift/0.02)
        #    zsam = self.redshift/nz
        #    zz = zsam*0.5
        #    sigma = np.zeros((nz, len(e), self._sigma.shape[1], self._sigma.shape[2]))
        #    for i in range(nz):
        #        z1 = zz+1.0
        #        sigma[i] = self._interpolate_sigma(x*z1)

    def _set_units(self, x_unit, y_unit):
        self.n0.unit = astropy_units.cm ** (-3)
        self.delta.unit = astropy_units.dimensionless_unscaled
        self.redshift.unit = astropy_units.dimensionless_unscaled
        self.temp.unit = astropy_units.K
        self.gamma.unit = astropy_units.dimensionless_unscaled
        self.xi.unit = (
            astropy_units.erg * astropy_units.cm * astropy_units.s ** (-1)
        )
        self.abundance.unit = astropy_units.dimensionless_unscaled
        self.fe_abundance.unit = astropy_units.dimensionless_unscaled

    @lru_cache(maxsize=5)
    def _interpolate_sigma_all(self, ekev):
        print("Hallo")
        sigma_all = np.zeros(
            (self._nz, len(ekev), self._sigma.shape[1], self._sigma.shape[2])
        )
        zz = 0.5 * self._zsam
        for i in range(self._nz):
            z1 = zz + 1.0
            sigma_all[i] = self._interpolate_sigma(ekev * z1)
            zz += self._zsam
        return sigma_all

    def evaluate(
        self, x, n0, delta, redshift, temp, xi, gamma, abundance, fe_abundance
    ):
        # define z shells
        nz = int(redshift / 0.02)
        zsam = redshift / nz
        zz = zsam * 0.5
        # spec = self._calc_ion_spec(gamma)
        num = self._calc_num(gamma, temp, xi)

        # get abundance TODO check this
        ab = np.copy(self._abundance)
        ab[2:-1] *= 10 ** abundance  # for elements>He
        ab[-1] *= 10 ** fe_abundance  # for iron

        # weight num by abundance
        num_ab = num * ab
        # array with the taus for alle energies
        taus = np.zeros(len(x))

        # self._z = redshift
        # self._nz = nz
        # self._zsam = zsam
        # sigma_all = self._interpolate_sigma_all(x)

        ################## Some kind of caching of the last calls (last 3 for the sigma interp, because 3ML calls the function on the edges and on the middle of the ebins for
        ################## simpson integration)... Have to improve this at some point

        if (
            self._last_gamma == gamma
            and self._last_temp == temp
            and self._last_xi == xi
        ):
            new_num = False
        else:
            new_num = True
            self._last_gamma = gamma
            self._last_temp = temp
            self._last_xi = xi

        new_sigma_interp = True

        sum_x = np.sum(x)
        # if np.all(self._last_x == x):
        if self._last_x_sum == sum_x:
            if self._last_z == redshift:
                # everything stayed the same
                # sigma_all = self._last_sigma_all
                # xsec_precalc = self._xsec_precalc
                new_sigma_interp = False
        if new_sigma_interp:
            self._last_z = redshift
            self._last_x_sum = sum_x
        # if np.all(self._2last_x == x):
        # if self._2last_x_sum == sum_x:
        #    if self._2last_z == redshift:
        #        #everything stayed the same
        #        #xsec_precalc = self._xsec_precalc2
        #        sigma_all = self._last_sigma_all2
        #        new_sigma_interp = False
        # if np.all(self._3last_x == x):
        # if self._3last_x_sum == sum_x:
        #    if self._3last_z == redshift:
        #        #everything stayed the same
        #        sigma_all = self._last_sigma_all3
        #        #xsec_precalc = self._xsec_precalc3
        #        new_sigma_interp = False
        if new_sigma_interp or new_num:
            # self._3last_z = self._2last_z
            # self._3last_x_sum = self._2last_x_sum
            # self._2last_z = self._last_z
            # self._2last_x_sum = self._last_x_sum
            # something changed we have to recalc the sigma interpolation
            # self._last_sigma_all3 = self._last_sigma_all2
            # self._last_sigma_all2 = self._last_sigma_all
            if new_sigma_interp:
                self._last_sigma_all = np.zeros(
                    (nz, len(x), self._sigma.shape[1], self._sigma.shape[2])
                )

            # self._xsec_precalc3 = self._xsec_precalc2
            # self._xsec_precalc2 = self._xsec_precalc
            self._xsec_precalc = np.zeros((nz, len(x)))
            for i in range(nz):
                z1 = zz + 1.0
                if new_sigma_interp:
                    self._last_sigma_all[i] = self._interpolate_sigma(x * z1)
                    self._xsec_precalc[i] = (
                        np.sum(num_ab * self._last_sigma_all[i], axis=(1, 2))
                        * 6.6e-5
                        * 1e-22
                    )
                else:

                    self._xsec_precalc[i] = (
                        np.sum(num_ab * self._last_sigma_all[i], axis=(1, 2))
                        * 6.6e-5
                        * 1e-22
                    )
                # self._last_sigma_all[i] = self._interpolate_sigma(x*z1)
                zz += zsam
            # xsec_precalc = self._xsec_precalc
            # sigma_all = self._last_sigma_all
            zz = zsam * 0.5
        xsec_precalc = self._xsec_precalc

        #################################################

        for i in range(nz):
            z1 = zz + 1.0
            # n in this shell
            n = n0 * z1 ** delta
            zf = z1 ** 2 / np.sqrt(self._omegam * z1 ** 3 + self._omegal)
            zf *= zsam * self._c * n * self._cmpermpc / self._h0

            # sigma = self._interpolate_sigma(x)
            # xsec = np.sum(num_ab*sigma, axis=(1, 2))*6.6e-5*1e-22

            taus += xsec_precalc[i] * zf
            zz += zsam

        return np.exp(-taus)
