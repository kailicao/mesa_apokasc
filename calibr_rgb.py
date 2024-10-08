import pathlib
from importlib.resources import files

import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline, interp1d, UnivariateSpline
from scipy.optimize import fsolve

from .common import Timer
from .sim_ctr import RgbGrid
from .sim_reduce import Steps


class RgbCalibr:
    '''
    RGB mixing length calibration.

    '''

    CATALOG_PATH = 'apokasc3/Table4_Apoor_RGB_Gold.csv'
    OBS_QTY_LIST = ['KIC', 'Mass', 'S_Mass', 'Radius', 'S_Radius', 'Logg_Seis', 'S_Logg_Seis', 'Teff', 'S_Teff',
                    '[Fe/H]', 'S_[Fe/H]', '[Alp/Fe]', 'S_[Alp/Fe]', '[C/Fe]', 'S_[C/Fe]', '[N/Fe]', 'S_[N/Fe]']

    def __init__(self, aMLT_list: [float], mass_list: [float] = RgbGrid.MASS_LIST,
                 FeH_list: [float] = RgbGrid.FEH_LIST, vary: str = 'both',
                 Salaris: bool = False, **kwargs):
        self.indir = pathlib.Path('rgb_grid')
        assert self.indir.exists(), 'rgb_grid does not exist'
        self.outdir = pathlib.Path('rgb_calibr')
        self.outdir.mkdir(exist_ok=True)

        self.aMLT_list = aMLT_list
        self.mass_list = mass_list
        self.FeH_list = FeH_list
        self.kwargs = kwargs  # Ybirth, Zbirth, Z_over_X_sun, YBBN

        self.timer = Timer()
        self(vary, Salaris)

    def __call__(self, vary: str = 'both', Salaris: bool = False):
        self.extract_sim_data(vary)
        self.build_interps()
        self.load_obs_data(Salaris)
        self.get_amlt_opt(vary, Salaris)
        self.clear()
        print(' > Finished calibrating all stars!', '@', self.timer(), flush=True)

    def extract_sim_data(self, vary: str = 'both'):
        shape = (len(self.aMLT_list), Steps.pre_RGBB-Steps.pre_FDU+1,
                 len(self.mass_list), len(self.FeH_list))
        self.existence = np.ones(shape[:1] + shape[2:], dtype=bool)
        self.Teff_data = np.zeros(shape)
        self.logg_data = np.zeros(shape)
        self.mass_data = np.zeros(shape)
        self.FeH_data  = np.zeros(shape)

        for k, aMLT in enumerate(self.aMLT_list):
            for j, mass in enumerate(self.mass_list):
                for i, FeH in enumerate(self.FeH_list):
                    Y, Z = RgbGrid.Y_Z_calc(FeH, **self.kwargs)
                    model = RgbModel(self, vary=vary, aMLT=aMLT, mass=mass, Z=Z, FeH=FeH)

                    if model.exists:
                        self.Teff_data[k, :, j, i] = model.Teff
                        self.logg_data[k, :, j, i] = model.logg
                        self.mass_data[k, :, j, i] = model.mass
                        self.FeH_data [k, :, j, i] = model.FeH
                    else:
                        self.existence[k, j, i] = False

                    model.clear_data(); del model

    def build_interps(self):
        self.Teff_interps = [[None for step in range(Steps.pre_RGBB-Steps.pre_FDU+1)]
                                   for aMLT in self.aMLT_list]
        self.logg_interps = [[None for step in range(Steps.pre_RGBB-Steps.pre_FDU+1)]
                                   for aMLT in self.aMLT_list]

        for k in range(len(self.aMLT_list)):
            for step in range(Steps.pre_RGBB-Steps.pre_FDU+1):
                self.Teff_interps[k][step] = SmoothBivariateSpline(
                    self.mass_data[k, step][self.existence[k]],
                    self.FeH_data [k, step][self.existence[k]],
                    self.Teff_data[k, step][self.existence[k]], kx=2, ky=2)
                self.logg_interps[k][step] = SmoothBivariateSpline(
                    self.mass_data[k, step][self.existence[k]],
                    self.FeH_data [k, step][self.existence[k]],
                    self.logg_data[k, step][self.existence[k]], kx=2, ky=2)

    def load_obs_data(self, Salaris: bool = False):
        self.apokasc3 = pd.read_csv(files(__package__).joinpath(RgbCalibr.CATALOG_PATH))  #,
        #                            index_col='KIC')  # [RgbCalibr.OBS_QTY_LIST]
        lower_rgb = self.apokasc3['Teff'] + self.apokasc3['Logg_Seis'] * 1000.0 > 7500.0
        self.apokasc3 = self.apokasc3[lower_rgb]; del lower_rgb

    def get_amlt_opt(self, vary: str = 'both', Salaris: bool = False):
        self.apokasc3['AMLT_OPT'] = np.nan

        for i, star in self.apokasc3.iterrows():
            mass  = star['Mass']
            metal = star['[Fe/H]']

            Teff_pred = np.zeros(len(self.aMLT_list))
            for k in range(len(self.aMLT_list)):
                Teff_arr = [interp(mass, metal)[0, 0] for interp in self.Teff_interps[k][::-1]]
                logg_arr = [interp(mass, metal)[0, 0] for interp in self.logg_interps[k][::-1]]
                Teff_pred[k] = interp1d(logg_arr, Teff_arr, kind='slinear',
                                        fill_value='extrapolate')(star['Logg_Seis'])
                Teff_arr.clear(); logg_arr.clear(); del Teff_arr, logg_arr

            f = UnivariateSpline(self.aMLT_list, Teff_pred, k=1)
            self.apokasc3.at[i, 'AMLT_OPT'] = fsolve(lambda x: f(x[0]) - star['Teff'], 2.0)[0]

        fname = 'Salaris-' + ('on' if Salaris else 'off') + '_vary-' + vary + '.csv'
        self.apokasc3.to_csv(self.outdir / fname)

    def clear(self):
        del self.existence, self.Teff_data, self.logg_data, self.mass_data, self.FeH_data
        for k in range(len(self.aMLT_list)):
            self.Teff_interps[k].clear()
            self.logg_interps[k].clear()
        del self.Teff_interps, self.logg_interps
        del self.apokasc3


class RgbModel:
    '''
    Data structure for an RGB model.

    '''

    SIM_QTY_LIST = ['Teff', 'log_g', 'star_mass', 'surface_[Fe/H]']
    MASS_LOSS_COEF = 1.0  # M_node = (1.0 - coef) * M_init + coef * M_MESA

    def __init__(self, calibr: RgbCalibr, vary: str = 'both', **kwargs):
        model_name = f'aMLT={kwargs["aMLT"]:.4f}_{kwargs["mass"]:.2f}M_' \
                     f'Z={kwargs["Z"]:.4f}_FeH={kwargs["FeH"]:+.2f}'
        fpath = calibr.indir / f'{model_name}.csv'
        self.exists = fpath.exists()
        if not self.exists: return
        data = pd.read_csv(fpath, index_col=0)[RgbModel.SIM_QTY_LIST]

        self.Teff = data['Teff' ][Steps.pre_FDU:Steps.pre_RGBB+1].copy()
        self.logg = data['log_g'][Steps.pre_FDU:Steps.pre_RGBB+1].copy()

        if vary in ['both', 'mass']:
            self.mass = data['star_mass'][Steps.pre_FDU:Steps.pre_RGBB+1].copy()

            coef = RgbModel.MASS_LOSS_COEF  # short cut
            if coef != 1.0:
                self.mass *= coef
                self.mass += (1.0 - coef) * kwargs["mass"]
        else:
            self.mass = kwargs["mass"]

        if vary in ['both', 'FeH']:
            self.FeH = data['surface_[Fe/H]'][Steps.pre_FDU:Steps.pre_RGBB+1].copy()
        else:
            self.FeH = kwargs["FeH"]

        del data

    def clear_data(self):
        if self.exists:
            del self.Teff, self.logg, self.mass, self.FeH
