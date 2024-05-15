import pathlib

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
#                  'font.size': 12, 'text.latex.preamble': r"\usepackage{amsmath}",
#                  'xtick.major.pad': 2, 'ytick.major.pad': 2, 'xtick.major.size': 6, 'ytick.major.size': 6,
#                  'xtick.minor.size': 3, 'ytick.minor.size': 3, 'axes.linewidth': 2, 'axes.labelpad': 1})

import mesa_reader as mr
from .common import Timer, Tools
from .sim_ctr import SunGrid


class SunCalibr:
    '''
    Solar calibration.

    '''

    Teff_target  = 5772.0
    Teff_sigma   =   65.0
    log_L_target = 0.00
    log_L_sigma  = 0.05

    def __init__(self, Z_over_X_sun: float = 0.02292, round_: int = 0, Z: float = 0.0187,
                 Y_list: [float] = SunGrid.Y_LIST, aMLT_list: [float] = SunGrid.AMLT_LIST):
        self.output_dir = pathlib.Path('sun_grid')
        assert self.output_dir.exists(), 'sun_grid does not exist'

        self.Z_over_X_sun = Z_over_X_sun
        self.round_ = round_
        self.Z = Z
        self.Y_list = Y_list
        self.aMLT_list = aMLT_list

        self.timer = Timer()
        self()

    def __call__(self, resolution: int = 201):
        print(f' > Round {self.round_} solar calibration:')
        self.build_interps()
        self.make_maps(resolution)
        self.draw_maps(resolution)
        self.clear_arrays()
        print(f' > Round {self.round_} solar calibration completed!', '@', self.timer())

    def build_interps(self):
        shape = (len(self.Y_list), len(self.aMLT_list))
        Teff_arr  = np.zeros(shape)
        log_L_arr = np.zeros(shape)
        surface_X_arr = np.zeros(shape)
        surface_Y_arr = np.zeros(shape)

        for j, Y in enumerate(self.Y_list):
            for i, aMLT in enumerate(self.aMLT_list):
                model = SunModel(self, aMLT, Y)
                Teff_arr [j, i] = model.Teff
                log_L_arr[j, i] = model.log_L
                surface_X_arr[j, i] = model.surface_X
                surface_Y_arr[j, i] = model.surface_Y
                del model

        self.interps = {}
        for qty in ['Teff', 'log_L', 'surface_X', 'surface_Y']:
            self.interps[qty] = RectBivariateSpline(
                self.Y_list, self.aMLT_list, locals()[f'{qty}_arr'],
                kx=min(len(self.Y_list)-1, 3), ky=min(len(self.aMLT_list)-1, 3))
        del Teff_arr, log_L_arr, surface_X_arr, surface_Y_arr

    @staticmethod
    def chi2_calc(Teff: float, log_L: float):
        return np.square((Teff  - SunCalibr.Teff_target)  / SunCalibr.Teff_sigma)\
             + np.square((log_L - SunCalibr.log_L_target) / SunCalibr.log_L_sigma)

    def make_maps(self, resolution: int = 201):
        self.Y_fine    = np.linspace(self.Y_list   [0], self.Y_list   [-1], resolution)
        self.aMLT_fine = np.linspace(self.aMLT_list[0], self.aMLT_list[-1], resolution)

        self.Teff_map  = self.interps['Teff'] (self.Y_fine, self.aMLT_fine, grid=True)
        self.log_L_map = self.interps['log_L'](self.Y_fine, self.aMLT_fine, grid=True)
        self.chi2_map  = SunCalibr.chi2_calc(self.Teff_map, self.log_L_map)

        def chi2_func(Y_aMLT: [float, float]):
            Teff = self.interps['Teff'] (*Y_aMLT)[0, 0]
            logL = self.interps['log_L'](*Y_aMLT)[0, 0]
            return SunCalibr.chi2_calc(Teff, logL)

        # find optimal Y and aMLT via 2-d minimization
        res = minimize(chi2_func, x0=[np.median(self.Y_list), np.median(self.aMLT_list)],
                       bounds=[(self.Y_list[0],    self.Y_list[-1]),
                               (self.aMLT_list[0], self.aMLT_list[-1])])
        self.Y_opt, self.aMLT_opt = res.x[0], res.x[1]
        print(f' > {self.Y_opt = }, {self.aMLT_opt = }')

        self.Teff_opt  = self.interps['Teff'] (self.Y_opt, self.aMLT_opt)[0, 0]
        self.log_L_opt = self.interps['log_L'](self.Y_opt, self.aMLT_opt)[0, 0]
        self.chi2_opt  = chi2_func((self.Y_opt, self.aMLT_opt))

        surface_X_map = self.interps['surface_X'](self.Y_fine, self.aMLT_fine, grid=True)
        surface_Y_map = self.interps['surface_Y'](self.Y_fine, self.aMLT_fine, grid=True)
        self.Z_over_X_map = (1.0 - surface_X_map - surface_Y_map) / surface_X_map
        del surface_X_map, surface_Y_map

        surface_X_opt = self.interps['surface_X'](self.Y_opt, self.aMLT_opt)[0, 0]
        surface_Y_opt = self.interps['surface_Y'](self.Y_opt, self.aMLT_opt)[0, 0]
        self.Z_over_X_opt = (1.0 - surface_X_opt - surface_Y_opt) / surface_X_opt
        print(' > Next Z value to try:', self.Z * self.Z_over_X_sun / self.Z_over_X_opt)

        # with open(str(self.output_dir / f'round={self.round_}_Z={self.Z:.4f}.json'), 'w') as f:
        #     json.dump({'Y_opt': self.Y_opt, 'aMLT_opt': self.aMLT_opt,
        #                'Z_next': self.Z * self.Z_over_X_sun / self.Z_over_X_opt}, f)

    def draw_maps(self, resolution: int = 201):
        plots = []

        for qty in ['Teff', 'log_L', 'chi2', 'Z_over_X']:
            fig, ax = plt.subplots()

            map_data = getattr(self, f'{qty}_map')
            im = ax.imshow(map_data, origin='lower', aspect='auto',
                           extent=(self.aMLT_list[0], self.aMLT_list[-1],
                                   self.Y_list[0],    self.Y_list[-1]))
            cbar = fig.colorbar(im, ax=ax)

            val_opt = getattr(self, f'{qty}_opt')
            ax.plot(self.aMLT_opt, self.Y_opt, 'rx', label=f'{qty}_opt$ = {val_opt:.6f}$')
            ax.legend(framealpha=0.3)

            if qty == 'Z_over_X':
                plt.clabel(ax.contour(
                    self.aMLT_fine, self.Y_fine, map_data, colors='k', linestyles='--',
                    levels=list(self.Z_over_X_sun * np.linspace(0.99, 1.01, 3))))

            ax.set_xticks(self.aMLT_list)
            ax.set_yticks(self.Y_list)
            ax.set_xlabel(r'mixing length $\alpha$')
            ax.set_ylabel('initial helium abundance $Y$')
            cbar.set_label(qty)

            figname = f'round_{self.round_}_{qty}_map'
            plots.append(figname)
            ax.set_title(figname)
            Tools.save_figure(fig, figname)

        Tools.merge_plots(self.output_dir, f'round={self.round_}_Z={self.Z:.4f}', plots)

    def clear_arrays(self):
        self.interps.clear(); del self.interps
        del self.Y_fine, self.aMLT_fine
        del self.Teff_map, self.log_L_map, self.chi2_map, self.Z_over_X_map


class SunModel:
    '''
    Data structure for a solar model.

    '''

    def __init__(self, calibr: SunCalibr, aMLT: float, Y: float):
        self.calibr = calibr
        self.aMLT = aMLT
        self.Y = Y

        self.extract_data()

    def extract_data(self):
        model_name = f'round={self.calibr.round_}_Z={self.calibr.Z:.4f}_Y={self.Y:.4f}_aMLT={self.aMLT:.4f}'
        histfile = self.calibr.output_dir / model_name / 'history_to_solar_age.data'
        assert histfile.exists(), 'input file does not exist'

        h = mr.MesaData(str(histfile))
        self.Teff  = h.Teff [-1]
        self.log_L = h.log_L[-1]
        self.surface_X = h.surface_h1 [-1] + h.surface_h2 [-1]
        self.surface_Y = h.surface_he3[-1] + h.surface_he4[-1]
