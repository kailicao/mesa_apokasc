import pathlib
from importlib.resources import files
import json

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

import mesa_reader as mr
from .common import Timer, Tools
from .sim_ctr import RgbGrid


class Steps:
    '''
    Steps corresponding to the key EEPs in the reduced models.

    '''

    start = 0
    mid_PMS = 20
    ZAMS = 60; ZAMS_c = 'tab:blue'
    mid_MS = 65
    TAMS = 90; TAMS_c = 'tab:orange'

    mid_SGB = 100
    pre_FDU  = 140; pre_FDU_c  = 'tab:green'
    post_FDU = 170; post_FDU_c = 'tab:red'
    pre_RGBB  = 200; pre_RGBB_c  = 'tab:purple'
    post_RGBB = 210; post_RGBB_c = 'tab:brown'
    end = 240


class ReduceGrid:

    with open(files(__package__).joinpath('chem/isotopes.json'), 'r') as f:
        iso_weights = json.load(f)

    def __init__(self, aMLT_list: [float], mass_list: [float] = RgbGrid.MASS_MINI,
                 FeH_list: [float] = RgbGrid.FEH_MINI, verbose: bool = False,
                 visualize: bool = True, **kwargs):
        self.outdir = pathlib.Path('rgb_grid')
        assert self.outdir.exists(), 'rgb_grid does not exist'
        with open('stdmix.json', 'r') as f:
            self.stdmix = json.load(f)

        self.aMLT_list = aMLT_list
        self.mass_list = mass_list
        self.FeH_list = FeH_list
        self.kwargs = kwargs  # Ybirth, Zbirth, Z_over_X_sun, YBBN

        self.timer = Timer()
        self(verbose, visualize)

    def __call__(self, verbose: bool = False, visualize: bool = True):
        for aMLT in self.aMLT_list:
            for mass in self.mass_list:
                for FeH in self.FeH_list:
                    Y, Z = RgbGrid.Y_Z_calc(FeH, **self.kwargs)
                    ReduceModel(self, verbose, visualize,
                                aMLT=aMLT, mass=mass, Z=Z, FeH=FeH)

        print(' > All models reduced!', '@', self.timer(), flush=True)


class ReduceModel:

    QTY_LIST = ['model_number', 'star_age', 'star_mass', 'conv_mx1_bot', 'he_core_mass',
                'Teff', 'log_L', 'log_R', 'log_g', 'center_h1', 'log_Lnuc_div_L',
                'surface_X', 'surface_Y', 'surface_[Fe/H]',
                'surface_[Li/H]', 'surface_[C/Fe]', 'surface_[N/Fe]',
                'surface_[C/N]', 'surface_12C/13C']

    def __init__(self, grid: ReduceGrid, verbose: bool = False,
                 visualize: bool = True, **kwargs) -> None:
        self.grid = grid
        self.model_name = f'aMLT={kwargs["aMLT"]:.4f}_{kwargs["mass"]:.2f}M_' \
                          f'Z={kwargs["Z"]:.4f}_FeH={kwargs["FeH"]:+.2f}'
        print(' > Reducing', self.model_name, '@', grid.timer())

        self.fpath = grid.outdir / self.model_name / 'history_to_past_rgb_bump.data'
        if self.fpath.exists():
            self(verbose, visualize)
        else:
            print(' > Warning: Input file does not exist.')
            if verbose: print()

    def __call__(self, verbose: bool = False, visualize: bool = True) -> None:
        if verbose:
            print(' > ReduceModel._extract_data', '@', self.grid.timer())
        self._extract_data()

        if verbose:
            print(' > ReduceModel._reduce_data', '@', self.grid.timer())
        self._reduce_data()

        if visualize:
            if verbose:
                print(' > ReduceModel._visualize_data', '@', self.grid.timer())
            self._visualize_data()

        if verbose:
            print(' > ReduceModel._clear_data', '@', self.grid.timer(), '\n')
        self._clear_data()

    def _get_number_ratios(self, h: mr.MesaData, isos: [str]) -> np.array:
        my_num = sum([h.data(f'surface_{iso}') / ReduceGrid.iso_weights[iso] for iso in isos])
        stdnum = sum([self.grid.stdmix [iso]   / ReduceGrid.iso_weights[iso] for iso in isos])
        return my_num / stdnum

    def _extract_data(self) -> None:
        h = mr.MesaData(str(self.fpath))

        # basic quantities
        self.raw_data = {qty: h.data(qty) for qty in ReduceModel.QTY_LIST[:10]}
        # MESA conv_mx1_bot values are m/Mstar
        self.raw_data['conv_mx1_bot'] *= self.raw_data['star_mass']
        self.raw_data['log_Lnuc_div_L'] = h.log_Lnuc - h.log_L

        # surface composition
        self.raw_data['surface_X'] = sum([h.surface_h1,  h.surface_h2 ])
        self.raw_data['surface_Y'] = sum([h.surface_he3, h.surface_he4])
        surface_h  = self._get_number_ratios(h, ['h1', 'h2'  ])
        surface_li = self._get_number_ratios(h, ['li7'       ])
        surface_c  = self._get_number_ratios(h, ['c12', 'c13'])
        surface_n  = self._get_number_ratios(h, ['n14', 'n15'])
        surface_fe = h.surface_mg24 / self.grid.stdmix['mg24']

        self.raw_data['surface_[Fe/H]' ] = np.log10(surface_fe / surface_h )
        self.raw_data['surface_[Li/H]' ] = np.log10(surface_li / surface_h )
        self.raw_data['surface_[C/Fe]' ] = np.log10(surface_c  / surface_fe)
        self.raw_data['surface_[N/Fe]' ] = np.log10(surface_n  / surface_fe)
        self.raw_data['surface_[C/N]'  ] = np.log10(surface_c  / surface_n )
        self.raw_data['surface_12C/13C'] = (h.surface_c12 / ReduceGrid.iso_weights['c12']) \
                                         / (h.surface_c13 / ReduceGrid.iso_weights['c13'])

        del h, surface_h, surface_li, surface_c, surface_n, surface_fe

    def _locate_key_EEPs(self) -> None:
        # locate the main sequence
        self.ZAMS = np.where(self.raw_data['log_Lnuc_div_L'] > np.log10(0.9))[0][0]
        self.TAMS = min(np.where(self.raw_data['center_h1']    <  0.1)[0][0],
                        np.where(self.raw_data['he_core_mass'] == 0.0)[0][-1])

        # locate the first dredge-up
        surface_C_N = self.raw_data['surface_[C/N]']  # shortcut
        self.pre_FDU  = np.where(surface_C_N[0] - surface_C_N  < 1e-3)[0][-1]
        self.post_FDU = np.where(surface_C_N - surface_C_N[-1] < 1e-3)[0][0]

        # locate the RGB bump
        log_L_RGB = self.raw_data['log_L'][self.pre_FDU:]
        self.post_RGBB = np.where(np.diff(log_L_RGB) < -log_L_RGB[-1] * 5e-5)[0][-1] + 1
        self.pre_RGBB  = np.argmax(log_L_RGB[:self.post_RGBB])
        self.post_RGBB = np.argmin(log_L_RGB[self.pre_RGBB:]) + self.pre_RGBB
        self.pre_RGBB += self.pre_FDU; self.post_RGBB += self.pre_FDU

        # for reduction purposes
        self.start = 0
        self.end   = self.raw_data['model_number'][-1] - self.raw_data['model_number'][0]

        log_star_age = np.log10(self.raw_data['star_age'][0:self.ZAMS+1])
        two_thirds = (log_star_age[0] + log_star_age[-1]*2) / 3
        self.mid_PMS = np.where(log_star_age > two_thirds)[0][0]
        del log_star_age

        self.mid_MS = np.argmax(self.raw_data['log_g'][self.ZAMS:self.TAMS+1])
        self.mid_MS += self.ZAMS

        Teff = self.raw_data['Teff'][self.TAMS:self.pre_FDU+1]
        self.mid_SGB = np.argmax(Teff)
        if Teff[self.mid_SGB] - Teff[0] < Teff[0] * 1e-3:  # 2022.11.28
            one_tenth = (Teff[0]*9 + Teff[-1]) / 10
            self.mid_SGB = np.where(Teff < one_tenth)[0][0]
        self.mid_SGB += self.TAMS
        del Teff

    def _sample_stage(self, step_i: str, step_f: str, coord: np.array) -> None:
        idx_i = getattr(Steps, step_i)
        idx_f = getattr(Steps, step_f)
        sample = np.linspace(coord[0], coord[-1], idx_f - idx_i + (step_f == 'end'),
                             endpoint=(step_f == 'end'))

        for qty in ReduceModel.QTY_LIST:
            try:
                f = interp1d(coord, self.raw_data[qty]\
                             [getattr(self, step_i):getattr(self, step_f)+1], kind='slinear')
            except:
                for idx in np.where(np.diff(coord) == 0)[0]:
                    coord[idx+1] += 1e-12
                f = interp1d(coord, self.raw_data[qty]\
                             [getattr(self, step_i):getattr(self, step_f)+1], kind='slinear')
            self.data[qty][idx_i:idx_f+(step_f == 'end')] = f(sample)
            del f

    def _reduce_data(self) -> None:
        self._locate_key_EEPs()
        self.data = {qty: np.zeros(Steps.end+1) for qty in ReduceModel.QTY_LIST}

        # PMS phase 1
        self._sample_stage('start', 'mid_PMS',               
            np.log10(self.raw_data['star_age'][self.start:self.mid_PMS+1]))
        # PMS phase 2
        self._sample_stage('mid_PMS', 'ZAMS',
            np.log10(self.raw_data['star_age'][self.mid_PMS:self.ZAMS+1]) \
                   + self.raw_data['Teff']    [self.mid_PMS:self.ZAMS+1] / 1000)

        if self.mid_MS > self.ZAMS:
            # MS phase 1
            self._sample_stage('ZAMS', 'mid_MS',
                self.raw_data['log_g'][self.ZAMS:self.mid_MS+1])
            # MS phase 2
            self._sample_stage('mid_MS', 'TAMS',
                self.raw_data['star_age'][self.mid_MS:self.TAMS+1])
        else:
            self._sample_stage('ZAMS', 'TAMS',
                self.raw_data['star_age'][self.ZAMS:self.TAMS+1])

        # SGB phase 1
        star_age = self.raw_data['star_age'][self.TAMS:self.mid_SGB+1]
        Teff     = self.raw_data['Teff']    [self.TAMS:self.mid_SGB+1]
        self._sample_stage('TAMS', 'mid_SGB',
            (star_age - star_age[0])  / (star_age[-1] - star_age[0]) \
          - (Teff     - Teff    [-1]) / (Teff    [0]  - Teff    [-1]))
        # SGB phase 2
        self._sample_stage('mid_SGB', 'pre_FDU',
            self.raw_data['Teff'][self.mid_SGB:self.pre_FDU+1])

        # RGB phase 1
        self._sample_stage('pre_FDU', 'post_FDU',
            self.raw_data['he_core_mass'][self.pre_FDU:self.post_FDU+1])
        # RGB phase 2
        self._sample_stage('post_FDU', 'pre_RGBB',
            self.raw_data['he_core_mass'][self.post_FDU:self.pre_RGBB+1])
        # RGB phase 3
        self._sample_stage('pre_RGBB', 'post_RGBB',
            self.raw_data['he_core_mass'][self.pre_RGBB:self.post_RGBB+1])
        # RGB phase 4
        self._sample_stage('post_RGBB', 'end',
            self.raw_data['he_core_mass'][self.post_RGBB:self.end+1])

        df = pd.DataFrame(self.data)
        df.to_csv(self.grid.outdir / f'{self.model_name}.csv')
        del df

    def _draw_curve(self, ax, x, y):
        ax.plot(self.raw_data[x], self.raw_data[y], 'k')
        if x in ['Teff', 'log_g']: ax.invert_xaxis()
        if y in ['Teff', 'log_g']: ax.invert_yaxis()

        for EEP in ['mid_PMS', 'ZAMS', 'mid_MS', 'TAMS', 'mid_SGB',
                    'pre_FDU', 'post_FDU', 'pre_RGBB', 'post_RGBB']:
            idx = getattr(self, EEP)
            ax.plot(self.raw_data[x][idx], self.raw_data[y][idx],
                    'o', c=getattr(Steps, f'{EEP}_c', 'tab:cyan'), ms=4)

        ax.plot(self.data[x], self.data[y], 'o', c='tab:olive', ms=1)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        Tools.format_axis(ax)

    def _visualize_data(self) -> None:
        # draw evolutionary tracks
        fig, axs = plt.subplots(1, 2)
        self._draw_curve(axs[0], 'Teff', 'log_L')
        self._draw_curve(axs[1], 'Teff', 'log_g')
        Tools.save_figure(fig, 'tracks')

        # draw coordinates
        fig, axs = plt.subplots(2, 1)
        self._draw_curve(axs[0], 'star_age', 'model_number')
        self._draw_curve(axs[1], 'model_number', 'star_age')
        Tools.save_figure(fig, 'coords')

        # draw histories
        for qty in ReduceModel.QTY_LIST[2:]:
            fig, axs = plt.subplots(2, 1)
            self._draw_curve(axs[0], 'star_age',     qty)
            self._draw_curve(axs[1], 'model_number', qty)
            Tools.save_figure(fig, qty.replace('/', '_'))

        Tools.merge_plots(self.grid.outdir, self.model_name, ['tracks', 'coords'] \
                          + [qty.replace('/', '_') for qty in ReduceModel.QTY_LIST[2:]])

    def _clear_data(self) -> None:
        self.raw_data.clear()
        self.data.clear()
        del self.raw_data, self.data
