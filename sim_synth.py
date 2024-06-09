import pathlib
import json

import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline, UnivariateSpline
import matplotlib as mpl
import matplotlib.pyplot as plt

from .common import Timer, Tools
from .sim_ctr import RgbGrid
from .sim_reduce import Steps, ReduceModel


class SynthGrid:

    POPT_PATH = 'rgb_calibr/Salaris-off_vary-both.json'
    AMLT_KEY = 'ms2_mt1'
    AMLT_MODEL = lambda x, a, b1, b2, c1: a + b1*x[0] + b2*x[0]**2 + c1*x[1]

    OUT_MASS_LIST = [0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    OUT_FEH_LIST = [-0.4, -0.3, -0.2, -0.1,  0. ,  0.1,  0.2,  0.3,  0.4]
    OUT_MASS_MINI = [1. , 1.4, 1.8]
    OUT_FEH_MINI = [-0.4,  0. ,  0.4]

    def __init__(self, aMLT_list: [float], mass_list: [float] = RgbGrid.MASS_LIST,
                 FeH_list: [float] = RgbGrid.FEH_LIST, **kwargs):
        self.indir = pathlib.Path('rgb_grid')
        assert self.indir.exists(), 'rgb_grid does not exist'
        self.outdir = pathlib.Path('synth_grid')
        self.outdir.mkdir(exist_ok=True)

        popt_path = pathlib.Path(SynthGrid.POPT_PATH)
        assert popt_path.exists(), 'popt_path does not exist'
        with open(popt_path, 'r') as f:
            popt_dict = json.load(f)
        self.aMLT_popt = popt_dict[SynthGrid.AMLT_KEY]

        self.aMLT_list = aMLT_list
        self.mass_list = mass_list
        self.FeH_list = FeH_list
        self.kwargs = kwargs  # Ybirth, Zbirth, Z_over_X_sun, YBBN

        self.timer = Timer()
        self()

    def __call__(self):
        self.extract_sim_data()
        self.build_interps()

        for mass in SynthGrid.OUT_MASS_LIST:
            for FeH in SynthGrid.OUT_FEH_LIST:
                aMLT_fit = SynthGrid.AMLT_MODEL([mass-1, FeH], *self.aMLT_popt)
                self.synthesize_model(mass, FeH, aMLT_fit)

        self.clear()
        print(' > All models synthesized!', '@', self.timer(), flush=True)

    def extract_sim_data(self):
        shape = (len(self.aMLT_list), Steps.end+1,
                 len(self.mass_list), len(self.FeH_list))
        self.existence = np.ones(shape[:1] + shape[2:], dtype=bool)
        self.data_dict = {qty: np.zeros(shape) for qty in ReduceModel.QTY_LIST}

        for k, aMLT in enumerate(self.aMLT_list):
            for j, mass in enumerate(self.mass_list):
                for i, FeH in enumerate(self.FeH_list):
                    Y, Z = RgbGrid.Y_Z_calc(FeH, **self.kwargs)
                    model = SynthModel(self, aMLT=aMLT, mass=mass, Z=Z, FeH=FeH)

                    if model.exists:
                        for qty in ReduceModel.QTY_LIST:
                            self.data_dict[qty][k, :, j, i] = model.data[qty]
                    else:
                        self.existence[k, j, i] = False

                    model.clear_data(); del model

    def build_interps(self):
        self.interp_dict = {}
        for qty in ReduceModel.QTY_LIST:
            self.interp_dict[qty] = [[None for step in range(Steps.end+1)]
                                           for aMLT in self.aMLT_list]

            for k in range(len(self.aMLT_list)):
                for step in range(Steps.end+1):
                    self.interp_dict[qty][k][step] = SmoothBivariateSpline(
                        self.data_dict['star_mass']     [k, step][self.existence[k]],
                        self.data_dict['surface_[Fe/H]'][k, step][self.existence[k]],
                        self.data_dict[qty]             [k, step][self.existence[k]], kx=2, ky=2)

    def synthesize_model(self, mass: float, FeH: float, aMLT_fit: float):
        Y, Z = RgbGrid.Y_Z_calc(FeH, **self.kwargs)
        model_name = f'{mass:.2f}M_Z={Z:.4f}_FeH={FeH:+.2f}'
        print(' > Synthesizing', model_name, '@', self.timer())

        pred = {}; data = {}
        for qty in ReduceModel.QTY_LIST:
            pred[qty] = np.zeros((len(self.aMLT_list), Steps.end+1))
            data[qty] = np.zeros(Steps.end+1)

        for qty in ReduceModel.QTY_LIST:
            for step in range(Steps.end+1):
                for k, aMLT in enumerate(self.aMLT_list):
                    pred[qty] [k, step] = self.interp_dict[qty][k][step](mass, FeH)[0, 0]
                data[qty][step] = UnivariateSpline(self.aMLT_list, pred[qty][:, step], k=1)(aMLT_fit)

        if mass in SynthGrid.OUT_MASS_MINI and FeH in SynthGrid.OUT_FEH_MINI:
            self._visualize_data(model_name, pred, data, aMLT_fit)

        df = pd.DataFrame(data)
        df.to_csv(self.outdir / f'{model_name}.csv')
        pred.clear(); data.clear()
        del df, pred, data

    def _draw_curve(self, pred, data, ax, x, y, colors):
        for k, aMLT in enumerate(self.aMLT_list):
            ax.plot(pred[x][k], pred[y][k], '--', c=colors[k])
        ax.plot(data[x], data[y], '-', c=colors[-1])

        if x in ['Teff', 'log_g']: ax.invert_xaxis()
        if y in ['Teff', 'log_g']: ax.invert_yaxis()

        for step in range(Steps.end+1):
            ax.plot([pred[x][k, step] for k in range(len(self.aMLT_list))],
                    [pred[y][k, step] for k in range(len(self.aMLT_list))],
                    ls='-', lw=0.5, c='lightgrey', zorder=-1)

        for EEP in ['mid_PMS', 'ZAMS', 'mid_MS', 'TAMS', 'mid_SGB',
                    'pre_FDU', 'post_FDU', 'pre_RGBB', 'post_RGBB']:
            idx = getattr(Steps, EEP)
            color = getattr(Steps, f'{EEP}_c', 'tab:cyan')
            ax.plot([pred[x][k, idx] for k in range(len(self.aMLT_list))],
                    [pred[y][k, idx] for k in range(len(self.aMLT_list))],
                    ls='-', lw=0.5, c=color, zorder=-1)
            ax.plot(data[x][idx], data[y][idx], 'o', c=color, ms=4)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        Tools.format_axis(ax)

    def _visualize_data(self, model_name, pred, data, aMLT_fit):
        cmap = mpl.colormaps['summer_r']
        norm = mpl.colors.Normalize(vmin=self.aMLT_list[0],
                                    vmax=self.aMLT_list[-1])
        colors = [cmap(norm(a)) for a in self.aMLT_list + [aMLT_fit]]

        # draw evolutionary tracks
        fig, axs = plt.subplots(1, 2)
        self._draw_curve(pred, data, axs[0], 'Teff', 'log_L', colors)
        self._draw_curve(pred, data, axs[1], 'Teff', 'log_g', colors)
        for i in range(2):
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         ax=axs[i], orientation='horizontal', label=r'mixing length $\alpha$')
        Tools.save_figure(fig, 'tracks')

        # draw coordinates
        fig, axs = plt.subplots(2, 1)
        self._draw_curve(pred, data, axs[0], 'star_age', 'model_number', colors)
        self._draw_curve(pred, data, axs[1], 'model_number', 'star_age', colors)
        for i in range(2):
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         ax=axs[i], orientation='vertical', label=r'ML $\alpha$')
        Tools.save_figure(fig, 'coords')

        # draw histories
        for qty in ReduceModel.QTY_LIST[2:]:
            fig, axs = plt.subplots(2, 1)
            self._draw_curve(pred, data, axs[0], 'star_age',     qty, colors)
            self._draw_curve(pred, data, axs[1], 'model_number', qty, colors)
            for i in range(2):
                fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                             ax=axs[i], orientation='vertical', label=r'ML $\alpha$')
            Tools.save_figure(fig, qty.replace('/', '_'))

        Tools.merge_plots(self.outdir, model_name, ['tracks', 'coords'] \
                          + [qty.replace('/', '_') for qty in ReduceModel.QTY_LIST[2:]])

    def clear(self):
        for qty in ReduceModel.QTY_LIST:
            for k in range(len(self.aMLT_list)):
                self.interp_dict[qty][k].clear()
            self.interp_dict[qty].clear()
        self.data_dict.clear()
        self.interp_dict.clear()
        del self.existence, self.data_dict, self.interp_dict


class SynthModel:

    def __init__(self, grid: SynthGrid, **kwargs) -> None:
        self.grid = grid
        self.model_name = f'aMLT={kwargs["aMLT"]:.4f}_{kwargs["mass"]:.2f}M_' \
                          f'Z={kwargs["Z"]:.4f}_FeH={kwargs["FeH"]:+.2f}'

        fpath = grid.indir / f'{self.model_name}.csv'
        self.exists = fpath.exists()
        if not self.exists:
            print(f' > Warning: {self.model_name} does not exist.')
            return
        self.data = pd.read_csv(fpath, index_col=0)

    def clear_data(self):
        if self.exists:
            del self.data
