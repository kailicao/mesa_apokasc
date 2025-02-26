import pathlib

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .common import Timer
from .sim_ctr import RgbGrid
from .sim_reduce import Steps
from .sim_synth import SynthGrid


class AnalGrid:

    def __init__(self, **kwargs) -> None:
        self.indir = pathlib.Path('synth_grid')
        assert self.indir.exists(), 'synth_grid does not exist'

        self.kwargs = kwargs  # Ybirth, Zbirth, Z_over_X_sun, YBBN

        self.timer = Timer()
        self()

    def __call__(self) -> None:
        datacube = np.zeros((12, len(SynthGrid.OUT_MASS_LIST),
                                 len(SynthGrid.OUT_FEH_LIST)))

        for j, mass in enumerate(SynthGrid.OUT_MASS_LIST):
            for i, FeH in enumerate(SynthGrid.OUT_FEH_LIST):
                model = AnalModel(self, mass, FeH, **self.kwargs)
                datacube[:, j, i] = model.data
                del model.data, model

        with open('datacube.npy', 'wb') as f:
            np.save(f, datacube)
        # print(datacube[:, 1, 4])
        del datacube

        print(' > Grid data extracted!', '@', self.timer(), flush=True)

class AnalModel:

    QTY_LIST = ['star_age', 'he_core_mass', 'log_g',
                'surface_[C/Fe]', 'surface_[N/Fe]',
                'surface_[C/N]', 'surface_12C/13C', 'surface_A(Li)']

    def __init__(self, grid: AnalGrid, mass: float,
                 FeH: float, **kwargs) -> None:
        self.grid = grid
        Y, Z = RgbGrid.Y_Z_calc(FeH, **kwargs)
        self.model_name = f'{mass:.2f}M_Z={Z:.4f}_FeH={FeH:+.2f}'
        self.extract_data()

    @staticmethod
    def _middle_RGBB(df: pd.core.frame.DataFrame, qty: str):
        return (df[qty][Steps.pre_RGBB ] +\
                df[qty][Steps.post_RGBB]) / 2.0

    def extract_data(self):
        fpath = self.grid.indir / f'{self.model_name}.csv'
        assert fpath.exists(), f' > Error: {self.model_name} does not exist.'
        df = pd.read_csv(fpath, index_col=0)[AnalModel.QTY_LIST]
        self.data = np.zeros((12,))

        # extract FDU data
        self.data[0] = AnalModel._middle_RGBB(df, 'surface_[C/Fe]')
        self.data[1] = AnalModel._middle_RGBB(df, 'surface_[N/Fe]')
        self.data[2] = AnalModel._middle_RGBB(df, 'surface_[C/N]')
        self.data[3] = AnalModel._middle_RGBB(df, 'surface_12C/13C')
        # print(self.model_name, df['surface_[C/N]'][Steps.post_FDU] -\
        #                        df['surface_[C/N]'][Steps.end])

        # extract RGBB data
        self.data[4] = df['log_g'][Steps.pre_RGBB ]
        self.data[5] = df['log_g'][Steps.post_RGBB]
        self.data[6] = AnalModel._middle_RGBB(df, 'log_g')

        # extract isochrone data
        f = interp1d(df['log_g'   ][Steps.pre_RGBB:Steps.pre_FDU-1:-1],
                     df['star_age'][Steps.pre_RGBB:Steps.pre_FDU-1:-1], kind='slinear')
        self.data[7] = f(3.0); del f
        self.data[8] = df['star_age'][Steps.end]
        self.data[9] = AnalModel._middle_RGBB(df, 'star_age')

        # extract Li depletion data
        self.data[10] = AnalModel._middle_RGBB(df, 'surface_A(Li)') - df['surface_A(Li)'][Steps.TAMS]
        # self.data[10] = df['surface_A(Li)'][Steps.pre_FDU] - df['surface_A(Li)'][Steps.TAMS]
        self.data[11] = df['surface_A(Li)'][Steps.TAMS] - df['surface_A(Li)'][Steps.start]

        del df
