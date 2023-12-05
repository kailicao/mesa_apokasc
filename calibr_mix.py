import pathlib
import json

import numpy as np
from scipy.interpolate import RectBivariateSpline

import mesa_reader as mr
from .common import Timer
from .sim_ctr import SunGrid


class MixCalibr:
    '''
    Calibrated solar mixture.

    '''

    ISOTOPES = ['h1', 'h2', 'he3', 'he4', 'li7',
                'c12', 'c13', 'n14', 'n15', 'o16', 'o17', 'o18',
                'f19', 'ne20', 'mg22', 'mg24']

    def __init__(self, Z_over_X_sun: float = 0.02292, round_: int = 0, Z: float = 0.0187,
                 Y_list: [float] = SunGrid.Y_LIST, aMLT_list: [float] = SunGrid.AMLT_LIST,
                 Y_opt: float = SunGrid.Y_LIST[1], aMLT_opt: float = SunGrid.AMLT_LIST[1]):
        self.output_dir = pathlib.Path('sun_grid')
        assert self.output_dir.exists(), 'sun_grid does not exist'

        self.Z_over_X_sun = Z_over_X_sun
        self.round_ = round_
        self.Z = Z
        self.Y_list = Y_list
        self.aMLT_list = aMLT_list
        self.Y_opt = Y_opt
        self.aMLT_opt = aMLT_opt

        self.timer = Timer()
        self()

    def __call__(self):
        self.read_models()
        self.perform_interps()
        self.clear_models()

        self.export_stdmix()
        print(' > Standard solar mixture established!', '@', self.timer())

    def read_models(self):
        self.models = {}

        for Y in self.Y_list:
            for aMLT in self.aMLT_list:
                self.models[(Y, aMLT)] = MixModel(self, aMLT, Y)

    def perform_interps(self):
        self.stdmix = {}
        aiso_arr = np.zeros((len(self.Y_list), len(self.aMLT_list)))  # a stands for abundance

        for iso in MixCalibr.ISOTOPES:
            for j, Y in enumerate(self.Y_list):
                for i, aMLT in enumerate(self.aMLT_list):
                    aiso_arr[j, i] = self.models[(Y, aMLT)].mix[iso]

            interp = RectBivariateSpline(self.Y_list, self.aMLT_list, aiso_arr,
                kx=min(len(self.Y_list)-1, 3), ky=min(len(self.aMLT_list)-1, 3))
            self.stdmix[iso] = interp(self.Y_opt, self.aMLT_opt)[0, 0]
            del interp

        del aiso_arr

    def clear_models(self):
        for model in self.models.values():
            model.clear_data()
        self.models.clear()
        del self.models

    def export_stdmix(self):
        # normalize the interpolation results
        total = sum(self.stdmix.values())
        for iso in MixCalibr.ISOTOPES:
            self.stdmix[iso] /= total
        print(' > Before normalization,', f'{total = }')

        self.stdmix['X_frac'] = self.stdmix['h1']  + self.stdmix['h2']
        self.stdmix['Y_frac'] = self.stdmix['he3'] + self.stdmix['he4']
        self.stdmix['Z_frac'] = 1.0 - self.stdmix['X_frac'] - self.stdmix['Y_frac']
        # self.stdmix['Z_frac'] = sum(self.stdmix[iso] for iso in MixCalibr.ISOTOPES[4:])
        self.stdmix['Z_over_X'] = self.stdmix['Z_frac'] / self.stdmix['X_frac']
        print(' > Z_over_X_sun =', self.stdmix['Z_over_X'])

        with open('stdmix.json', 'w') as f:
            json.dump(self.stdmix, f, indent=4)

        self.stdmix.clear()
        del self.stdmix

class MixModel:
    '''
    Data structure for the mixture of a solar model.

    '''

    def __init__(self, calibr: MixCalibr, aMLT: float, Y: float):
        self.calibr = calibr
        self.aMLT = aMLT
        self.Y = Y

        self.extract_data()

    def extract_data(self):
        model_name = f'round={self.calibr.round_}_Z={self.calibr.Z:.4f}_Y={self.Y:.4f}_aMLT={self.aMLT:.4f}'
        histfile = self.calibr.output_dir / model_name / 'history_to_solar_age.data'
        assert histfile.exists(), 'input file does not exist'

        h = mr.MesaData(str(histfile))
        self.mix = {}
        for iso in MixCalibr.ISOTOPES:
            self.mix[iso] = h.data(f'surface_{iso}')[-1]
        del h

    def clear_data(self):
        self.mix.clear()
        del self.mix
