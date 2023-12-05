import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline

# from mesa_apokasc.sim_ctr import RgbGrid
from mesa_apokasc.calibr_rgb import RgbCalibr


class AmltModel:
    '''
    Fitting models for the mixing length parameter.

    '''

    IN_QTY_LIST = ['KIC', 'TEFF', 'S_TEFF', 'LOGG_C_MO', 'S_LOGG_C_MO',
                   'M_C_MO', 'S_M_C_MO']  # metallicity columns to be specified
    OUT_QTY_LIST = ['AMLT_FIT', 'DAMLT_DM', 'DAMLT_D[M/H]', 'TEFF_FIT',
                    'DTEFF_DLOGG', 'DTEFF_DM', 'DTEFF_D[M/H]', 'DTEFF_DAMLT',
                    'S_TEFF_LOGG', 'S_TEFF_M', 'S_TEFF_[M/H]', 'S_TEFF_TOT']
    DELTA = 1e-4  # for computing numerical derivatives

    def __init__(self, order_mass: int = 2, order_metal: int = 1,
                 cross_term: bool = False, solar_value: float = None):
        self.order_mass = order_mass
        self.order_metal = order_metal
        self.cross_term = cross_term
        self.solar_value = solar_value

        self.model_name = f'ms{order_mass}_mt{order_metal}'
        if cross_term: self.model_name += '_cr'
        self.num_params = 1 + order_mass + order_metal + cross_term

        if solar_value is not None:
            self.order_mass = 0
            self.order_metal = 0
            self.cross_term = False
            self.model_name = 'solar'
            self.num_params = 0

        self.build_funcs()

    def build_funcs(self):
        func_stem = 'lambda x, a'
        for i in range(1, self.order_mass+1):
            func_stem += f', b{i}'
        for j in range(1, self.order_metal+1):
            func_stem += f', c{j}'
        if self.cross_term:
            func_stem += ', d'
        func_stem += ': '

        func_aMLT = func_stem + 'a'
        for i in range(1, self.order_mass+1):
            func_aMLT += f' + b{i}*x[0]**{i}'.rstrip('**1')
        for j in range(1, self.order_metal+1):
            func_aMLT += f' + c{j}*x[1]**{j}'.rstrip('**1')
        if self.cross_term:
            func_aMLT += ' + d*x[0]*x[1]'

        if self.solar_value is not None:
            func_aMLT = func_stem + repr(self.solar_value)
        self.func_aMLT = eval(func_aMLT)

        daMLT_dmass = func_stem + '0'
        for i in range(1, self.order_mass+1):
            daMLT_dmass += ' + b1' if i == 1 else f' + {i}*b{i}*x[0]**{i-1}'.rstrip('**1')
        if self.cross_term:
            daMLT_dmass += ' + d*x[1]'
        self.daMLT_dmass = eval(daMLT_dmass)

        daMLT_dmetal = func_stem + '0'
        for i in range(1, self.order_metal+1):
            daMLT_dmetal += ' + c1' if i == 1 else f' + {i}*c{i}*x[1]**{i-1}'.rstrip('**1')
        if self.cross_term:
            daMLT_dmetal += ' + d*x[0]'
        self.daMLT_dmetal = eval(daMLT_dmetal)

    @staticmethod
    def get_Teff_pred(calibr, mass: float, metal: float, logg: float, amlt: float):
        Teff_pred = np.zeros(len(calibr.aMLT_list))
        for k in range(len(calibr.aMLT_list)):
            Teff_arr = [interp(mass, metal)[0, 0] for interp in calibr.Teff_interps[k][::-1]]
            logg_arr = [interp(mass, metal)[0, 0] for interp in calibr.logg_interps[k][::-1]]
            Teff_pred[k] = interp1d(logg_arr, Teff_arr, kind='slinear',
                                    fill_value='extrapolate')(logg)
            Teff_arr.clear(); logg_arr.clear(); del Teff_arr, logg_arr

        return UnivariateSpline(calibr.aMLT_list, Teff_pred, k=1)(amlt)

    def analyze_error(self, calibr, vary: str = 'both', Salaris: bool = False):
        catalog = calibr.apokasc3.copy()
        METAL = '[FE/H]' if not Salaris else '[M/H]_SAL'
        catalog = catalog[AmltModel.IN_QTY_LIST + [METAL, f'S_{METAL}', 'AMLT_OPT']]
        for QTY in AmltModel.OUT_QTY_LIST: catalog[QTY] = np.nan

        # fit to the AMLT_OPT distribution
        if self.solar_value is not None:
            catalog['AMLT_FIT'] = self.solar_value
            catalog['DAMLT_DM'] = 0.0
            catalog['DAMLT_D[M/H]'] = 0.0
        else:
            model_input = [catalog['M_C_MO']-1, catalog[METAL]]
            popt, pcov = curve_fit(self.func_aMLT, model_input, catalog['AMLT_OPT'])
            catalog['AMLT_FIT'] = self.func_aMLT(model_input, *popt)
            catalog['DAMLT_DM'] = self.daMLT_dmass(model_input, *popt)
            catalog['DAMLT_D[M/H]'] = self.daMLT_dmetal(model_input, *popt)
            del model_input
            # print(np.sum(np.square(catalog['AMLT_FIT'] - catalog['AMLT_OPT'])))

        # loop over stars, compute TEFF_FIT and numerical derivatives
        delta = AmltModel.DELTA; get_Teff_pred = AmltModel.get_Teff_pred

        for i, star in catalog.iterrows():
            mass  = star['M_C_MO']
            metal = star[METAL]
            logg  = star['LOGG_C_MO']
            amlt  = star['AMLT_FIT']

            catalog.at[i, 'TEFF_FIT'] =\
                get_Teff_pred(calibr, mass, metal, logg, amlt)
            catalog.at[i, 'DTEFF_DLOGG'] =\
                (get_Teff_pred(calibr, mass, metal, logg+delta, amlt) -\
                 get_Teff_pred(calibr, mass, metal, logg-delta, amlt)) / (delta*2)
            catalog.at[i, 'DTEFF_DM'] =\
                (get_Teff_pred(calibr, mass+delta, metal, logg, amlt) -\
                 get_Teff_pred(calibr, mass-delta, metal, logg, amlt)) / (delta*2)
            catalog.at[i, 'DTEFF_D[M/H]'] =\
                (get_Teff_pred(calibr, mass, metal+delta, logg, amlt) -\
                 get_Teff_pred(calibr, mass, metal-delta, logg, amlt)) / (delta*2)
            catalog.at[i, 'DTEFF_DAMLT'] =\
                (get_Teff_pred(calibr, mass, metal, logg, amlt+delta) -\
                 get_Teff_pred(calibr, mass, metal, logg, amlt-delta)) / (delta*2)

        # compute contributions to S_TEFF_TOT and itself
        catalog['S_TEFF_LOGG']  = catalog['S_LOGG_C_MO'] * catalog['DTEFF_DLOGG']
        catalog['S_TEFF_M']     = catalog['S_M_C_MO'] * (catalog['DTEFF_DM'] \
                                + catalog['DTEFF_DAMLT'] * catalog['DAMLT_DM'])
        catalog['S_TEFF_[M/H]'] = catalog[f'S_{METAL}'] * (catalog['DTEFF_D[M/H]'] \
                                + catalog['DTEFF_DAMLT'] * catalog['DAMLT_D[M/H]'])
        catalog['S_TEFF_TOT']   = np.sqrt(sum([np.square(catalog[s]) for s in \
                                  ['S_TEFF', 'S_TEFF_LOGG', 'S_TEFF_M', 'S_TEFF_[M/H]']]))

        fname = 'Salaris-' + ('on' if Salaris else 'off') +\
            '_vary-' + vary + '_' + self.model_name + '.csv'
        catalog.to_csv(calibr.outdir / fname)


class RgbCalErr(RgbCalibr):
    '''
    Error analysis for RGB mixing length calibration.

    '''

    def __call__(self, vary: str = 'both', Salaris: bool = False):
        self.extract_sim_data(vary)
        self.build_interps()

        # load RgbCalibr.get_amlt_opt results
        fname = 'Salaris-' + ('on' if Salaris else 'off') + '_vary-' + vary + '.csv'
        self.apokasc3 = pd.read_csv(self.outdir / fname, index_col=0)

        self.model_controller(vary, Salaris)
        self.clear()
        print(' > Finished analyzing all models!', '@', self.timer(), flush=True)

    def model_controller(self, vary: str = 'both', Salaris: bool = False):
        # for order_mass in range(5):
        #     for order_metal in range(4):
        #         for cross_term in [False, True]:
        #             if (order_mass == 0 or order_metal == 0) and cross_term: continue

        for order_mass in range(2, 3):
            for order_metal in range(1, 2):
                for cross_term in [False]:
                    model = AmltModel(order_mass, order_metal, cross_term)
                    print(f' > Analyzing AmltModel: {model.model_name}',
                          '@', self.timer(), flush=True)
                    model.analyze_error(self, vary, Salaris)
                    del model
