import os
import pathlib
import shutil

from importlib.resources import files
from .common import Timer


class SimGrid:
    '''
    Base class of SunGrid and RgbGrid.

    '''

    # for numerical convergence
    TIME_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
    MESH_LIST = [1.0]

    def __init__(self, output_dir: pathlib.Path, sim_mode: bool = False):
        self.output_dir = output_dir
        self.sim_mode = sim_mode
        self.import_mesa()

    def import_mesa(self):
        self.input_dir = pathlib.Path('prototype')
        assert self.input_dir.exists(), 'prototype does not exist'

        for item in self.input_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, self.output_dir / item.name)
            else:
                shutil.copy(item, self.output_dir / item.name)
        if not self.default_rates:
            shutil.copytree(files(__package__).joinpath('rate_tables'),
                            self.output_dir / 'rate_tables')

        if self.rgb_mode:
            (self.output_dir / 'inlist_to_solar_age').unlink()
            (self.output_dir / 'inlist_to_solar_age_header').unlink()
            with open(self.output_dir / 'rn', 'r') as f: my_rn = f.readlines()
            my_rn[ 9] = '# do_one inlist_to_solar_age_header solar_age.mod LOGS_to_solar_age\n'
            my_rn[10] = 'do_one inlist_to_past_rgb_bump_header past_rgb_bump.mod LOGS_to_past_rgb_bump\n'
            with open(self.output_dir / 'rn', 'w') as f: f.writelines(my_rn)

        else:
            (self.output_dir / 'inlist_to_past_rgb_bump').unlink()
            (self.output_dir / 'inlist_to_past_rgb_bump_header').unlink()
            with open(self.output_dir / 'rn', 'r') as f: my_rn = f.readlines()
            my_rn[ 9] = 'do_one inlist_to_solar_age_header solar_age.mod LOGS_to_solar_age\n'
            my_rn[10] = '# do_one inlist_to_past_rgb_bump_header past_rgb_bump.mod LOGS_to_past_rgb_bump\n'
            with open(self.output_dir / 'rn', 'w') as f: f.writelines(my_rn)

    def remove_mesa(self):
        if not self.sim_mode:
            os.chdir(self.output_dir.name)
            os.system('./clean')
            os.chdir('..')

        for item in ['.mesa_temp_cache', 'eosDT_cache', 'kap_cache', 'rates_cache',
                     'rate_tables', 'photos', 'inlist', 'testhub.yml']:
            item_ = self.output_dir / item
            if item_.exists():
                if item_.is_dir(): shutil.rmtree(item_)
                else: item_.unlink()

        for item in self.input_dir.iterdir():
            item_ = self.output_dir / item.name
            if item_.exists():
                if item_.is_dir(): shutil.rmtree(item_)
                else: item_.unlink()


class SimCtr:
    '''
    Controller of a single simulation.

    '''

    # isotopic *mass* fractions for hydrogen and helium
    H1_FRAC = 1.0 / (1.0 + 0.0000194*2.0141018 / (0.9999806*1.007825))
    H2_FRAC = 1.0 - H1_FRAC  # (0.9999612305528522, 3.8769447147841e-05)
    HE4_FRAC = 1.0 / (1.0 + 0.00016597*3.0160293 / (0.99983403*4.0026033))
    HE3_FRAC = 1.0 - HE4_FRAC  # 0.9998749336809143, 0.0001250663190856427

    def __init__(self, grid: SimGrid, num_conv: bool = False, **kwargs):
        self.grid = grid

        if not num_conv:
            self.aMLT = kwargs['aMLT']
            self.mass = kwargs['mass']
            self.FeH  = kwargs['FeH']
            self.Y = kwargs['Y']
            self.Z = kwargs['Z']

            if isinstance(grid, SunGrid):
                self.model_name = f'round={grid.round_}_Z={self.Z:.4f}_Y={self.Y:.4f}_aMLT={self.aMLT:.4f}'
            else:
                self.model_name = f'aMLT={self.aMLT:.4f}_{self.mass:.2f}M_Z={self.Z:.4f}_FeH={self.FeH:+.2f}'

        else:
            self.time_coeff = kwargs['time_coeff']
            self.mesh_coeff = kwargs['mesh_coeff']
            self.model_name = f'time_coeff={self.time_coeff:.2f}_mesh_coeff={self.mesh_coeff:.2f}'

        self(num_conv)

    def __call__(self, num_conv: bool = False):
        self.modify_inlists(num_conv)
        self.evolve_model()

    @staticmethod
    def frac_wrapper(frac: float):
        return f'{frac:.8e}'.replace('e', 'd').replace('-0', '-')

    def modify_inlists(self, num_conv: bool = False):
        for inlist in ['inlist_start', 'inlist_to_solar_age', 'inlist_to_past_rgb_bump']:
            inlist_file = self.grid.output_dir / inlist
            if not inlist_file.exists(): continue
            with open(inlist_file, 'r') as f: my_inlist = f.readlines()

            if not num_conv:
                X = 1.0 - self.Y - self.Z
                for i, line in enumerate(my_inlist):
                    if inlist == 'inlist_start':
                        if 'initial_h1 = ' in line:
                            my_inlist[i] = f'      initial_h1 = {SimCtr.frac_wrapper(X*SimCtr.H1_FRAC)}\n'
                        elif 'initial_h2 = ' in line:
                            my_inlist[i] = f'      initial_h2 = {SimCtr.frac_wrapper(X*SimCtr.H2_FRAC)}\n'
                        elif 'initial_he3 = ' in line:
                            my_inlist[i] = f'      initial_he3 = {SimCtr.frac_wrapper(self.Y*SimCtr.HE3_FRAC)}\n'
                        elif 'initial_he4 = ' in line:
                            my_inlist[i] = f'      initial_he4 = {SimCtr.frac_wrapper(self.Y*SimCtr.HE4_FRAC)}\n'
                        elif 'xa_central_lower_limit(1) = ' in line:
                            my_inlist[i] = f'      xa_central_lower_limit(1) = {SimCtr.frac_wrapper(X*SimCtr.H2_FRAC*0.01)}\n'

                    if 'Zbase = ' in line:
                        my_inlist[i] = f'      Zbase = {self.Z:.10f}d0\n'
                    elif 'initial_mass = ' in line:
                        if self.mass is not None:
                            my_inlist[i] = f'      initial_mass = {self.mass:.2f}d0\n'
                    elif 'initial_z = ' in line:
                        my_inlist[i] = f'      initial_z = {self.Z:.10f}d0\n'
                    elif 'initial_y = ' in line:
                        my_inlist[i] = f'      initial_y = {self.Y:.10f}d0\n'
                    elif 'mixing_length_alpha = ' in line:
                        my_inlist[i] = f'      mixing_length_alpha = {self.aMLT:.10f}d0\n'

                    if isinstance(self.grid, LiGrid):
                        if 'max_age = ' in line:
                            my_inlist[i] = 'max_age = 1.25d6'
                        elif 'varcontrol_target = ' in line:
                            my_inlist[i] = '      varcontrol_target = 1d-4\n'

            else:
                for i, line in enumerate(my_inlist):
                    if 'time_delta_coeff = ' in line:
                        my_inlist[i] = f'      time_delta_coeff = {self.time_coeff:.2f}d0\n'
                    elif 'mesh_delta_coeff = ' in line:
                        my_inlist[i] = f'      mesh_delta_coeff = {self.mesh_coeff:.2f}d0\n'

                    if not self.grid.rgb_mode:
                        if 'varcontrol_target = ' in line:
                            my_inlist[i] = '      varcontrol_target = 1d-4\n'

            with open(inlist_file, 'w') as f: f.writelines(my_inlist)

    def evolve_model(self):
        model_dir = self.grid.output_dir / self.model_name
        if model_dir.exists(): return
        model_dir.mkdir()

        print(' > Evolving model:', self.model_name, '@', self.grid.timer(), flush=True)
        if not self.grid.sim_mode:
            os.chdir(self.grid.output_dir.name)
            os.system('./rn > terminal.out')
            os.chdir('..')

        for inlist in ['inlist_start', 'inlist_to_solar_age', 'inlist_to_past_rgb_bump']:
            inlist_file = self.grid.output_dir / inlist
            if inlist_file.exists():
                shutil.copy(inlist_file, model_dir / inlist)

        for log in ['_start', '_to_solar_age', '_to_past_rgb_bump', '']:
            log_dir = self.grid.output_dir / f'LOGS{log}'
            if log_dir.exists():
                (log_dir / 'history.data').rename(model_dir / f'history{log}.data')
                shutil.rmtree(log_dir)

        for mod in ['start.mod', 'solar_age.mod', 'past_rgb_bump.mod']:
            mod_file = self.grid.output_dir / mod
            if mod_file.exists():
                mod_file.rename(model_dir / mod)

        if not self.grid.sim_mode:
            out_file = self.grid.output_dir / 'terminal.out'
            out_file.rename(model_dir / 'terminal.out')


class SunGrid(SimGrid):
    '''
    A grid of solar simulations.

    '''

    rgb_mode = False

    Y_LIST = [0.2590, 0.2690, 0.2790]
    AMLT_LIST = [1.5, 1.8, 2.1]

    def __init__(self, sim_mode: bool = False, round_: int = 0, Z: float = 0.0187,
                 Y_list: [float] = None, aMLT_list: [float] = None,
                 num_conv: bool = False, time_list: [float] = None,
                 mesh_list: [float] = None, default_rates: bool = False):
        output_dir = pathlib.Path('sun_grid' if not num_conv else 'sun_conv')
        output_dir.mkdir(exist_ok=True)
        self.default_rates = default_rates
        super(SunGrid, self).__init__(output_dir, sim_mode)

        if not num_conv:
            self.round_ = round_
            self.Z = Z
            self.Y_list    = Y_list    if Y_list    is not None else SunGrid.Y_LIST
            self.aMLT_list = aMLT_list if aMLT_list is not None else SunGrid.AMLT_LIST
        else:
            self.time_list = time_list if time_list is not None else SimGrid.TIME_LIST
            self.mesh_list = mesh_list if mesh_list is not None else SimGrid.MESH_LIST

        self.timer = Timer()
        self(num_conv)

    def __call__(self, num_conv: bool = False):
        if not self.sim_mode:
            os.chdir(self.output_dir.name)
            os.system('./mk')
            os.chdir('..')

        if not num_conv:
            for aMLT in self.aMLT_list:
                for Y in self.Y_list:
                    SimCtr(self, aMLT=aMLT, mass=None, FeH=None, Y=Y, Z=self.Z)
        else:
            for time_coeff in self.time_list:
                for mesh_coeff in self.mesh_list:
                    SimCtr(self, num_conv=True, time_coeff=time_coeff, mesh_coeff=mesh_coeff)

        self.remove_mesa()
        print(' > Finished evolving all models!', '@', self.timer(), flush=True)


class RgbGrid(SimGrid):
    '''
    A grid of RGB simulations.

    '''

    rgb_mode = True

    AMLT_LIST = [1.5, 1.8, 2.1]
    AMLT_MINI = [1.6, 2.0]
    MASS_LIST = [0.90, 1.00, 1.15, 1.30, 1.45, 1.65, 1.85]
    MASS_MINI = [1.00, 1.30, 1.65]
    FEH_LIST = [-0.48, -0.32, -0.16,  0.00,  0.15,  0.30,  0.45]
    FEH_MINI = [-0.48,  0.00,  0.45]

    def __init__(self, sim_mode: bool = False, aMLT_list: [float] = None,
                 mass_list: [float] = None, FeH_list: [float] = None,
                 num_conv: bool = False, time_list: [float] = None,
                 mesh_list: [float] = None, **kwargs):
        output_dir = pathlib.Path('rgb_grid' if not num_conv else 'rgb_conv')
        output_dir.mkdir(exist_ok=True)
        self.default_rates = kwargs.pop('default_rates', False)
        super(RgbGrid, self).__init__(output_dir, sim_mode)

        if not num_conv:
            self.aMLT_list = aMLT_list if aMLT_list is not None else RgbGrid.AMLT_LIST
            self.mass_list = mass_list if mass_list is not None else RgbGrid.MASS_LIST
            self.FeH_list  = FeH_list  if FeH_list  is not None else RgbGrid.FEH_LIST
            self.kwargs = kwargs  # Ybirth, Zbirth, Z_over_X_sun, YBBN
        else:
            self.time_list = time_list if time_list is not None else SimGrid.TIME_LIST
            self.mesh_list = mesh_list if mesh_list is not None else SimGrid.MESH_LIST

        self.timer = Timer()
        self(num_conv)

    @staticmethod
    def Y_Z_calc(FeH: float, Ybirth: float = 0.2690, Zbirth: float = 0.0187,
                 Z_over_X_sun: float = 0.02292, YBBN: float = 0.24709):
        Z_over_X = Z_over_X_sun * 10.0 ** FeH
        DY_over_Zbirth = (Ybirth - YBBN) / Zbirth
        Z = (1.0 - YBBN) / (1.0 + 1.0 / Z_over_X + DY_over_Zbirth)
        Y = YBBN + DY_over_Zbirth * Z
        return Y, Z

    def __call__(self, num_conv: bool = False):
        if not self.sim_mode:
            os.chdir(self.output_dir.name)
            os.system('./mk')
            os.chdir('..')

        if not num_conv:
            for aMLT in self.aMLT_list:
                for mass in self.mass_list:
                    for FeH in self.FeH_list:
                        Y, Z = RgbGrid.Y_Z_calc(FeH, **self.kwargs)
                        SimCtr(self, aMLT=aMLT, mass=mass, FeH=FeH, Y=Y, Z=Z)
        else:
            for time_coeff in self.time_list:
                for mesh_coeff in self.mesh_list:
                    SimCtr(self, num_conv=True, time_coeff=time_coeff, mesh_coeff=mesh_coeff)

        self.remove_mesa()
        print(' > All models completed!', '@', self.timer(), flush=True)


class LiGrid(SimGrid):
    '''
    A grid of PMS lithium depletion simulations.

    '''

    rgb_mode = False

    AMLT_LIST = [1.8]
    MASS_LIST = [0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3]
    FEH_LIST = [-0.1,  0. ,  0.1]

    def __init__(self, sim_mode: bool = False, aMLT_list: [float] = None,
                 mass_list: [float] = None, FeH_list: [float] = None, **kwargs):
        output_dir = pathlib.Path('li_grid')
        output_dir.mkdir(exist_ok=True)
        self.default_rates = kwargs.pop('default_rates', False)
        super(LiGrid, self).__init__(output_dir, sim_mode)

        self.aMLT_list = aMLT_list if aMLT_list is not None else LiGrid.AMLT_LIST
        self.mass_list = mass_list if mass_list is not None else LiGrid.MASS_LIST
        self.FeH_list  = FeH_list  if FeH_list  is not None else LiGrid.FEH_LIST
        self.kwargs = kwargs  # Ybirth, Zbirth, Z_over_X_sun, YBBN

        self.timer = Timer()
        self()

    def __call__(self):
        if not self.sim_mode:
            os.chdir(self.output_dir.name)
            os.system('./mk')
            os.chdir('..')

        for aMLT in self.aMLT_list:
            for mass in self.mass_list:
                for FeH in self.FeH_list:
                    Y, Z = RgbGrid.Y_Z_calc(FeH, **self.kwargs)
                    SimCtr(self, aMLT=aMLT, mass=mass, FeH=FeH, Y=Y, Z=Z)

        self.remove_mesa()
        print(' > All models completed!', '@', self.timer(), flush=True)
