import os
import pathlib
import shutil

from .common import Timer


class SimGrid:
    '''Base class of SunGrid and RgbGrid.
    '''

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
            # os.system('$MESA_DIR/empty_caches')

        for item in ['.mesa_temp_cache', 'photos', 'inlist', 'testhub.yml']:
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
    '''Controller of a single simulation.
    '''

    def __init__(self, grid: SimGrid, **kwargs):
        self.grid = grid

        self.aMLT = kwargs['aMLT']
        self.mass = kwargs['mass']
        self.FeH  = kwargs['FeH']
        self.Y = kwargs['Y']
        self.Z = kwargs['Z']

        if not grid.rgb_mode:
            self.model_name = f'round={grid.round_}_Z={self.Z:.4f}_Y={self.Y:.4f}_aMLT={self.aMLT:.4f}'
        else:
            self.model_name = f'aMLT={self.aMLT:.4f}_{self.mass:.2f}M_Z={self.Z:.4f}_FeH={self.FeH:+.2f}'
        self()

    def __call__(self):
        self.modify_inlists()
        self.evolve_model()

    @staticmethod
    def frac_wrapper(frac: float):
        return f'{frac:.8e}'.replace('e', 'd').replace('-0', '-')

    def modify_inlists(self):
        X = 1.0 - self.Y - self.Z

        for inlist in ['inlist_start', 'inlist_to_solar_age', 'inlist_to_past_rgb_bump']:
            inlist_file = self.grid.output_dir / inlist
            if not inlist_file.exists(): continue
            with open(inlist_file, 'r') as f: my_inlist = f.readlines()

            for i, line in enumerate(my_inlist):
                if inlist == 'inlist_start':
                    if 'initial_h1 = ' in line:
                        my_inlist[i] = f'      initial_h1 = {SimCtr.frac_wrapper(X*0.9999806)}\n'
                    elif 'initial_h2 = ' in line:
                        my_inlist[i] = f'      initial_h2 = {SimCtr.frac_wrapper(X*0.0000194)}\n'
                    elif 'initial_he3 = ' in line:
                        my_inlist[i] = f'      initial_he3 = {SimCtr.frac_wrapper(self.Y*0.00016597)}\n'
                    elif 'initial_he4 = ' in line:
                        my_inlist[i] = f'      initial_he4 = {SimCtr.frac_wrapper(self.Y*0.99983403)}\n'
                    elif 'xa_central_lower_limit(1) = ' in line:
                        my_inlist[i] = f'      xa_central_lower_limit(1) = {SimCtr.frac_wrapper(X*0.000000194)}\n'

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
    '''A grid of solar simulations.
    '''

    rgb_mode = False

    Y_LIST = [0.2590, 0.2690, 0.2790]
    AMLT_LIST = [1.5, 1.8, 2.1]

    def __init__(self, sim_mode: bool = False, round_: int = 0, Z: float = 0.0187,
                 Y_list: [float] = None, aMLT_list: [float] = None):
        output_dir = pathlib.Path('sun_grid')
        output_dir.mkdir(exist_ok=True)
        super(SunGrid, self).__init__(output_dir, sim_mode)

        self.round_ = round_
        self.Z = Z
        self.Y_list = Y_list if Y_list is not None else SunGrid.Y_LIST
        self.aMLT_list = aMLT_list if aMLT_list is not None else SunGrid.AMLT_LIST

        self.timer = Timer()
        self()

    def __call__(self):
        if not self.sim_mode:
            os.chdir(self.output_dir.name)
            os.system('./mk')
            os.chdir('..')

        for aMLT in self.aMLT_list:
            for Y in self.Y_list:
                SimCtr(self, aMLT=aMLT, mass=None, FeH=None, Y=Y, Z=self.Z)

        self.remove_mesa()
        print(' > Finished evolving all models!', '@', self.timer(), flush=True)


class RgbGrid(SimGrid):
    '''A grid of RGB simulations.
    '''

    rgb_mode = True

    AMLT_LIST = [1.5, 1.8, 2.1]
    AMLT_MINI = [1.6, 2.0]
    MASS_LIST = [0.85, 1.00, 1.15, 1.30, 1.45, 1.65, 1.85]
    MASS_MINI = [1.00, 1.30, 1.65]
    FEH_LIST = [-0.45, -0.30, -0.15,  0.00,  0.15,  0.30,  0.45]
    FEH_MINI = [-0.45,  0.00,  0.45]

    def __init__(self, sim_mode: bool = False, aMLT_list: [float] = None,
                 mass_list: [float] = None, FeH_list: [float] = None, **kwargs):
        output_dir = pathlib.Path('rgb_grid')
        output_dir.mkdir(exist_ok=True)
        super(RgbGrid, self).__init__(output_dir, sim_mode)

        self.aMLT_list = aMLT_list if aMLT_list is not None else RgbGrid.AMLT_LIST
        self.mass_list = mass_list if mass_list is not None else RgbGrid.MASS_LIST
        self.FeH_list  = FeH_list  if FeH_list  is not None else RgbGrid.FEH_LIST
        self.kwargs = kwargs  # Ybirth, Zbirth, Z_over_X_sun, YBBN

        self.timer = Timer()
        self()

    @staticmethod
    def Y_Z_calc(FeH: float, Ybirth: float = 0.2690, Zbirth: float = 0.0187,
                 Z_over_X_sun: float = 0.02292, YBBN: float = 0.24709):
        Z_over_X = Z_over_X_sun * 10.0 ** FeH
        DY_over_Zbirth = (Ybirth - YBBN) / Zbirth
        Z = (1.0 - YBBN) / (1.0 + 1.0 / Z_over_X + DY_over_Zbirth)
        Y = YBBN + DY_over_Zbirth * Z
        return Y, Z

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

