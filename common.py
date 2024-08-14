from time import perf_counter
import pathlib
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
                 'font.size': 12, 'text.latex.preamble': r"\usepackage{amsmath}",
                 'xtick.major.pad': 2, 'ytick.major.pad': 2, 'xtick.major.size': 6, 'ytick.major.size': 6,
                 'xtick.minor.size': 3, 'ytick.minor.size': 3, 'axes.linewidth': 2, 'axes.labelpad': 1})
from pypdf import PdfMerger


class Timer:
    '''
    All-purpose timer.

    '''

    def __init__(self):
        self.tstart = perf_counter()

    def __call__(self, reset: bool = False):
        tnow = perf_counter()
        tstart = self.tstart
        if reset:
            self.tstart = tnow
        return tnow - tstart


class Tools:
    '''
    Collection of common tools.

    '''

    @staticmethod
    def format_axis(ax: mpl.axes._axes.Axes, grid_on: bool = True) -> None:
        ax.minorticks_on()
        if grid_on: ax.grid(visible=True, which='major', linestyle=':')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.patch.set_alpha(0.0)

    @staticmethod
    def save_figure(fig: mpl.figure.Figure, fig_name: str) -> None:
        fig.tight_layout()
        fig.savefig(f'{fig_name}.pdf', dpi=128)
        plt.close(fig)

    @staticmethod
    def merge_plots(output_dir: pathlib.Path, filename: str, plots: [str]) -> None:
        pdf_merger = PdfMerger()
        for num, plot in enumerate(plots):
            pdf_merger.append(f'{plot}.pdf')
            pdf_merger.add_outline_item(title=f'{plot}', pagenum=num)

        with (output_dir / f'{filename}.pdf').open(mode='wb') as file:
            pdf_merger.write(file)
        pdf_merger.close()

        for plot in plots:
            pathlib.Path(f'{plot}.pdf').unlink()

    @staticmethod
    def generate_rgb_threads(grid_name: str, num_metal_line: int = 40,
                             num_metal_nodes: int = 7) -> None:
        '''
        This method is supposed to be used in the project directory
        (the directory containing all grid directories).

        '''

        grid_dir = pathlib.Path(grid_name)
        assert grid_dir, 'grid directory does not exist'

        for i in range(num_metal_nodes):
            thread_dir = pathlib.Path(f'{grid_name}_{i}')
            if thread_dir.exists(): continue
            thread_dir.mkdir()

            shutil.copytree(grid_dir / 'prototype', thread_dir / 'prototype')
            for suffix in ['.py', '.sh']:
                shutil.copy(grid_dir / f'{grid_name}{suffix}',
                            thread_dir / f'{grid_name}{suffix}')

            with open(thread_dir / f'{grid_name}.py', 'r') as f: my_py = f.readlines()
            my_py[num_metal_line-1] = my_py[num_metal_line-1].replace(
                'RgbGrid.FEH_LIST', f'RgbGrid.FEH_LIST[{i}:{i+1}]')
            with open(thread_dir / f'{grid_name}.py', 'w') as f: f.writelines(my_py)
