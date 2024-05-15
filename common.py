from time import perf_counter
import pathlib

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
    def format_axis(ax: mpl.axes._axes.Axes) -> None:
        ax.minorticks_on(); ax.grid(visible=True, which='major', linestyle=':')
        ax.tick_params(axis='both', which='both', direction='out')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.patch.set_alpha(0.0)

    @staticmethod
    def save_figure(fig: mpl.figure.Figure, fig_name: str) -> None:
        fig.tight_layout()
        fig.savefig(f'{fig_name}.pdf', dpi=128)
        plt.close(fig)

    @staticmethod
    def merge_plots(output_dir: pathlib.Path, filename: str, plots: [str]):
        pdf_merger = PdfMerger()
        for num, plot in enumerate(plots):
            pdf_merger.append(f'{plot}.pdf')
            pdf_merger.add_outline_item(title=f'{plot}', pagenum=num)

        with (output_dir / f'{filename}.pdf').open(mode='wb') as file:
            pdf_merger.write(file)
        pdf_merger.close()

        for plot in plots:
            pathlib.Path(f'{plot}.pdf').unlink()
