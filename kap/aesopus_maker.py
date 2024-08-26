import pathlib
import urllib
import json

import h5py
import numpy as np


def display_grid_info(h5name='AESOPUS_AGSS09'):
    with h5py.File(f'{h5name}.h5', 'r') as f:
        for key in ['Zsun', 'fCO_ref', 'fC_ref', 'fN_ref']:
            print(key, np.array(f[key], dtype=float))
        print('')

        print('Zs', [round(Z, 5) for Z in f['Zs']])
        print('shape of opacity tables:',
              np.array(f['0.000010']['kap'].shape))

        for key in ['logTs', 'logRs']:
            print(key, [round(Z, 3) for Z in f[key]])
        print('')

        for Z in f['Zs']:
            print(f'Z = {Z:.6f}')
            for key in ['Xs', 'fCOs', 'fCs', 'fNs']:
                print(key, [round(Z, 3) for Z in f[f'{Z:.6f}'][key]])
            print('')


def extract_grid_info(h5name='AESOPUS_AGSS09'):
    if pathlib.Path('grid_info.json').exists():
        with open('grid_info.json') as f:
            grid_info = json.load(f)
        return grid_info

    grid_info = {}

    with h5py.File(f'{h5name}.h5', 'r') as f:
        Zs = [float(v) for v in f['Zs']]
        grid_info['Zs'] = Zs

        for Z in Zs:
            Z_key = f'{Z:.6f}'
            grid_info[Z_key] = {}

            for f_key in ['Xs', 'fCOs', 'fCs', 'fNs']:
                grid_info[Z_key][f_key] = [float(v) for v in f[Z_key][f_key]]

    return grid_info


def CFe_NFe_calc(FeH):
    CFe = 0.268 * FeH**2 + 0.0258 * FeH - 0.00983
    NFe = 0.373 * FeH**2 + 0.373  * FeH + 0.0260
    return CFe, NFe


SOLMIX_DICT = {'GS98': 3, 'AGSS09': 7, 'AAG21': 9, 'MBS22': 10}
AESOPUS_CGI = 'http://stev.oapd.inaf.it/cgi-bin/aesopus_1.0'

def build_full_urls(grid_info=None, solmix='GS98', AFe=None, CNFe=None):
    h5name = f'AESOPUS_{solmix}'
    if AFe is not None:
        h5name += f'_A{AFe:+.1f}'
    if CNFe is not None:
        h5name += f'_CN{CNFe:+.2f}'
        CFe, NFe = CFe_NFe_calc(CNFe)
    if pathlib.Path(f'{h5name}.txt').exists(): return

    print(' > Building full urls for', h5name)
    if grid_info is None: grid_info = extract_grid_info()

    with open(f'{h5name}.txt', 'w') as f:
        for Z in grid_info['Zs']:
            Z_key = f'{Z:.6f}'
            f.write(f'aesopus_z{Z:.6f}'.rstrip('0') +
                    f'_{h5name[8:].lower()}_varcno.tab' + '\n')

            data = {'solmix': SOLMIX_DICT[solmix],
                    'zeta_ref': Z_key.rstrip('0'),
                    'xhmin': 0.4, 'xhmax': 0.8, 'dxh': 0.1}

            if AFe is not None:
                for elem in ['O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca']:
                    data[f'afe_{elem}'] = AFe
            if CNFe is not None:
                data['afe_C'] = CFe
                data['afe_N'] = NFe

            for i, fCO in enumerate(grid_info[Z_key]['fCOs']):
                data[f'fco{i+1:d}'] = round(fCO, 3)

            for i, fC in enumerate(grid_info[Z_key]['fCs']):
                data[f'fc{i+1:d}'] = round(fC, 3)

            for i, fN in enumerate(grid_info[Z_key]['fNs']):
                data[f'fn{i+1:d}'] = round(fN, 3)

            url_values = urllib.parse.urlencode(data)
            full_url = AESOPUS_CGI + '?' + url_values
            f.write(full_url + '\n' * 3)


def validate_tables(grid_info, suffix='GS98', TOL=1e-6):
    comp = suffix.lower()
    for Z in grid_info['Zs']:
        Z_dict = grid_info[f'{Z:.6f}']
        Z_file = pathlib.Path(comp) /\
            (f'aesopus_z{Z:.6f}'.rstrip('0') + f'_{comp}_varcno.tab')
        print(f' > Validating table: {Z_file}')

        with open(Z_file) as f:
            for line in f.readlines():
                if line.startswith('# TABLE'):
                    TABLE, X, Y, Zref, Z, FCO, FC, FN, FO, C_O = line.split()[2::2]
                    TABLE = int(TABLE) - 1

                    if not all([abs(float(X) - Z_dict['Xs'][TABLE // 225]) < TOL,
                                abs(float(FCO) - Z_dict['fCOs'][TABLE % 225 // 25]) < TOL,
                                abs(float(FC) - Z_dict['fCs'][TABLE % 25 // 5]) < TOL,
                                abs(float(FN) - Z_dict['fNs'][TABLE % 5]) < TOL]):
                        print(line)


def compile_tables(suffix='GS98', Zsun_dict={}):
    if pathlib.Path(f'{suffix}.yaml').exists(): return
    print(' > Making YAML file for compiling', suffix)

    with open('AGSS09.yaml', 'r') as f:
        my_yaml = f.readlines()

    for i, line in enumerate(my_yaml):
        my_yaml[i] = line.replace('ags09', suffix.lower())\
                         .replace('AGSS09', suffix)

        for k, v in Zsun_dict.items():
            if line.startswith(k):
                my_yaml[i] = f'{k}: {v}\n'

    with open(f'{suffix}.yaml', 'w') as f:
        f.writelines(my_yaml)
