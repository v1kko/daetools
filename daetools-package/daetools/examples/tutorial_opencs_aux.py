"""
***********************************************************************************
                           tutorial_opencs_aux.py
                DAE Tools: pyOpenCS module, www.daetools.com
                Copyright (C) Dragan Nikolic
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************
"""
import os, sys

def compareResults(inputFilesDirectory, variables):
    try:
        import pandas
        expectedResultsCSV = inputFilesDirectory + '.csv'
        csv_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), expectedResultsCSV)
        df = pandas.read_csv(csv_filepath, sep=';', header=1, skiprows=None, quotechar='"', skipinitialspace=True, dtype=float)
        ye0 = df.iloc[0]
        ye = df.iloc[-1]
        csv_filepath = os.path.join(os.path.abspath(inputFilesDirectory), 'results', 'results-0.csv')
        df = pandas.read_csv(csv_filepath, sep=';', header=1, skiprows=None, quotechar='"', skipinitialspace=True, dtype=float)
        y0 = df.iloc[0]
        y = df.iloc[-1]
        if y['time'] != ye['time']:
            print(y['time'], ye['time'])
            raise RuntimeError('Time horizons do not match')
        print('Comparison between the OpenCS and the original results:')
        print('--------------------------------------------------------------')
        print('           %25s %25s' % (('t = 0.0').center(25), ('t = %.2f' % y['time']).center(25)))
        print('            ------------------------- ------------------------')
        print('           %12s %12s %12s %12s' % ('OpenCS', 'Original', 'OpenCS', 'Original'))
        print('--------------------------------------------------------------')
        for var in variables:
            print('%-10s %12.5e %12.5e %12.5e %12.5e' % (var, y0[var], ye0[var], y[var], ye[var]))

        print('--------------------------------------------------------------')
    except Exception as e:
        print(str(e))
