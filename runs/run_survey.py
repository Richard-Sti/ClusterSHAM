# Copyright (C) 2021  Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import numpy

from parser_wp_survey import Projected2PointCorrelationParser
from parser_survey import SurveyConfigParser

# Get argparse to specify the config and nthreads


def get_randoms(path, N):
    randoms = numpy.load(path)
    # Ensure I don't have to do this
    randoms = randoms[numpy.logical_and(randoms['RA'] > 100, randoms['RA'] < 270)]
    mask = numpy.random.choice(randoms.size, size=N, replace=False)
    randRA = randoms['RA'][mask]
    randDEC = randoms['DEC'][mask]
    return randRA, randDEC

def main():
    wp_parser = Projected2PointCorrelationParser('NSAconfig.toml')
    wp_model = wp_parser()
    survey_parser = SurveyConfigParser('NSAconfig.toml')
    survey = survey_parser()
    print('Size', survey['RA'].size)

    path = survey_parser.cnf['Main']['randoms_path']
    N = survey_parser.cnf['Main']['Nmult'] * survey['RA'].size
    randRA, randDEC = get_randoms(path, N)

    res = wp_model.survey_wp(survey['RA'], survey['DEC'], survey['ZDIST'], randRA, randDEC)
    # Save the result to the output folder?
    print(res)



if __name__ == '__main__':
    main()
