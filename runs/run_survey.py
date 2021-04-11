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
import argparse
from joblib import dump

from parser_wp import Projected2PointCorrelationParser
from parser_survey import SurveyConfigParser



def get_randoms(path, N):
    """
    Loads `N` random RA-dec points from file `path`. This file must respect
    the survey angular geometry.

    Parameters
    ----------
    path : str
        Path to the randoms file.
    N : int
        Number of randoms to be selected.

    Returns
    -------
    randRA : numpy.ndarray
        Right ascension randoms.
    randDEC : numpy.ndarray
        Declination randoms.
    """
    randoms = numpy.load(path)
    mask = numpy.random.choice(randoms.size, size=N, replace=False)
    randRA = randoms['RA'][mask]
    randDEC = randoms['DEC'][mask]
    return randRA, randDEC


def main():
    parser = argparse.ArgumentParser(description='Survey wp submitter.')
    parser.add_argument('--path', type=str, help='Config file path.')
    parser.add_argument('--sub_id', type=str, help='Subsample ID')
    args = parser.parse_args()

    survey_parser = SurveyConfigParser(args.path, args.sub_id)
    survey = survey_parser()
    # Get the wp_model
    wp_parser = Projected2PointCorrelationParser(args.path)
    wp_model = wp_parser()
    # Get the randoms
    randRA, randDEC = get_randoms(
            survey_parser.cnf['Main']['randoms_path'],
            survey_parser.cnf['Main']['Nmult'] * survey['RA'].size)

    attr = survey_parser.cut_condition.attr[0]
    cut_range = survey_parser.cut_condition.ext_range

    print('Calculating {} for {}'.format(attr, cut_range))
    res = wp_model.survey_wp(survey['RA'], survey['DEC'],
                             survey['COMOVING_DIST'], randRA,
                             randDEC, is_comoving=True)
#                             Npoints_kmeans=survey['RA'].size * 5)

    res.update({'cut_range': cut_range,
                'attr': attr})
    # A good place to append the external range here...
    fname = survey_parser.cnf['Main']['out_folder']
    fname += "CF_{}_{}_{}_{}.p".format(attr, args.sub_id, *cut_range)
    print("Saving results to {}".format(fname))
    print(res)
    dump(res, fname)
    print("Done!")


if __name__ == '__main__':
    main()
