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

from parser_paper_model import PaperModelConfigParser


def main():
    parser = argparse.ArgumentParser(description='Paper model submitted.')
    parser.add_argument('--path', type=str, help='Config file path.')
    parser.add_argument('--sub_id', type=str, help='Subsample ID')
    args = parser.parse_args()

    parser = PaperModelConfigParser()

    # ... Now will have to come up with the MPI grid search
