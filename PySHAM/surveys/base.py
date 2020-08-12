"""lala"""
from abc import (ABCMeta, abstractmethod)
from six import add_metaclass


@add_metaclass(ABCMeta)
class BaseSurvey(object):
    """Base model to hold properties of different surveys that are being
    analysed
    """
    _name = None
    _redshift_range = None
    _apparent_mr_range = None
    _survey_area = None
    _data = None

    @property
    def name(self):
        """The name of the survey"""
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise ValueError("must provide a string for name")
        self._name = name

    @property
    def redshift_range(self):
        """Redshift range"""
        return self._redshift_range

    @redshift_range.setter
    def redshift_range(self, zrange):
        if not isinstance(zrange, tuple):
            raise ValueError("must provide a tuple")
        if not len(zrange) == 2:
            raise ValueError("must provide an upper and lower limit")
        self._redshift_range = zrange

    @property
    def apparent_mr_range(self):
        """Apparent magnitude range"""
        return self._apparent_mr_range

    @apparent_mr_range.setter
    def apparent_mr_range(self, apprange):
        if not isinstance(apprange, tuple):
            raise ValueError("must provide a tuple")
        if not len(apprange) == 2:
            raise ValueError("must provide an upper and lower limit")
        self._apparent_mr_range = apprange

    @property
    def survey_area(self):
        """Survey area in degrees"""
        return self._survey_area

    @survey_area.setter
    def survey_area(self, area):
        if not isinstance(area, float):
            raise ValueError("must provide a float")
        self._survey_area = area

    @property
    @abstractmethod
    def data(self):
        pass

    @abstractmethod
    def _parse_catalog(self):
        pass
