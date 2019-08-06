import os.path

from bmcs_exceptions import BmCS_Exception

class BaseModel(object):
    def __init__(self, *args, **kwargs):
        self._fname = None
        self._model = None

    def fname(self, fname=None, *args, **kwargs):
        if fname is not None:
            self._fname=fname
        return self._fname

    def model(self, model=None):
        if model is not None:
            self._model=model
        return self._model

    def from_file(self, fname):
        if (not os.path.exists(fname)) or (not os.path.isfile(fname)):
            msg = "File \"{}\" does not exist or not a file.".format(fname)
            raise BmCS_Exception(msg)

    def process(self, dX):
        if self.model() is None:
            msg = "Can't process data: model is not initialized."
            raise BmCS_Exception(msg)

        if not isinstance(dX, dict):
            msg = "\"dX\" parameter should be a dictionary."
            raise BmCS_Exception(msg)

        if len(dX) == 0:
            msg = "Dictionary parameter \"dX\" should not be empty."
            raise BmCS_Exception(msg)
