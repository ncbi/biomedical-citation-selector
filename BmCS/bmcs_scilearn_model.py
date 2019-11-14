import os

from sklearn.externals import joblib

from .bmcs_basemodel import BaseModel
from .bmcs_exceptions import BmCS_Exception

class SciLearnModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def from_file(self, fname):
        if fname is None:
            fname = "None"

        # Init from supplied file or "standard" place
        if not os.path.exists(fname):
            fname_local = resource_filename(__name__, "models/ensemble.joblib")
            if not os.path.exists(fname_local):
                msg = "SciLearn model file can't be found neither as \"{}\" nor as \"{}\".".format(fname, fname_local)
                raise BmCS_Exception(msg)
            else:
                fname = fname_local

        super().from_file(fname)

        model = None
        try:
            model = joblib.load(fname)
        except Exception as e:
            msg = "Can't load SciLearn model from \"{}\". Reason: \"{}\".".format(fname, str(e))
            raise BmCS_Exception(msg)

        if model is None:
            msg = "Empty SciLearn model detected. Source file: \"{}\".".format(fname)
            raise BmCS_Exception(msg)

        self.model(model)

    def process(self, dX):
        super().process(dX)

        result = None
        try:
            result = self.model().predict_proba(dX)[:, 0]
        except Exception as e:
            msg = "Can't run prediction on SciLearn model! Reason: \"{}\".".format(str(e))
            raise BmCS_Exception(msg)

        return result
