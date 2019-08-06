from keras.models import model_from_json
from pkg_resources import resource_filename
import os.path

from embedding_custom import EmbeddingWithDropout

from bmcs_basemodel import BaseModel
from bmcs_exceptions import BmCS_Exception


class CnnModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Init CNN model from file
    def from_file(self, fname, weights_fname, d_custom_objects=None, loss='binary_crossentropy', optimizer='adam'):
        super().from_file(fname)

        if d_custom_objects is None:
            d_custom_objects = {EmbeddingWithDropout.__name__: EmbeddingWithDropout}

        if (not os.path.exists(weights_fname)) or (not os.path.isfile(weights_fname)):
            msg = "Weights file \"{}\" does not exist or not a file.".format(weights_fname)
            raise BmCS_Exception(msg)

        model_json, model = None, None

        try:
            with open(fname, 'rt') as cnn_fd:
                model_json = cnn_fd.read()
        except Exception as e:
            msg = "Can't get content of \"{}\". Reason: \"{}\".".format(fname, str(e))
            raise BmCS_Exception(msg)

        try:
            model = model_from_json(model_json, custom_objects=d_custom_objects)
        except Exception as e:
            msg = "Can't init CNN model from JSON file: \"{}\". Reason: \"{}\".".format(fname, str(e))
            raise BmCS_Exception(msg)

        if model is None:
            msg = "Empty CNN model detected. JSON file: \"{}\".".format(fname)
            raise BmCS_Exception(msg)

        try:
            model.load_weights(weights_fname)
        except Exception as e:
            msg = "Can't load weights from \"{}\". Reason: \"{}\".".format(weights_fname, str(e))
            raise BmCS_Exception(msg)

        try:
            model.compile(loss=loss, optimizer=optimizer)
        except ValueError as ve:
            msg = "Can't compile CNN model! Reason: \"{}\".".format(str(ve))
            msg += " Check \"optimizer\", \"loss\", \"metrics\" or \"sample_weight_mode\" parameters values."
            raise BmCS_Exception(msg)
        except Exception as e:
            msg = "Can't compile CNN model! Reason: \"{}\".".format(str(e))
            raise BmCS_Exception(msg)

        self.model(model)

    # Process source data
    def process(self, dX):
        super().process(dX)

        result = None
        try:
            result = self.model().predict(dX).flatten()
        except ValueError as ve:
            msg = "Can't run prediction on CNN model! Reason: \"{}\".".format(str(ve))
            msg += " Possible mismatch between the provided input data and the model's expectations."
            raise BmCS_Exception(msg)
        except Exception as e:
            msg = "Can't run prediction on CNN model! Reason: \"{}\".".format(str(e))
            raise BmCS_Exception(msg)

        return result
