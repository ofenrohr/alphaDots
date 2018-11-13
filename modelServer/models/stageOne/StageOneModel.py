from keras.models import load_model


class StageOneModel:
    def __init__(self):
        print "StageOneModel()"

        self.model = load_model('model/train-checkpoint-5x5kernels.h5')

    def predict(self, input):
        return model.predict(input)

