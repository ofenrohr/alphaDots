import keras.callbacks as cbks

class DebugLogger(cbks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Epoch begin: ")
        print(logs)

    def on_batch_begin(self, batch, logs=None):
        print("Batch begin: ")
        print(logs)

    def on_train_begin(self, logs=None):
        print("Train begin: ")
        print(logs)