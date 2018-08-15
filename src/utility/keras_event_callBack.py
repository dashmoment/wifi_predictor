import save_load_Keras as keras_io
from keras.callbacks import TensorBoard
import keras


class saveModel_Callback(keras.callbacks.Callback):
    
    def __init__(self, save_epoch, model, graph_path, weight_path):
        
        self.model = model
        self.save_epoch = save_epoch
        self.graph_path = graph_path
        self.weight_path = weight_path
       
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        
        if epoch%self.save_epoch == 0:
            keras_io.save_model( self.model, self.graph_path, self.weight_path)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        return

def tensorBoard_Callback(
                            log_dir='../logs', 
                            histogram_freq=0,
                            write_graph=True, 
                            write_images=False):
    
    tensorboard = TensorBoard(
                                log_dir=log_dir, 
                                histogram_freq=histogram_freq,
                                write_graph=write_graph, 
                                write_images=write_images)
    
    return tensorboard
