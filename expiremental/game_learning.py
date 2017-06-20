# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 13:45:53 2017

@author: dandrews

Based on https://www.youtube.com/watch?v=EaY5QiZwSP4

"""

import tables
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, Conv2DTranspose
from keras.layers import Dropout
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import numpy as np
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras import regularizers

import sys
if r'D:\local\machinelearning' not in sys.path:
    sys.path.append(r'D:\local\machinelearning')
from daml.parameters import hyperparameters    


class game_learn:
    
    def __init__(self, hp: hyperparameters, data_file_name: str):
        data_file = tables.open_file(data_file_name)
        self.hp = hp
        self.data_file = data_file
        self.data = data_file.root.data
        self.labels = data_file.root.labels
        self.input_shape = self.data[0].shape
        self.output_shape = self.labels[0].shape
        self.data_len = len(self.labels)
        train_split = int(0.8 * self.data_len)        
        self.train_order = np.arange(train_split)
        self.test_order= np.arange(train_split, self.data_len)
        np.random.shuffle(self.train_order)
        np.random.shuffle(self.train_order)
        self.data_index = 0
        self.checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             monitor='loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')
        self.early_stop = EarlyStopping(monitor='loss', patience=100, mode='auto')
        self.epoch = 0
        self.tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        self.reduce_lr_callback = ReduceLROnPlateau()


    def create_model_autoencoder(self):
        print("creating autoencoder model")
        K.clear_session()
        
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=self.input_shape ))
        model.add(Dropout(hp.dropout))
        model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2), data_format="channels_first", input_shape=self.input_shape))
        model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2), data_format="channels_first"))
        model.add(Dropout(hp.dropout))
        model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2), data_format="channels_first"))
        model.add(Conv2D(64, (3, 3), activation='elu', data_format="channels_first"))
        model.add(Conv2D(64, (3, 3), activation='elu', data_format="channels_first"))
        model.add(Dropout(hp.dropout))
        model.add(Conv2DTranspose(64, (3, 3), activation='elu', data_format="channels_first"))
        model.add(Conv2DTranspose(64, (3, 3), activation='elu', data_format="channels_first"))
        model.add(Conv2DTranspose(36, (6, 5), strides=(2,2), activation='elu', data_format="channels_first"))
        model.add(Conv2DTranspose(24, (6, 6), strides=(2,2), activation='elu', data_format="channels_first"))
        model.add(Conv2DTranspose(3, (5, 6), strides=(2,2), activation='elu', data_format="channels_first"))
        model.add(Lambda(lambda x: x*127.5+1.0))
        model.summary()        
        model.compile(optimizer=self.hp.optimizer,
              loss=self.hp.loss,
              metrics=['accuracy'])
        return model

    def create_model_classify_from_autoencoder(self, weights_file):
       
        model = self.create_model_classify(compile_model=False)
        model.load_weights(weights_file, by_name=True)

#        for layer in model.layers:
#            if 'conv2d' in layer.name:
#                layer.trainable = False
        model.compile(optimizer=self.hp.optimizer,
              loss=self.hp.loss,
              metrics=['accuracy'])
        model.summary()
        return model
    
    def create_model_classify(self, compile_model = True):
        print("creating classify model")
        data_format="channels_last"
        activation = 'tanh'
        K.clear_session()
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=self.input_shape ))
        #model.add(Dropout(hp.dropout))
        model.add(Conv2D(24, (5, 5), activation=activation, strides=(2, 2), data_format=data_format,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
        model.add(Conv2D(36, (5, 5), activation=activation, strides=(2, 2), data_format=data_format, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
        #model.add(Dropout(hp.dropout))
        model.add(Conv2D(48, (5, 5), activation=activation, strides=(2, 2), data_format=data_format,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
        model.add(Conv2D(64, (3, 3), activation=activation, data_format=data_format, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
        model.add(Conv2D(64, (3, 3), activation=activation, data_format=data_format, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
        #model.add(Dropout(hp.dropout))
        model.add(Flatten())
        model.add(Dense(100, activation=activation,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
        model.add(Dense(50, activation=activation))
        model.add(Dense(10, activation=activation))
        model.add(Dense(self.output_shape[0], activation='tanh', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
        
        
        if compile_model:
            model.summary()
            model.compile(
                  optimizer= Adam(clipnorm = .001, lr=self.hp.learning_rate),
                  loss=self.hp.loss,)
                  #metrics=['loss'])        
        return model
    
    def get_samples(self, num_samples: int, train = True):
        data = np.zeros((num_samples,) + self.input_shape, dtype=np.float32)
        labels = np.zeros((num_samples,) + self.output_shape, dtype=np.float32)
        
        for i in range(num_samples):
            if train:
                index = self.data_index % len(self.train_order)
                
            else:
                index = self.data_index % len(self.test_order)
                
            item = self.train_order[index]
            #print(index,item)
            data[i] = self.data[item]
            labels[i] = self.labels[item]
            self.data_index += 1
        return data, labels
    
    
    
    def train_autoencoder(self, model = None):
        sample = 1024
        if model == None:
            print('No model, creating one')
            self.model = self.create_model_autoencoder()
        else:
            print('using existing model.')
        x_train, y_train = self.get_samples(sample)
        x_test, y_test = self.get_samples(512, train=False)
        history = self.model.fit(                
                x_train, 
                x_train,
                callbacks=[self.checkpoint],
                epochs=self.hp.epochs + self.epoch, 
                batch_size=self.hp.minibatch_size, 
                verbose=1,
                validation_data=(x_test, x_test),
                initial_epoch=self.epoch)
        self.epoch += self.hp.epochs
        return history.history
    
    
    
    def train_model(self, model = None):
        sample = 1024
        if model == None:
            self.model = self.create_model_classify()
        x_train, y_train = self.get_samples(sample)
        x_test, y_test = self.get_samples(512, train=False)
        history = self.model.fit(                
                x_train, 
                y_train,
                callbacks=[
                        self.checkpoint,  
                        self.tbCallBack, 
                        self.reduce_lr_callback,
                        self.early_stop,
                        ],
                epochs=self.hp.epochs, 
                batch_size=self.hp.minibatch_size, 
                verbose=1,
                validation_data=(x_test, y_test),
                initial_epoch=self.epoch
                )
        self.epoch += self.hp.epochs
        return history.history
            
    def plot_data(self, data, name):                            
        #plotdata["avgloss"] = plotdata["loss"]
        plt.figure(1)    
        plt.subplot(211)
        plt.plot(data)
        plt.xlabel('Epoch number')
        plt.ylabel('name')
        #plt.yscale('log', basex=10)
        plt.title('Minibatch run vs. ' + name)
        plt.show()        
    

    
    def cleanup(self):
        self.data_file.close()
        
    def get_activations(self, model_inputs, print_shape_only=False, layer_name=None):
        model = self.model
        
        print('----- activations -----')
        activations = []
        inp = model.input
    
        model_multi_inputs_cond = True
        if not isinstance(inp, list):
            # only one input! let's wrap it in a list.
            inp = [inp]
            model_multi_inputs_cond = False
        names = [layer.name for layer in model.layers if
                   layer_name in layer.name or layer_name is None]
        outputs = [layer.output for layer in model.layers if
                   layer_name in layer.name or layer_name is None]  # all layer outputs
    
        funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    
        if model_multi_inputs_cond:
            list_inputs = []
            list_inputs.extend(model_inputs)
            list_inputs.append(1.)
        else:
            list_inputs = [model_inputs, 1.]
    
        # Learning phase. 1 = Test mode (no dropout or batch normalization)
        # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
        layer_outputs = [func(list_inputs)[0] for func in funcs]
        for layer_activations in layer_outputs:
            activations.append(layer_activations)
            if print_shape_only:
                print(layer_activations.shape)
            else:
                print(layer_activations)
                
        ret_val = {}
        for i in range(len(names)):
            ret_val[names[i]] = activations[i]
        return ret_val

    def create_new_inception_based_model(self):
        with open('inception.json', 'r') as j:
                inception_json = j.read()       
        inception_model = models.model_from_json(inception_json)
        inception_out_shape = inception_model.layers[-1].output.shape.as_list()
        inception_out_shape = tuple(inception_out_shape)
        output_shape = 4
        # Add custom classifier on the top
        x = inception_model.layers[-1].output
        #x = concatenate(axis=-1)(x)
        x = Flatten()(x)
        x = Dense(100, activation='elu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
        x = Dense(50, activation='elu')(x)
        x = Dense(10, activation='elu')(x)
        x = Dense(output_shape, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
        model = models.Model(inception_model.layers[0].input, x)
        model.compile(hp.optimizer, hp.loss)
        return model
        

    def load_latest_model(self):
        model_saves = glob.glob("model*")
        if len(model_saves) > 0:
            file_name = glob.glob("model*")[-1]
            self.model = models.load_model(file_name)
            m = re.search('\d+', file_name)  
            self.epoch = int(m.group(0))
        else:
            raise FileNotFoundError('No models to load')
            
    def show_random_prediction(self):
        x,y = self.get_samples(1)
        p = self.model.predict(x)
        plt.imshow(x.reshape(299,299), 'gray')
        plt.show()
        print('Predicted', p[0])
        print('Recorded', y[0])
        delta = p[0] - y[0]
        print('Delta', delta)
        return delta

#%%    
if __name__ == '__main__':

    K.set_image_data_format('channels_last')
    import glob
    import re    
    hp = hyperparameters()
    hp.epochs = 100
    hp.minibatch_size = 10
    hp.optimizer = 'adam'
    hp.learning_rate = 1e-4
    hp.loss = 'mse'
    hp.dropout = 0.20
    
#    tl = game_learn(hp, 'game_data_four_label.h5')
    tl = game_learn(hp, 'game_data_five.h5')

#%%
#    tl.load_latest_model();
#    history = tl.train_model(tl.model)


    
#%%
#    x,y = tl.get_samples(1)
#    activations = tl.get_activations(x, True, 'conv')
#
#    for name in sorted(activations):
#        print('\n\n')
#        print(name)
#        fig = plt.gcf()
#        fig.set_dpi(150)
#        plt.imshow(activations[name][0][0], 'gray')
#        plt.show() 

      