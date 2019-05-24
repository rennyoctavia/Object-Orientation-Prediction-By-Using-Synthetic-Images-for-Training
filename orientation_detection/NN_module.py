import keras
from keras.models import Sequential,Model,load_model
from keras.layers import Conv2D, MaxPooling2D,Dense, Dropout, Flatten,Activation,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications import VGG16
import numpy as np
from pyquaternion import Quaternion
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt
from keras import optimizers

"""
    Customized loss function to fit with the vector orientation property. this function will normalize the predicted vectors to leghth of 1 and calculate the mean squared error. Due to, what matters for the accuracy is the vector orientation, the length of the vector does not matter.
    :param y_true : The target value
    :param y_pred : The prediction value
    :return mse
"""
def customized_mse(y_true,y_pred):
    vector_1 = y_pred[:,:3]
    vector_2 = y_pred[:,3:]

    # To normalize the predictions vectors to have length of 1
    normalized_vector_1 = tf.divide(vector_1,tf.sqrt(tf.reduce_sum(tf.square(vector_1),1,keepdims=True)))
    normalized_vector_2 = tf.divide(vector_2,tf.sqrt(tf.reduce_sum(tf.square(vector_2),1,keepdims=True)))
    normalized_y_pred = tf.concat([normalized_vector_1, normalized_vector_2], axis=1)
    return tf.reduce_mean(tf.square(tf.subtract(normalized_y_pred,y_true)))


def load_CNN_model(target_type,model_filename):
    return keras.models.load_model(model_filename,custom_objects={'customized_mse': customized_mse})



"""
    For calculating the mse and rmse for the scaled vectors
    :param y_true : The target value
    :param y_pred : The prediction value
    :return mse and rmse
"""
def calculate_accuracy(y_true,y_pred):
    #normalize the y_pred
    vector1 = y_pred[:,:3]
    vector2 = y_pred[:,3:]
    num_row = y_pred.shape[0]

    normalized_vectors1 = np.divide(vector1,np.sqrt(np.sum(np.square(vector1),axis=1).reshape(num_row,1)))
    normalized_vectors2=  np.divide(vector2,np.sqrt(np.sum(np.square(vector2),axis=1).reshape(num_row,1)))

    normalized_predictions = np.concatenate((normalized_vectors1,normalized_vectors2),axis=1)
    
    mse = mean_squared_error(y_true, normalized_predictions)
    return mse,mse**0.5


"""
    For calculating the angle difference between the training and test 
    :param y_true : The target value
    :param y_pred : The prediction value
    :return angle diffence
"""
def calculate_angle_accuracy(y_true,y_pred):
    #normalize the y_pred
    forward_pred = y_pred[:,:3]
    upward_pred = y_pred[:,3:]
    
    num_row = y_pred.shape[0]
    #normalized
    f_pred = np.divide(forward_pred,np.sqrt(np.sum(np.square(forward_pred),axis=1).reshape(num_row,1)))
    u_pred = np.divide(upward_pred,np.sqrt(np.sum(np.square(upward_pred),axis=1).reshape(num_row,1)))

    f_true = y_true[:,:3]
    u_true = y_true[:,3:]
    
    cross_f_u_pred = np.cross(f_pred,u_pred)
    cross_f_u_true = np.cross(f_true,u_true)

    V = np.cross(f_pred,f_true) + np.cross(u_pred,u_true) +np.cross(cross_f_u_pred,cross_f_u_true)
    len_V = np.sqrt(np.sum(np.square(V),axis = 1))
    angle_errors = np.arcsin(len_V/2)
    
    return angle_errors, np.mean(angle_errors)


"""
    For calling function to create model based on the parameters, train model, also save the best model, history, and parameters
    :param filename : filename for saving the information of the model (this is in string of date and time)
    :param xTrain : training data
    :param xVal : validation data
    :param yTrain : target value of training data
    :param yVal : target value of validation data
    :return history
"""
# Function to train the model, save the best model, history and the parameters
def train_model(filename,xTrain, xVal, yTrain, yVal, ML_type, parameters):

    # If machine learning algorithm choosen is Convolutional neural network
    if ML_type == 'conv':
        model = get_CNN(parameters)
        filepath_model = "../trained_model/new_chair/CNN/CNN_model_"+filename
        filepath_hist = "../trained_model/new_chair/CNN/CNN_history_"+filename
        filepath_param = "../trained_model/new_chair/CNN/CNN_param_"+filename
    
    elif ML_type == 'vgg16':
        model = get_vgg16(parameters)
        filepath_model = "../trained_model/new_chair/vgg16/vgg16_model_"+filename
        filepath_hist = "../trained_model/new_chair/vgg16/vgg16_history_"+filename
        filepath_param = "../trained_model/new_chair/vgg16/vgg16_param_"+filename
   
    elif ML_type == 'resnet50':
        model = get_resnet50(parameters)
        filepath_model = "../trained_model/new_chair/resnet50/resnet50_model_"+filename
        filepath_hist = "../trained_model/new_chair/resnet50/resnet50_history_"+filename
        filepath_param = "../trained_model/new_chair/resnet50/resnet50_param_"+filename
    
    # If machine learning algorithm chosen is Fully connected neural network
    else:
        n_input = len(xTrain[0])
        model = get_FCNN(parameters,n_input)
        filepath_model = "../trained_model/new_chair/FC/FC_model_"+filename
        filepath_hist = "../trained_model/new_chair/FC/FC_history_"+filename
        filepath_param = "../trained_model/new_chair/FC/FC_param_"+filename
    
    # setting callbacks
    callbacks = [
    #keras.callbacks.TensorBoard(logdir),
    keras.callbacks.EarlyStopping(patience=10),
    keras.callbacks.ModelCheckpoint(filepath=filepath_model, monitor='val_loss', save_best_only=True),
    ]
    
    # fit the model into training data
    history = model.fit(xTrain, yTrain,
          batch_size=parameters['batch_size'],
          epochs=parameters['epochs'],
          verbose=1,
          validation_data=(xVal, yVal),
          callbacks = callbacks)
    
    #Save the history
    history_dict = history.history
    f = open(filepath_hist,"wb")
    pickle.dump(history_dict,f)
    f.close()
    
    #Save parameters in file
    f=open(filepath_param,"w")
    for k,v in parameters.items():
        s=str(k)+"   "+str(v)+"\n"
        b=f.write(s)
    f.close()
    
    print(model.summary())
        
    return history


"""
    Function to create CNN model based on the passed parameters
    :param parameters : parameters for building the model, in a form of dictionary with below example of parameters
     - 'image_size':image_size,
     - 'num_images': num_images,
     - 'target_type':'vectorOrientations',
     - 'batch_size':64,
     - 'epochs':100,
     - 'conv':[[96,11,'relu'],[384,5,'relu']],
     - 'padding':'same',  
     - 'maxpool':[2,2],  
     - 'FC':[[512,'relu'],[64,'relu']],
     - 'dropout':[0.0,0.3],
     - 'output_activation':'linear',
     - 'optimizer': 'adam' 
    :return model
"""
def get_CNN(parameters):

    if parameters['target_type'] == 'quaternions':
        output_layer = 4
    else:
        output_layer = 6
    
    model = Sequential()
    
    ## Set up convolutional layers
    for i in range(len(parameters['conv'])):
        if i == 0:
            model.add(Conv2D(parameters['conv'][i][0], (parameters['conv'][i][1], parameters['conv'][i][1]),\
                             padding=parameters['padding'][i], input_shape=(parameters['image_size'], parameters['image_size'], 3)))
        elif i == 1:
            model.add(Conv2D(parameters['conv'][i][0], (parameters['conv'][i][1], parameters['conv'][i][1]),padding=parameters['padding'][i]))  
        else:
            model.add(Conv2D(parameters['conv'][i][0], (parameters['conv'][i][1], parameters['conv'][i][1])))  
        
        model.add(Activation(parameters['conv'][i][2]))
        model.add(MaxPooling2D(pool_size=(parameters['maxpool'][i], parameters['maxpool'][i])))
    
    ## Set up FC layers
    model.add(Flatten())
    for j in range(len(parameters['FC'])):
        model.add(Dense(parameters['FC'][j][0],activation=parameters['FC'][j][1]))
        #model.add(Activation(parameters['FC'][j][1]))
        model.add(Dropout(rate=parameters['dropout'][j]))
        
    ## Set output layer
    model.add(Dense(output_layer))
    model.add(Activation(parameters['output_activation'])) 
        
    # Compile the model
    model.compile(loss=customized_mse,
    optimizer=parameters['optimizer'])
    
    print(model.summary())

    return model

"""
    Function to create Fully Connected Neural Network model based on the passed parameters
    :param parameters : parameters for building the model, in a form of dictionary with below example of parameters
    - 'image_size':image_size,
    - 'num_images': num_images,
    - 'target_type':'vectorOrientations',
    - 'batch_size':20,
    - 'epochs':5,
    - 'FC':[[512,'relu'],[256,'relu'],[128,'relu']],
    - 'dropout':[0.0,0.3],
    - 'output_activation':'linear',
    - 'optimizer': 'adam'
    :return model
"""
def get_FCNN(parameters, n_input):
    
    if parameters['target_type'] == 'quaternions':
        n_output = 4
    else:
        n_output = 6
    
    model = Sequential()
    
    for i in range(len(parameters['FC'])):
        if i == 0:
            model.add(Dense(parameters['FC'][i][0], input_dim=n_input, kernel_initializer='normal', activation=parameters['FC'][i][1]))
        else:
            model.add(Dense(parameters['FC'][i][0], activation=parameters['FC'][i][1])) 
            model.add(Dropout(rate=parameters['dropout'][i-1]))
        
    model.add(Dense(n_output, activation=parameters['output_activation']))
    model.compile(loss=customized_mse, optimizer=parameters['optimizer'])
    
    return model


"""
    Function to create VGG16 model based on the passed parameters
    :param parameters : parameters for building the model, in a form of dictionary with below example of parameters
     -'image_size':image_size,
     -'num_images': num_images,
     -'target_type':'vectorOrientations',
     -'not_freeze': 0, #specify number of vgg which is not going to be freezed for training, 0 means freeze all
     -'pre_trained':True,
     -'batch_size':64,
     -'epochs':100,
     -'FC':[[512,'relu'],[512,'relu']],
     -'dropout':[0.0,0.3],
     -'output_activation':'linear',
     -'optimizer': 'adam',
     -'learning_rate': 'default',
     -'preprocessing':'yes'
    :return model
"""
def get_vgg16(parameters):
    if parameters['target_type'] == 'quaternions':
        output_layer = 4
    else:
        output_layer = 6
    
    print('pre_trained', parameters['pre_trained'])
    if parameters['pre_trained'] == True:
        weight = 'imagenet'
        print('using weight', weight)
    else:
        weight = None
        print('Not using pre-trained weight')
        
    if parameters['optimizer'] == 'sgd':
        optimizer = optimizers.SGD(lr=parameters['learning_rate'], decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optimizer = parameters['optimizer']
    
    vgg16_conv = VGG16(weights= weight, include_top=False, input_shape=(parameters['image_size'], parameters['image_size'], 3))
    
    #Define if layers are freezed from training or not
    if parameters['pre_trained'] == True:
        #open all layers for training   
        if parameters['not_freeze'] == 'all':
            print('All layers open for training')            
            
        elif parameters['not_freeze'] == 0: #freeze all layers
            for layer in vgg16_conv.layers:
                layer.trainable = False
        else:
            #Freeze the layers except the last n layers
            for layer in vgg16_conv.layers[:-parameters['not_freeze']]:
                layer.trainable = False
            
    
    #Check the trainable status of the individual layers
    for layer in vgg16_conv.layers:
        print(layer, layer.trainable)
    
    # Create the model
    model = Sequential()
 
    # Add the vgg convolutional base model
    model.add(vgg16_conv)
    print(vgg16_conv.summary())
 
    # Add new layers
    model.add(Flatten())
    for i in range(len(parameters['FC'])):
        model.add(Dense(parameters['FC'][i][0], activation=parameters['FC'][i][1])) 
        model.add(Dropout(rate=parameters['dropout'][i-1]))
        
    model.add(Dense(output_layer, activation=parameters['output_activation']))
    
    if parameters['loss']=='custom':
        print('loss function used : custom')
        model.compile(loss=customized_mse, optimizer=optimizer)
    else:
        print('loss function used :'+parameters['loss'])
        model.compile(loss=parameters['loss'], optimizer=optimizer)
    
    # Show a summary of the model. Check the number of trainable parameters
    print(model.summary())
    
    return model


"""
    Function to create resnet50 model based on the passed parameters
    :param parameters : parameters for building the model, in a form of dictionary with below example of parameters
     -'image_size':image_size,
     -'num_images': num_images,
     -'target_type':'vectorOrientations',
     -'not_freeze': 0, #specify number of vgg which is not going to be freezed for training
     -'pre_trained':True,
     -'batch_size':64,
     -'epochs':100,
     -'output_activation':'linear',
     -'optimizer': 'adam'
     -'learning_rate': 'default',
     -'preprocessing':'yes'
    :return model
"""
def get_resnet50(parameters):
    if parameters['target_type'] == 'quaternions':
        output_layer = 4
    else:
        output_layer = 6
    
    if parameters['pre_trained'] == True:
        weight = 'imagenet'
    else:
        weight = None
        
    if parameters['optimizer'] == 'sgd':
        optimizer = optimizers.SGD(lr=parameters['learning_rate'])
    else:
        optimizer = parameters['optimizer']

    # Get base model
    base_model = ResNet50(include_top=False, weights=weight,input_shape=(parameters['image_size'], parameters['image_size'], 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Dense(512, activation='relu',name='fc8')(x)
    last_layer = Dense(output_layer, activation=parameters['output_activation'], name='fc9')(x)
    model = Model(inputs=base_model.inputs, outputs=last_layer)

    if parameters['pre_trained'] == True:
        #open all layers for training   
        if parameters['not_freeze'] == 'all':
            print('All layers open for training')            
        elif parameters['not_freeze'] == 0: #freeze all layers
            for layer in base_model.layers:
                layer.trainable = False
        else:
            #Freeze the layers except the last n layers
            for layer in base_model.layers[:-parameters['not_freeze']]:
                layer.trainable = False
    
    # Freeze the batch normalization layers
    for i in range(len(model.layers)):
        if str.startswith(model.layers[i].name, 'bn'):
            model.layers[i].trainable = False
        if str.startswith(model.layers[i].name, 'bn_conv1'):
            model.layers[i].trainable = True
           
    #Print the trainable status of the individual layers
    for layer in model.layers:
        print(layer, layer.trainable)

    model.compile(loss=customized_mse, optimizer=optimizer)
    print(model.summary())
    return model 


"""
    For getting prediction 
    :param target_type : Quaternions or vectorOrientations
    :param model_filename : path to the model to be load
    :param xTest : the array of images to be predicted
    :yTest = default is empty list. for real images cropped from the webcam, this list will be empty. not empty when the images to be predicted is the 
     self generated
    :return predictions
"""
def get_prediction(target_type,model_filename,xTest, yTest =[]):
    #Load the saved model
    model = keras.models.load_model(model_filename,custom_objects={'customized_mse': customized_mse})
    
    if len(yTest) != 0: 
        test_evaluation = model.evaluate(xTest, yTest, verbose=1)
        print('Test loss/MSE:', test_evaluation)
        #print('Test accuracy:', test_evaluation[1])
        print('RMSE:',test_evaluation**0.5)
    
    # Run the prediction into test data
    predictions = model.predict(xTest)
    
    return predictions


"""
    Plotting learning curves from the history
"""
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    #plt.gca().set_ylim(0, 1)
    plt.xlabel("Epoch",fontsize=12)
    plt.ylabel("Loss",fontsize=12)
    plt.show()
    


