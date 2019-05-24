from render_module.main import RenderModule
import numpy as np
from pyquaternion import Quaternion
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from orientation_detection.NN_module import calculate_accuracy, calculate_angle_accuracy 



"""
    Function for generating train and test images.
    :param n_image: number of image
    :param image_size: size of image in pixel, will generate image with dimension: (image_size,image_size)
    :quaternion = if not None, generate image with given quaternion information
    :vectorOrientation = if not None, generate image with given vector Orienatation
    when both quaternion and vector Orientation is none, it will generate images with random quaternion and vector orientation
    :return : dictionary consists of images, Quaternions, vectorOrientations
"""
def get_images(n_image,image_size,quaternion=None,vectorOrientation=None, high_resolution = False):
    image_generator = RenderModule()
    
    image_generator.loadModel('models/09_real_chair/model.obj', 'models/09_real_chair/texture.jpg','models/09_real_chair/specular_map.jpg')
    image_generator.setModelSize(0.20) # size of the object (initially 0.16, then 0.20 which fits with current model)
    if high_resolution == False:
        fov = 12/180*np.pi # Field of view (initially 12, change to 20 just to generate train data)
    else: 
        fov = 20/180*np.pi # When generating training data to be feed to the object detection module, to be cropped
    
    image_generator.createBackground(RenderModule.FLAT_BACKGROUND)
    
    if quaternion is not None:
        return image_generator.getImages(numImages = n_image, width=image_size, \
               height = image_size,quaternion=quaternion,fov=fov,limitAngle=True,randomColors = False, blur=0,noiseLevel=0.7,limitRollAngle=True)
    elif vectorOrientation is not None:
        return image_generator.getImages(numImages = n_image, width=image_size, \
               height = image_size,vectorOrientation=vectorOrientation,fov=fov,limitAngle=True,randomColors = False, blur=0,\
               noiseLevel=0.7,limitRollAngle=True)
    else:    
        return image_generator.getImages(numImages = n_image, width=image_size, \
               height = image_size,fov=fov,limitAngle=True,randomColors = False, blur=0, noiseLevel = 0.7,limitRollAngle=True)
    
    
    
    
"""
    Function for splitting the list of images and the target to train and validation sets -> xTrain, xVal, yTrain, yVal
    :param images_dict: dictionary contains of list of images, quaternions, and vector orientations
    :param target_type: Quaternion or vectorOrientations, this will define what kind of format the target value will have. what we are using currently 
                        is vector Orientations with six number, first three number represent the forward vector, and last three numbers represent the                           upward vector
    :return : xTrain, xVal, yTrain, yVal
""" 
# To split data to train and validation set 
def get_train_val(images_dict, target_type):
    if target_type=='quaternions':
        #to get the quaternions numbers from the quaternions object
        target = np.array([np.array([quaternion[0],quaternion[1],quaternion[2],quaternion[3]]) for quaternion in images_dict[target_type]])    
    else:
        target = np.array([np.concatenate(vectorOrientation) for vectorOrientation in images_dict[target_type]])

    return train_test_split(images_dict['images'], target, test_size = 0.1, random_state = 42)




"""
    For reconstructing images from the predicted orientation and printed the pair of true target and the prediction. this is used only when testing the     model performance with the synthetic images (self generated images with labels of quaternions or vectorOrientations)
    :param y_true : target value
    :param y_pred : prediction value
    :param xTest  : test images
    :param target_type : quaternions or vectorOrientations
    :param image_size : image pixel size
""" 
def show_images(y_true,y_pred,xTest,target_type,image_size,mathTool):
    
    # To limit the print out of images to 200 pairs of images, since if it is too many, the system will crash
    if len(y_pred)<=10:
        num_test = len(y_pred)
    else:
        num_test = 10

    angle_discrepancies = []
    ## Angle discrepancy using new formula    
    f0=  y_pred[:,:3]
    u0= y_pred[:,3:]
    f1 = y_true[:,:3]
    u1 = y_true[:,3:]

    
    
    for i in range(len(f0)):
        angle_discrepancies.append(mathTool.angleDifference(f0[i],u0[i],f1[i],u1[i]))
        
    #angle_errors, angle_err = calculate_angle_accuracy(y_true,y_pred) 
    mse,rmse = calculate_accuracy(y_true,y_pred)
    
    #for i in range(num_test):
    #    if target_type == 'vectorOrientations':
    #        predicted_image = get_images(1,image_size,vectorOrientation=y_pred[i].reshape(2,3))
    #    else:
    #        predicted_image = get_images(1,image_size,quaternion=Quaternion(y_pred[i])) 
        
       # print('====================================================================================================')
       # print('Test data', i)
       # print ('true orientation= ',y_true[i])
       # print('prediction orientation = ', y_pred[i])
       # print('angle error initial formula = ', angle_errors[i])
       # print('angle error new formula=',angle_discrepancies[i])
       # plt.imshow(xTest[i].astype(int).reshape(image_size,image_size,3))
       # plt.show()
        
       # plt.imshow(np.array(predicted_image['images']).astype(int).reshape(image_size,image_size,3))    
       # plt.show()
       # print('====================================================================================================')

    print('mse = ', mse)
    print('rmse =', rmse)

    #print("angle Error : ", angle_err*180/np.pi)
    print('angle error new formula:', np.mean(np.array(angle_discrepancies))*180/np.pi)
