import os # Pour interagir avec le système d'exploitation
import scipy # pour manipuler les données et visualiser les données
import skimage # pour le traitement d'images
import imageio # pour lire et écrire un large éventail de données d'image
from skimage import transform # pour transformer la transformation des images
import numpy as np # pour travailler avec des tableaux
from imageio import imread, imwrite # pour lire et écrire

dir_name = os.getcwd()

def mass_center(img,is_round=True):
    Y = img.mean(axis=1)
    X = img.mean(axis=0)
    Y_ = np.sum(np.arange(Y.shape[0]) * Y)/np.sum(Y)
    X_ = np.sum(np.arange(X.shape[0]) * X)/np.sum(X)
    if is_round:
        return int(round(X_)),int(round(Y_))
    return X_,Y_
    
def image_extract(img,newsize):
    x_s = np.where(img.mean(axis=0)!=0)[0].min()
    x_e = np.where(img.mean(axis=0)!=0)[0].max()
    
    y_s = np.where(img.mean(axis=1)!=0)[0].min()
    y_e = np.where(img.mean(axis=1)!=0)[0].max()
    
    x_c,_ = mass_center(img)
    x_s = x_c-newsize[1]//2
    x_e = x_c+newsize[1]//2
    img = img[y_s:y_e,x_s if x_s>0 else 0:x_e if x_e<img.shape[1] else img.shape[1]]
    return skimage.transform.resize(img,newsize)
    
for dir_ in os.listdir(dir_name + '/Segmented_data_test'):
    folder = dir_name + '/Segmented_data_test/%s' % dir_
    create_folder = dir_name + '/GEI_TEST/%s' % dir_
    
    if not os.path.exists(create_folder):
        os.makedirs(create_folder)
        
    for sub_dir in os.listdir(folder):
        count = 0
        create_sub_folder = create_folder + '/%s/' % sub_dir
        sub_folder = folder + '/%s/' % sub_dir
        
        if not os.path.exists(create_sub_folder):
            os.makedirs(create_sub_folder)
        print('Start Save in ', folder)

        root_ = folder + '/' + sub_dir
        img_array_list = []
        for file in os.listdir(folder + '/' + sub_dir):
            img = imread(root_ + "/" + file)
            img = img[:,:,0]
            im_gei = image_extract(img,(460,460))
            img_array_list.append(im_gei)
            count +=1
        gei = np.mean(img_array_list,axis=0)
        imageio.imwrite(create_sub_folder +'/' + dir_ + str(count) +'.jpg', gei)