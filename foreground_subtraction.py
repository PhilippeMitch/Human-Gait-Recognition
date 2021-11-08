import torchvision.transforms as T # pour la transformations des images
import numpy as np # pour travailler avec des tableaux
import torch # permet d'accéder aux fonctions de PyTorch
import os # Pour interagir avec le système d'exploitation

from torchvision import models # pour acceder aux models pre-entrainer
from PIL import Image # Manupulation des images

# charger le modèle fcn_resnet101
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
dir_name = os.getcwd()

def background_remove(img):
    # Appliquer les transformations nécessaires
    trf = T.Compose([T.Resize(256),
                    T.CenterCrop(460),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    # Passer l'entrée à travers le net
    out = fcn(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    return om

# Définir la fonction d'assistance
def decode_segmap(image, nc=21):

    # (192, 128, 128), couleur de la personne avant
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128,
                                                        0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64,
                                                              0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0,
                                                          128), (64, 128, 128), (250, 250, 250),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb
    
for dir_ in os.listdir(dir_name + '/test_data2'):
    folder = dir_name + '/test_data2/%s' % dir_
    new_folder = dir_name + '/Segmented_data_2/%s' % dir_
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    print('Commencer à enregistrer dans ', folder)

    for subdir in os.listdir(folder):
    
        new_subfolder = new_folder + '/%s' % subdir
        subfolder = folder + '/%s' % subdir
        if not os.path.exists(new_subfolder):
            os.makedirs(new_subfolder) 
        for file in os.listdir(folder + '/' + subdir):
            print(file)
            img = Image.open(folder + '/' + subdir + "/" + file)
            om = background_remove(img)
            rgb = decode_segmap(om)
            im = Image.fromarray(rgb)
            # #UPLOAD_DIRECTORY + 'myphoto.jpg', 'JPEG'
            print('Commencer à enregistrer dans ', subfolder + "/" + file)
            im.save(new_subfolder + '/' + file, 'JPEG')