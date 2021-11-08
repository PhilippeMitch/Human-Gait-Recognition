# Importation de toutes les bibliothèques nécessaires
import os # Pour interagir avec le système d'exploitation
import cv2 # permet d'acceder aux fonctions d'openCv
import argparse # permet de lancer le programme tout en fournissant des parametres 
import imutils # pour le traitement de base des images

# détecteur
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# construire l'analyseur d'arguments et analyser les arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", type=str, required=True, help="chemin d'accès à votre vidéo d'entrée")
args = vars(ap.parse_args())

# Lire la vidéo à partir du chemin spécifié
cam = cv2.VideoCapture(args["video"]) 

try:
    # création d'un dossier nommé test_data
    if not os.path.exists('test_data'): 
        os.makedirs('test_data') 
  
# si le fichier n'est pas créé, générer une erreur 
except OSError: 
    print ('Erreur : création du répertoire de test_data') 
  
# image 
currentframe = 0

while(True): 
    # lecture de la vidéo 
    ret,frame = cam.read() 
  
    if ret: 
        # si la durée de la vidéo n'est pas encore écoulé, continuez à créer des images
        name = 'test_data/frame' + str(currentframe)
        print ('Creating...' + name)

        frame = imutils.resize(frame,
                      width=min(400, frame.shape[1]))
   
        # Détecter toutes les régions de l'image contenant des piétons
        (regions, _) = hog.detectMultiScale(frame, 
                                            winStride=(4, 4),
                                            padding=(3, 3),
                                            scale=1.2)
  
        # écrire les images extraites 
        # Dessiner les régions dans l'image
        idx = 0
        for (x, y, w, h) in regions:

            if w>60 and h>60:
                idx+=1
                new_img=frame[y:y+h,x:x+w]
                # recadrage des images
                cv2.imwrite(name + '.png', new_img)

        # compteur croissant pour qu'il montre combien d'images sont créés
        currentframe += 1
    else: 
        break
  
# Libérez tout l'espace et les fenêtres une fois terminé 
cam.release() 
cv2.destroyAllWindows()