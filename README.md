In human identification, the process of identifying a person by gait is an emerging research trend in the field of visual surveillance and the only way to identify a person from a distance. Identifying a person by gait is a type of biometric technology, it has recently been used to recognize a person by the style of their walk.

This work first reviews the different techniques developed over the past decades to better understand the subject, then presents our solution approach based on the use of images to represent walking, such as the Gait Energy Image (GEI ) to create the model.

We start with the conventional approach to gait recognition which includes feature extraction and classification using SVM. Feature vectors for classification were constructed using dimension reduction on cumulants calculated by principal component analysis (PCA). Then a second Deep Learning model where the representations of walking are used as input into a convolutional neural network, which is used to perform classification or to extract a feature vector which is then classified using methods of machine learning to create our second model.

The models are trained and tested with a dataset that we have built, containing seven (7) people in a 180 degree angle.
After implementation and experimentation, the results are interesting, on the other hand open to the necessary improvements. 
