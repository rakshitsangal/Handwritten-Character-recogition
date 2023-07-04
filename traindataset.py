
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
DATADIR="C:\dataset1\letter"

CATEGORIES=["A","B","C","D"]
for category in CATEGORIES:
    path=DATADIR+'/'+category
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break
print(img_array)
print(img_array.shape)
IMG_SIZE=30
new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array,cmap="gray")
plt.show()


training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.imread(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()

print(len(training_data))


import random
random.shuffle(training_data)

for sample in training_data:
    print(sample[1])

X=[]
Y=[]

for features,labels in training_data:
    X.append(features)
    Y.append(label)
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

import pickle

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("Y.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close()

pickle_in=open("X.pickle","rb")
X=pickle.load(pickle_in)

X[0]
