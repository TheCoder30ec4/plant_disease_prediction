import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# define your model
model = "C:/Users/Varun/OneDrive/Desktop/ml project/potatoes.h5"

# generate predictions on test data
y_true = "C:/Users/Varun/OneDrive/Desktop/ml project/Data_Set/dataset/test/Potato___healthy/1ae826e2-5148-47bd-a44c-711ec9cc9c75___RS_HL 1954.JPG"
y_pred = model.predict(y_true)

# calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       xlabel='Predicted label',
       ylabel='True label',
       title='Confusion matrix')
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.show()
