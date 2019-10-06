import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os
labenc=LabelEncoder()
onehotenc=OneHotEncoder(sparse=False)

def load_data(dirtype):
  images=[]
  labels=[]
  filenames=[]
  dirlabelpaths=[os.path.join(dirtype,d) for d in os.listdir(dirtype) if os.path.isdir(os.path.join(dirtype,d))]
  for d in os.listdir(dirtype):
    if(os.path.isdir(os.path.join(dirtype,d))):
      for m in os.listdir(os.path.join(dirtype,d)):
        if "pyn" not in m:
          labels.append(m)

  for dirlabelpath in dirlabelpaths:
    for f in os.listdir(dirlabelpath):
      if f.endswith(".jpg"):
        filenames.append(os.path.join(dirlabelpath,f))
  for f in filenames:
    images.append(np.array(cv2.resize(cv2.imread(f,1),(64,64)))/255)
#     print(f)
  return np.array(images),np.array(labels)


images_train,labels_train=load_data("gendata/")
images_train=np.array([image.flatten() for image in images_train])
labels_train=labenc.fit_transform(labels_train)
labels_train=onehotenc.fit_transform(labels_train.reshape(-1,1))


print(images_train.shape,labels_train.shape)

x=tf.placeholder(tf.float32,[None,64*64],name="x")
y=tf.placeholder(tf.float32,[None,24],name="y")

W1=tf.Variable(tf.random_normal([64*64,300],stddev=0.03),name="W1")
b1=tf.Variable(tf.random_normal([300]),name="b1")

W2=tf.Variable(tf.random_normal([300,24],stddev=0.03),name="W2")
b2=tf.Variable(tf.random_normal([24]),name="b2")

layer1=tf.nn.relu(tf.add(tf.matmul(x,W1),b1))
y_=tf.nn.softmax(tf.add(tf.matmul(layer1,W2),b2))

yc=tf.clip_by_value(y_,1e-10,0.9999999)
cross_entropy=-tf.reduce_mean(tf.reduce_sum(y*tf.log(yc)+(1-y)*tf.log(1-yc),axis=1))

optimiser=tf.train.GradientDescentOptimizer(learning_rate=0.003).minimize(cross_entropy)

init_op=tf.global_variables_initializer()

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))




tf.set_random_seed(1234)
sess=tf.Session()
sess.run(init_op)

batch_size=20
batches=int(3880/batch_size)
epochs=50
print(images_train[4].shape)
for i in range(epochs):
  cost=0
  for b in range(batches):
    _,loss=sess.run([optimiser,cross_entropy],feed_dict={x:images_train[b*batch_size:(b+1)*batch_size],y:labels_train[b*batch_size:(b+1)*batch_size]})
    cost+=loss/batches
  print("Done with epoch ",i,"cost:",cost)





images_test,labels_test=load_data("test/")
labels_test=labenc.fit_transform(labels_test)
labels_test=onehotenc.fit_transform(labels_test.reshape(-1,1))



print(images_test.shape,labels_test.shape)

predicted,acc=sess.run([correct_prediction,accuracy],feed_dict={x:[image.flatten() for image in images_test],y:labels_test})

print(predicted,acc)


for i in range(len(labels_test)):
  # plt.subplot(5,1,i+1)
  color='green' if predicted[i] else "red"
  plt.text(16,20,"Prediction:{0} truth {1}".format(predicted[i],labels_test[i]),fontsize=10,color=color)
  plt.imshow(images_test[i],cmap="gray")
  plt.show() 
