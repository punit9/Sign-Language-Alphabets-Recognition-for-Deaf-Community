import os
import cv2
import numpy as np
import pickle as p
#import autopy as ap
import random as ran

#import tensorflow as tf
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
      	labels.append(d)

  for dirlabelpath in dirlabelpaths:
    for f in os.listdir(dirlabelpath):
      if f.endswith(".jpg"):
        filenames.append(os.path.join(dirlabelpath,f))
  for f in filenames:
    images.append(np.array(cv2.resize(cv2.imread(f,1),(64,64)))/255)
#     print(f)
  return np.array(images),np.array(labels)


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
  
def createCnn():
    model=Sequential()#32,(3,3)

    model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=[64,64,3]))

    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))#32,(3,3)

    model.add(Dropout(0.25))
  #64,(3,3),
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))

    model.add(Conv2D(64,(3,3),activation='relu'))#tivation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.5))#32,(3,3)
  
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))#model.add(Conv2D(64,(3,3),ac

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
  
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))#model.add(Conv2D(64,(3,3),ac

    model.add(Dropout(0.5))

    model.add(Dense(24,activation='softmax'))
  
    return model


images_train,labels_train=load_data("train/")
# images_train=np.array([image.flatten() for image in images_train])
labels_train=labenc.fit_transform(labels_train)
labels_train=onehotenc.fit_transform(labels_train.reshape(-1,1))

images_test,labels_test=load_data("test/")
print(images_train.shape,labels_train.shape,images_test.shape,labels_test.shape)
labels_test=labenc.fit_transform(labels_test)
labels_test=onehotenc.fit_transform(labels_test.reshape(-1,1))


slcnn = createCnn()
batch_size = 200
epochs = 10
slcnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = [ModelCheckpoint(filepath='models.hdf5')]

history = slcnn.fit(images_train, labels_train, batch_size=batch_size, epochs=epochs,callbacks=checkpoint, verbose=1,
                   validation_data=(images_test,labels_test))
  
slcnn.evaluate(images_test,labels_test)


	
def crhsv(img):
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	return hsv
	
def largestrect(contours):
	maxim=maxim2=mx2=my2=mw2=mh2=0
	mx=my=mw=mh=1
	for i in range(len(np.array(contours))):
		x,y,w,h = cv2.boundingRect(contours[i])
		if((w*h)>maxim):
			mx2=mx
			my2=my
			mw2=mw
			mh2=mh
			maxim=w*h
			mx=x
			my=y
			mw=w
			mh=h
	return mx,my,mw,mh  #(return  mx2,my2,mw2,mh2  for second largest rect)	
	

cap=cv2.VideoCapture(0) # mobile cam using droidcam
ret,back=cap.read()
back=back[0:400,40:400]
#xr=(ap.screen.size()[0]/x)
#yr=(ap.screen.size()[1]/y)
back=cv2.cvtColor(back,cv2.COLOR_BGR2GRAY)
#back=cv2.resize(back,None,fx=0.6,fy=0.6)
print(back.shape)
prev=back
curr1=back
pc=back
tc=back
t=back
t[:,:]=160
tc[:,:]=30
a=1   #0.7

ret,back=cap.read()
back=back[0:400,40:400]
#sback=skinback(back)
back=cv2.cvtColor(back,cv2.COLOR_BGR2GRAY)
hs,ss,vs=np.load("[h,s,v]histskin.npy")
sums=np.sum(hs)
hn,sn,vn=np.load("[h,s,v]histnonskin.npy")
sumn=np.sum(hn)
cv2.imshow("back",back)

while(1):
				k = cv2.waitKey(1) & 0xFF
				if k == 27:
					break
				ret,grdepc=cap.read()
				grdepc=grdepc[0:400,40:400]
				#grdepc=cv2.resize(grdepc,None,fx=0.6,fy=0.6)
				grdep=cv2.cvtColor(grdepc,cv2.COLOR_BGR2GRAY)
				
				curr2=grdep.copy()
				imgdiff1=grdep.copy()
				imgdiff2=grdep.copy()
				#hsvimg=crhsv(grdepc)
				cv2.absdiff(prev,curr1,imgdiff1)
				cv2.absdiff(prev,curr2,imgdiff2)
				curr2b=curr2.copy()
				prod=((imgdiff1>=8)*(imgdiff2>=8))*255
				prod=np.array(prod).astype("uint8")#----------image differencing(detect moving pixels)--------
				#cv2.imshow("product",prod)
				#ret,prod=cv2.threshold(prod,1,255,cv2.THRESH_BINARY_INV)
				#cv2.imshow("prod",prod)
				imgdiff,contours,hierarchy=cv2.findContours(prod,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

				mx,my,mw,mh=largestrect(contours)
				cv2.rectangle(grdepc,(mx,my),(mx+mw,my+mh),(0,255,0),2)
				#cv2.rectangle(grdepc,(mx2,my2),(mx2+mw2,my2+mh2),(0,255,0),2)

				backsub=curr2.copy()
				cv2.absdiff(back,curr2b,backsub)#--------------background sub------------------
				#back=((a*back)+((1-a)*curr2)).astype("uint8")#-------------<update>
				#back[my:my+mh,mx:mx+mw]=pc[my:my+mh,mx:mx+mw]
				#t=((a*t)+(5*(1-a)*backsub)).astype("uint8")
				#t[my:my+mh,mx:mx+mw]=tc[my:my+mh,mx:mx+mw]#--------------</update>
				#ret,backsub=cv2.threshold(backsub,30,255,cv2.THRESH_BINARY_INV)
				#currbacksub=((backsub<t)*255).astype("uint8")...
				#imgdiff,contours,hierarchy=cv2.findContours(currbacksub,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				#mx,my,mw,mh=largestrect(contours)
				#cv2.rectangle(grdepc,(mx,my),(mx+mw,my+mh),(0,0,255),2)
				#cv2.rectangle(backsub,(mx,my),(mx+mw,my+mh),(255),2)
				#xcoord=1023-(int((mx+mw)*xr)%1024)
				#ycoord=(int((my+mh)*yr)%768)
				#print(xcoord,ycoord)
				#ap.mouse.move(xcoord,ycoord)
				#cv2.imshow("backmodel",back)
				cv2.imshow("Motion Detection",grdepc)
				#white[:,:]=255
				for x in range(my,my+mh):@-----------------------------------
					for y in range(mx,mx+mw):
						ps=float(hs[hsvimg[x,y][0]]+ss[hsvimg[x,y][1]]+vs[hsvimg[x,y][2]])/(3*sums)
						pn=float(hn[hsvimg[x,y][0]]+sn[hsvimg[x,y][1]]+vn[hsvimg[x,y][2]])/(3*sumn)
						r=ps/pn
						#print(r)
						#print(ps,pn,ps/pn,int((float(255)/10)*r))
						if(r<=3):
							white[x,y]=255#int((float(255)/16)*r)
				cv2.imshow("Skin Color Detection",white)#--------------
				#cv2.waitKey(0)
				
				#pc=back.copy()#--------copy current values for future use------
				#tc=t.copy()
				prev=curr1.copy()
				curr1=curr2.copy()
				#ret,grdep=cv2.threshold(grdep,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				#grdep=crhsv(grdep)
				#gbdep = cv2.GaussianBlur(grdep,(2*m+1,2*m+1),1)
		#		ret,thresh=cv2.threshold(grdep,m,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
		#		cv2.imshow("threshold",thresh)
				#candep=cv2.Canny(gbdep,10+mi,70+ma,0)
				#cv2.imshow('Canny',candep)
