#import
import googlesearch
import os
import sip
import sys
import cv2 as cv
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt
import PyQt5 as qt
import PyQt5.QtCore
import PyQt5.QtWidgets as qwd
import re
import math
import shutil
import datetime
import inspect
####from
from google_images_download import google_images_download
from time import sleep
#from PyQt5.QtWebKit import *
#from PyQt5.QtWebKitWidgets import *
from PyQt5.QtNetwork import *
from PyQt5 import QtGui as gui
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import *
from pathlib2 import Path

##MACRO DEFINITIONS 
WINDOW_TITLE="INF501 Term Project"
WINDOW_SIZE_X=1280#1920
WINDOW_SIZE_Y=720#1080
WINDOW_SMALL_TITLE="Select Recursive Feedback"
QDIALOG_IMAGE_PREFIX_TYPE = "*.png *.jpg *.jpeg *.JPG"
QDIALOG_IAMGE_SELECTION_DIALOG_NAME="Select image For SVN"
SVM_TRAINED_FILE_LOCATION="bof.pkl"
SVM_FEATURE_DETECTOR_EXTRACTOR_TYPE="SIFT"
SVM_TRAIN_PATH="/home/candan/Desktop/tp4_ws/Train"
SVM_TEST_PATH="/home/candan/Desktop/tp4_ws/Test"
SVM_RETRIVED_RELATIVE_IMAGES_PATH="/home/candan/Desktop/tp4_ws/RetrivedImages/"
GOOGLE_GIMAGE_OUTPUT_PATH="/home/candan/Desktop/tp4_ws/RetrivedImages/
GOOGLE_GIMAGE_COUNT=2
GOOGLE_URL_COUNT=4
searchList=["accordion" ,"airplanes" ,"anchor" ,"ant" ,"barrel" ,"bass" ,"beaver" ,"binocular" ,"bonsai" \
            ,"brain" ,"brontosaurus" ,"buddha" ,"butterfly" ,"camera" ,"cannon" ,"car_side" ,"ceiling_fan" \
            ,"cellphone" ,"chair" ,"chandelier" ,"cougar_body" ,"cougar_face" ,"crab" ,"crayfish" ,"crocodile" \
            ,"crocodile_head" ,"cup" ,"dalmatian" ,"dollar_bill" ,"dolphin" ,"dragonfly" ,"electric_guitar" \
            ,"elephant" ,"emu" ,"euphonium" ,"ewer" ,"Faces" ,"Faces_easy" ,"ferry" ,"flamingo" ,"flamingo_head" \
            ,"garfield" ,"gerenuk" ,"gramophone" ,"grand_piano" ,"hawksbill" ,"headphone" ,"hedgehog" \
            ,"helicopter" ,"ibis" ,"inline_skate" ,"joshua_tree" ,"kangaroo" ,"ketch" ,"lamp" ,"laptop" \
            ,"Leopards" ,"llama" ,"lobster" ,"lotus" ,"mandolin" ,"mayfly" ,"menorah" ,"metronome" \
            ,"minaret" ,"Motorbikes" ,"nautilus" ,"octopus" ,"okapi" ,"pagoda" ,"panda" ,"pigeon" ,"pizza" \
            ,"platypus" ,"pyramid" ,"revolver" ,"rhino" ,"rooster" ,"saxophone" ,"schooner" ,"scissors" \
            ,"scorpion" ,"sea_horse" ,"snoopy" ,"soccer_ball" ,"stapler" ,"starfish" ,"stegosaurus" ,"stop_sign" \
            ,"strawberry" ,"sunflower" ,"tick" ,"trilobite" ,"umbrella" ,"watch" ,"water_lilly" ,"wheelchair" \
            ,"wild_cat" ,"windsor_chair" ,"wrench", "yin_yang"]
numberOfColumnInSecondUI=4
SecondUIButtonNames = []
SECOND_GUI_ICON_NAME_PATH="align.png"

class PicButton(qwd.QAbstractButton):
    def __init__(self, pixmap, parent=None):
        super(PicButton, self).__init__(parent)
        self.pixmap = pixmap

    def paintEvent(self, event):
        painter = gui.QPainter(self)
        painter.drawPixmap(event.rect(), self.pixmap)

    def sizeHint(self):
        return self.pixmap.size()

class QGuiSmall(qwd.QMainWindow):
    def __init__(self, parent=None):
        super(QGuiSmall,self).__init__(parent)
        self.setWindowTitle(WINDOW_SMALL_TITLE)
        self.setGeometry(250, 250, 640, 480)
        self.setWindowIcon(gui.QIcon(SECOND_GUI_ICON_NAME_PATH))
        self.InitSmallUI()

    def InitSmallUI(self):
        self.SetSecondGuiLayout()      
        self.CheckButtonNameIteration() #check more than one press 
        self.SetImageButtonLayout()
        self.InitiateImageButtons()

    def SetSecondGuiLayout(self):
        self.grid = qwd.QGridLayout()
        self.widget= qwd.QWidget()
        self.widget.setLayout(self.grid)
        self.scroll = qwd.QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)
        self.setCentralWidget(self.scroll)

    def CheckButtonNameIteration(self):
        logData("Correcting the button name list and button status ")
        for i, secondUIButtonName in enumerate(SecondUIButtonNames):
            if secondUIButtonName =='update':
                SecondUIButtonNames.remove(secondUIButtonName)
            if secondUIButtonName == 'cancel':
                SecondUIButtonNames.remove(secondUIButtonName)
        if SecondUIButtonNames.__len__()%numberOfColumnInSecondUI ==0:
            SecondUIButtonNames.append('') ## to aviod deleting last existing two images
        while SecondUIButtonNames.__len__()%numberOfColumnInSecondUI != 0:
            SecondUIButtonNames.append('')
        if  SecondUIButtonNames[SecondUIButtonNames.__len__()-2] != 'update':
            SecondUIButtonNames[SecondUIButtonNames.__len__()-2]='update'##adjust last two element with update and cancel
        if SecondUIButtonNames[SecondUIButtonNames.__len__()-1] != 'update':
            SecondUIButtonNames[SecondUIButtonNames.__len__()-1]='cancel'
        logData("The number of ["+str(SecondUIButtonNames.__len__())+"] will be displayed on screen")

    def SetImageButtonLayout(self):
        self.buttonList={}
        self.sequentialObjectNameList={}
        positions = [(i,j) for i in range( SecondUIButtonNames.__len__()/numberOfColumnInSecondUI) for j in range(numberOfColumnInSecondUI)]
        for position, name in zip(positions, SecondUIButtonNames):
            if name == '':
                continue
            if name =='update' or name == 'cancel':
                button= qwd.QPushButton(name)                
            else:
                button= PicButton(gui.QPixmap(name),self)
                self.sequentialObjectNameList[name]=str(button.objectName)
            self.buttonList[name]=button
            self.grid.addWidget(button,*position)

    def InitiateImageButtons(self):
        self.imageLocationList=[]
        self.buttonImageList={}
        for key in self.buttonList:
            if key =="cancel":
                btnCancel = self.buttonList.get(key)
            elif key=="update":
                btnUpdate= self.buttonList.get(key)
            else:
                btnSelect=self.buttonList.get(key)
                btnSelect.clicked.connect(self.SetButtonImage)
                logData("Setting buttonImage with name "+ str(key))
                self.buttonImageList[btnSelect.objectName]=key               
        btnCancel.clicked.connect(self.BtnCloseSystem)
        logData("Setting buttonImage with name cancel")
        btnUpdate.clicked.connect(self.BtnUpdateSystem)
        logData("Setting buttonImage with name update")
 
    def SetImageFolderName(self,name):
        self.glbFolderName=name

    def BtnUpdateSystem(self):
        try:   
            logData("moving data from source file to destination file")
            for imageLocation in self.imageLocationList:
                moveAFile(imageLocation,self.glbFolderName)#"C:/Users/EXT02D17919/Desktop/python/temp"
            logData("renaming the train set for next train iteration")       
            organizeFile(self.glbFolderName+"/",".jpg",'image_')#"C:/Users/EXT02D17919/Desktop/python/temp/"
            del self.imageLocationList[:]
            self.destroy()
            logData("renaming complete, object destorying for next itartion")
        except Exception as ex:
            logData("Exception on "+ str(ex))

    def BtnCloseSystem(self):
        logData("desposing second gui")
        self.destroy()

    def SetButtonImage(self):
        sending_button = self.sender()
        logData("the object name of pressed button " + str(sending_button.objectName))
        self.imageLocationList.append(self.buttonImageList.get(sending_button.objectName))
        sending_button.setVisible(False)

class QGui(qwd.QDialog):
    def __init__(self, parent=None):
        super(QGui, self).__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(0, 0, WINDOW_SIZE_X, WINDOW_SIZE_Y)
        self.InitUI()

    def InitUI(self):
        ####Timer Functions
        ####Label OpenImg 
        logData("creating main UI") 
        self.lblOpenImg = qwd.QLabel('Open Img', self)
        self.lblOpenImg.setEnabled(True)
        self.lblOpenImg.setGeometry(20, 100, 500, 300)
        self.lblOpenImg.setSizePolicy(qwd.QSizePolicy.Ignored, qwd.QSizePolicy.Ignored)
        self.lblOpenImg.setScaledContents(True)
        self.lblOpenImg.setAutoFillBackground(True)
        self.lblOpenImg.setFrameShape(qwd.QFrame.Panel)
        self.lblOpenImg.setFrameShadow(qwd.QFrame.Plain)
        self.lblOpenImg.setLineWidth(2)
        ####Text Box Definitions
        ##text system status
        self.txtSystemStatus = qwd.QTextBrowser(self)
        self.txtSystemStatus.setGeometry(500, 200, 400, 300)
        self.txtSystemStatus.setOpenExternalLinks(True)
        self.txtSystemStatus.setAcceptRichText(True)
        self.txtSystemStatus.setMouseTracking(True)
        ##global txtSystemStatus
        ####Button Definitions
        ##Button Close
        btnClose = qwd.QPushButton('Close', self)
        btnClose.setToolTip("Close Program")
        btnClose.clicked.connect(self.BtnCloseSystem)
        ##button open image
        btnOpenImage = qwd.QPushButton("Open Image", self)
        btnOpenImage.setToolTip("Open image for signal processing")
        btnOpenImage.clicked.connect(self.BtnOpenImage)
        ##button open second guis 
        btnOpenSecondGui = qwd.QPushButton("Open Retrived Images",self)
        btnOpenSecondGui.setToolTip("This button opens the images from google web services")
        btnOpenSecondGui.clicked.connect(self.BtnOpenNewGui)
        ##button for traning data set 
        btnTrainDataSet= qwd.QPushButton("Train Data Set",self)
        btnTrainDataSet.setToolTip("This is for re-training data set")
        btnTrainDataSet.clicked.connect(self.BtnTrainDataSet)
        ####layouts
        lytQHBtns=qwd.QHBoxLayout() 
        lytQHBtns.addWidget(btnOpenImage)
        lytQHBtns.addWidget(btnOpenSecondGui)
        lytQHBtns.addWidget(btnTrainDataSet)
        lytQHBtns.addWidget(btnClose)
        ##Box Layout
        #system status
        lytQBoxTxt=qwd.QBoxLayout(qwd.QBoxLayout.LeftToRight, parent=None)
        lytQBoxTxt.addWidget(self.txtSystemStatus)
        lytQBoxTxt.setGeometry(QtCore.QRect(200, 200, 100, 300))
        #image
        lytQBoxImg=qwd.QBoxLayout(qwd.QBoxLayout.LeftToRight, parent=None)
        lytQBoxImg.addWidget(self.lblOpenImg)
        lytQVImg=qwd.QVBoxLayout()
        lytQVImg.addLayout(lytQBoxImg)
        ##Line Edit + Lbl Layout
        lytQVLnedsAndLabels = qwd.QVBoxLayout()
        ####Grid Layout
        lytQGridMain = qwd.QGridLayout()
        lytQGridMain.addLayout(lytQVImg, 1, 0, 6, 1)
        lytQGridMain.addLayout(lytQVLnedsAndLabels, 1, 0)
        lytQGridMain.addLayout(lytQBoxTxt, 8, 0, 6, 1)
        lytQGridMain.addLayout(lytQHBtns, 7, 0)
        self.setLayout(lytQGridMain)

    def BtnOpenNewGui(self):
        global GUI2
        if self.isSystemSearched==False:
            self.txtSystemStatus.append("Please first press open image button adn scan the image ")
        else :
            self.txtSystemStatus.append("new ui is creating ")
            logData("Creating new gui for selectin relative feedback")
            GUI2=QGuiSmall()
            GUI2.SetImageFolderName(self.folderName)
            GUI2.show()
    
    def BtnTrainDataSet(self):
        logData("pressing train data set button ,begining to train data set.")
        trainDataSet()

    def BtnCloseSystem(self):
        sys.exit()

    def BtnOpenImage(self):
        fileName=qwd.QFileDialog.getOpenFileNames(self,QDIALOG_IAMGE_SELECTION_DIALOG_NAME,"",QDIALOG_IMAGE_PREFIX_TYPE)        
        self.folderName=os.path.dirname(str(fileName[0]).split("'",1)[1])
        if not fileName:
            self.txtSystemStatus.setText("Please Select a jpg or png file")
        else:  
            self.txtSystemStatus.setText("The file path " + str(fileName[0]) + " is selected ")
            logData("the file path "+ str(fileName[0]) + " is selected ")
            self.txtOpenFilePath = fileName[0]
            self.GetAndSetImg(fileName[0])

    def GetAndSetImg(self,file_name):
        try:        
            self.matImg= cv.imread(file_name[0])
            self.image_path=file_name[0]
            self.image_paths_len=3
            #self.trainData(matImg)
            svmPrediction = self.CheckSVM(self.matImg)
            self.txtSystemStatus.append("svm prediction number is "+str(svmPrediction)+ " and name is "+ searchList[svmPrediction])
            logData("svm prediction " + str(searchList[svmPrediction]))
            self.GImageSearch(searchList[svmPrediction],str(file_name[0]))#"CAR"
            self.GSearch(searchList[svmPrediction])#searchList[svmPrediction]
            organizeFile(GOOGLE_GIMAGE_OUTPUT_PATH+searchList[svmPrediction]+"/",".jpg",'temp_')
            setForRetrivalFileNames( GOOGLE_GIMAGE_OUTPUT_PATH+searchList[svmPrediction],svmPrediction)
            self.SetSystemSearchedFlag(True)
            matRGBImg=cv.cvtColor(self.matImg, cv.COLOR_BGR2RGB)
            Qimg = gui.QImage(matRGBImg.data, matRGBImg.shape[1], matRGBImg.shape[0], gui.QImage.Format_RGB888)
            pixelImg = gui.QPixmap.fromImage(Qimg)
            self.lblOpenImg.setPixmap(pixelImg)
        except Exception as ex:
            logData("Exception on "+ str(ex))
            self.txtSystemStatus.append("Exception on "+ str(ex))
            
    def SetSystemSearchedFlag(self,flag):
        self.isSystemSearched=flag

    def CheckSVM(self,img):
        logData("checking image on svm classifier")
        clf, classes_names, stdSlr, k, voc = joblib.load(SVM_TRAINED_FILE_LOCATION)
        fea_det = cv.FeatureDetector_create(SVM_FEATURE_DETECTOR_EXTRACTOR_TYPE)
        des_ext = cv.DescriptorExtractor_create(SVM_FEATURE_DETECTOR_EXTRACTOR_TYPE) 
        des_list = []
        kpts = fea_det.detect(img)
        kpts, des = des_ext.compute(img, kpts)
        des_list.append((self.image_path,des))
        # Stack all the descriptors vertically in a numpy array
        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[0:]:
            descriptors = np.vstack((descriptors, descriptor))
        # 
        test_features = np.zeros((self.image_paths_len, k), "float32")
        for i in xrange(1):
            words, distance = vq(des_list[i][1],voc)
            for w in words:
                test_features[i][w] += 1
        # Perform Tf-Idf vectorization
        nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*self.image_paths_len+1) / (1.0*nbr_occurences + 1)), 'float32')
        # Scale the features
        test_features = stdSlr.transform(test_features)
        # Perform the predictions
        predictions =  [classes_names[i] for i in clf.predict(test_features)]
        predictionForReturn  = clf.predict(test_features)
        for prediction in predictions:
            logData ("SVM PREDICTIONS ARE ["+ prediction+"] sending result as is "+ str(predictionForReturn[0]))
        return predictionForReturn[0]

    def GSearch(self, name):
        count=0
        try:
            from googlesearch import search
            self.txtSystemStatus.append("The founded urls are listed")
            for i in search(query=name, stop=GOOGLE_URL_COUNT):
                count += 1
                self.txtSystemStatus.append(str(count)+" <a >"+i+"</a>")
            logData("The first ["+str(count)+"] data will searching with name of  ["+name+"] on google.com")
        except ImportError:
            logData("No Module named 'google' Found")

    def GImageSearch(self, query,image_dir):
        logData("The first ["+str(GOOGLE_GIMAGE_COUNT)+"] data will searching with name of  ["+query+"] on googleImage.com")
        logData("image dir "+ image_dir + " will be used for images")
        response =  google_images_download.googleimagesdownload()
        arguments = {"keywords":query,
                     "format":"jpg",
                     "limit":GOOGLE_GIMAGE_COUNT, 
                     "print_urls":False,
                     "size": "medium",
                     "output_directory":GOOGLE_GIMAGE_OUTPUT_PATH}
        try:
            response.download(arguments)
        except:
            logData("error on search")
                
def trainDataSet():
    try:
        train_path=SVM_TRAIN_PATH
        training_names = os.listdir(train_path)
        image_classes = []
        image_paths = []
        class_id = 0
        loopCounter=0
        loopCounterForPrint=0

        logData(" getting names of classes")
        for training_name in training_names:
            dir = os.path.join(train_path, training_name)
            class_path = [os.path.join(dir, f) for f in os.listdir(dir)]
            image_paths+=class_path
            image_classes += [class_id]*len(class_path)
            class_id+=1
        logData("creating feature extractions with method "+ SVM_FEATURE_DETECTOR_EXTRACTOR_TYPE)
        # Create feature extraction and keypoint detector objects
        fea_det = cv.FeatureDetector_create(SVM_FEATURE_DETECTOR_EXTRACTOR_TYPE)
        des_ext = cv.DescriptorExtractor_create(SVM_FEATURE_DETECTOR_EXTRACTOR_TYPE)
        logData("starting to extract features from each images")
        # List where all the descriptors are stored
        des_list = []
        for image_path in image_paths:
            im= cv.imread(image_path)
            kpts = fea_det.detect(im)
            kpts, des = des_ext.compute(im, kpts)
            des_list.append((image_path, des))
            loopCounter=loopCounter+1
            if loopCounter%100==0:
                loopCounter=0
                loopCounterForPrint=loopCounterForPrint+1
                logData("extracting  features continues with iteration "+ str(loopCounterForPrint))
        logData("setting descriptors values from extracted feautures")
        # Stack all the descriptors vertically in a numpy array
        loopCounter=0
        loopCounterForPrint=0
        descriptors = des_list[0][1]    
        for image_path, descriptor in des_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))
            loopCounter=loopCounter+1
            if loopCounter%100==0:
                loopCounter=0
                loopCounterForPrint=loopCounterForPrint+1
                logData("setting descriptors continues with iteration "+ str(loopCounterForPrint))

        # Perform k-means clustering
        logData("calculation kmeans clustring")
        k = 100
        voc, variance = kmeans(descriptors, k, 1)
        # Calculate the histogram of features
        logData("calculating the histogram of features")
        loopCounter=0
        loopCounterForPrint=0
        im_features = np.zeros((len(image_paths), k), "float32")
        for i in xrange(len(image_paths)):
            words, distance = vq(des_list[i][1],voc)
            for w in words:
                im_features[i][w] += 1
            loopCounter=loopCounter+1
            if loopCounter%100==0:
                loopCounter=0
                loopCounterForPrint=loopCounterForPrint+1
                logData("calculating histogram feautres continues with iteration "+ str(loopCounterForPrint))
        logData("performing TF-IDF  vectorization")
        # Perform Tf-Idf vectorization
        nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
        # Scaling the words
        stdSlr = StandardScaler().fit(im_features)
        im_features = stdSlr.transform(im_features)
        # Train the Linear SVM
        logData("training Linear SVM")
        clf = LinearSVC()
        clf.fit(im_features, np.array(image_classes))
        # Save the SVM
        logData("saving values on with names " + SVM_TRAINED_FILE_LOCATION)
        joblib.dump((clf, training_names, stdSlr, k, voc), SVM_TRAINED_FILE_LOCATION, compress=3)
    except Exception as ex:
        logData("Exception on "+ str(ex))
     
def testDataSet():
    try:
        svmRateOfSuccessList ={}
        test_path=SVM_TEST_PATH
        # Load the classifier, class names, scaler, number of clusters and vocabulary 
        clf, classes_names, stdSlr, k, voc = joblib.load(SVM_TRAINED_FILE_LOCATION)
        testing_names = os.listdir(test_path)
        image_paths = []
        numberOfLocalImages=0
        counterOfLocalImages=0
        logData("creating feature extractions with method "+ SVM_FEATURE_DETECTOR_EXTRACTOR_TYPE)
        # Create feature extraction and keypoint detector objects
        fea_det = cv.FeatureDetector_create(SVM_FEATURE_DETECTOR_EXTRACTOR_TYPE)
        des_ext = cv.DescriptorExtractor_create(SVM_FEATURE_DETECTOR_EXTRACTOR_TYPE)
        
        logData(" getting names of classes")
        for i, testing_name in enumerate(testing_names):            
            try:
                dir = os.path.join(test_path, testing_name)
                class_path = [os.path.join(dir, f) for f in os.listdir(dir)]
                image_paths=class_path#image_paths+=class_path
                numberOfLocalImages=class_path.__len__()
                logData("the class name ["+ str(testing_names[i])+"] has ["+str(numberOfLocalImages)+"] images " )
                logData("starting to extract features from each images") 
                # List where all the descriptors are stored
                des_list = []
                for image_path in image_paths:
                    im = cv.imread(image_path)
                    kpts = fea_det.detect(im)
                    kpts, des = des_ext.compute(im, kpts)
                    des_list.append((image_path, des))

                # Stack all the descriptors vertically in a numpy array
                logData("setting descriptors values from extracted feautures")

                descriptors = des_list[0][1]
                for image_path, descriptor in des_list[0:]:
                    descriptors = np.vstack((descriptors, descriptor))
                # 
                logData("calculating the histogram of features")

                test_features = np.zeros((len(image_paths), k), "float32")
                for i in xrange(len(image_paths)):
                    words, distance = vq(des_list[i][1],voc)
                    for w in words:
                        test_features[i][w] += 1           
                logData("performing TF-IDF  vectorization")
                # Perform Tf-Idf vectorization
                nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
                idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
                # Scale the features
                logData("testing the predictions")
                test_features = stdSlr.transform(test_features)
                # Perform the predictions
                predictions=[]
                predictions =  [classes_names[i] for i in clf.predict(test_features)]
                nummberOfOccurrencesOfPrediction = predictions.count(str(testing_names[counterOfLocalImages]))
                rateOfSuccess=      (nummberOfOccurrencesOfPrediction*100)/ predictions.__len__()               
                prediction  = clf.predict(test_features)
                logData("the succes rate of  ["+str(testing_names[counterOfLocalImages]) +"] is : [%"+ str(rateOfSuccess) +\
                    "] of total number of ["+str(predictions.__len__())+"] data with tp ["+ str(nummberOfOccurrencesOfPrediction)+"]")
                logData("")
                svmRateOfSuccessList[testing_names[counterOfLocalImages]]=rateOfSuccess
                #logData("Prediction results "+ str(clf.predict(test_features)))
                #logData("the predictions :"+ str(predictions))
                counterOfLocalImages=counterOfLocalImages+1
            except Exception as ex:
                logData("Exception on "+ str(ex))
                counterOfLocalImages=counterOfLocalImages+1  
        plotResultOfSVM(svmRateOfSuccessList)
    except Exception as ex :
        logData("Exception on "+ str(ex))

def plotResultOfSVM(labelAndValues):
    label=[]
    values=[]
    for labelAndValue in labelAndValues:
        label.append(labelAndValue)
        values.append(labelAndValues.get(labelAndValue))
    logData("label values"+ str(label)+" , log values "+str(values))
    index = np.arange(len(label))
    plt.bar(index, values)
    plt.xlabel('classs Types', fontsize=15)
    plt.ylabel('Rates of percentage', fontsize=15)
    plt.xticks(index, label, fontsize=8, rotation=45, horizontalalignment='right')
    plt.title('Success rate of SVM classifier')  
    plt.show()

def setForRetrivalFileNames(src, svmPrediction):
    try:
        for i , filename in enumerate(os.listdir(src)):
            tmpImageOrj= cv.imread(str(GOOGLE_GIMAGE_OUTPUT_PATH+searchList[svmPrediction]+"/"+filename))
            imgResized = cv.resize(tmpImageOrj,(240,320))
            cv.imwrite(str(GOOGLE_GIMAGE_OUTPUT_PATH+searchList[svmPrediction]+"/"+filename),imgResized)
            SecondUIButtonNames.append( GOOGLE_GIMAGE_OUTPUT_PATH+searchList[svmPrediction]+"/"+filename)
            logData("existing file names "+ filename)
    except Exception as ex:
        logData("Exception on "+ str(ex))

def moveAFile(src,dst):
    try:
       shutil.move(src,dst)
    except Exception as ex:
        logData("Exception on " +str(ex))

def organizeFile(src, ext,prefix):
    try:
        _src = src
        _ext = ext
        #endsWithNumber =  re.compile(r'(\d+)'+(re.escape(_ext))+'$')
        fileCounter=0
        for i , filename in enumerate(os.listdir(_src)):
            fileCounter=i
        fileCounter=int(math.floor(math.log10(fileCounter))+1)  
        logData("file Counter ["+ str(fileCounter)+"]")
        if fileCounter <= 3:
            fileCounter=3    
        for i , filename in enumerate(os.listdir(_src)):
            if filename.endswith(_ext) or filename.endswith(".jpeg"):
                src=_src+ prefix + str(i).zfill(fileCounter)+".jpg"
                if (_src+filename) != src:
                    os.rename(_src+filename, src)
    except Exception as ex:
        logData("Exception on "+ str(ex))

def removeAFile(src):
    for filename in os.listdir(src):
        file_path = os.path.join(src, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logData('Failed to delete %s. Reason: %s' % (file_path, e))   

def logData(data):
    print  "["+ str(datetime.datetime.now())+"]  " + data
    
def main():
    cv.__version__
    app=qwd.QApplication(sys.argv)
    ## main gui
    ex=QGui()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    #organizeFile( SVM_RETRIVED_RELATIVE_IMAGES_PATH,".jpeg",'image_')
    #trainDataSet()
    #testDataSet()
    main()