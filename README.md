# Bag-Of-Word-PyQT-CBIR-system
This project an basic application of ontent base image retrival system and relative feedback system.
To use code you should download conda environment and use conda for dowloading external libraries.<br />
Program contains ;<br />
A gui (used with PyQT framework)  <br />
A classifier (SVM classifer used. The train and test code is inspired from this [github](https://github.com/bikz05/bag-of-words) 
user.<br />
A google image/url search service (URL for geting result of svm classifier from web. Images for relative feedback on second gui)<br />
A Second gui for relative feedback system (Used with PyQt framework)<br />
The usage of program details explained on [Wiki](https://github.com/Candan26/Bag-Of-Word-PyQT-CBIR-system/wiki/About-Code-Description) page of this repository 
The program Basicly has two Gui <br />
The first gui has  4 button which are <br />
Open Image : is using for selecting images which will predicted from pre-trained svm classifier and searched from google web services <br />
Open Retrived Images : opens Second gui for dowloaded images froom google image service in order to update train set efficiently <br />
Train Data Set: re-trains and saves the svm classifer. <br />
Close : Terminates program. <br />
![gui1](https://user-images.githubusercontent.com/21033733/71325912-9c43b000-2504-11ea-938a-0378b017b7b7.png)

The second gui has 2 button  and images for relative feedback ; <br />
Relative feed back images : is also button  which contains images. When user select an image program takes 
the path of image . The selected make those buttond dissapear for avoding multiple selection.  <br />
Update : moves selected images from temp folder to selected file location. After moving object renames all
directory according to ascending order. Thanks to that after update users can re-train data sets.  <br />
Cancel: Disposes the second gui.


![gui2](https://user-images.githubusercontent.com/21033733/71325922-ac5b8f80-2504-11ea-8d4c-cf4ae7d9b9ec.png)
