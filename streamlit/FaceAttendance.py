import cv2
import face_recognition

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path} / {cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)