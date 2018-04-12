import time
import os
import torch
import torchvision
from PIL import Image
import numpy as np

class PathError(BaseException):
    def __init__(self, mesg="raise a PathError"):
        print(mesg)

class FileError(BaseException):
    def __init__(self, mesg="raise a FileError"):
        print(mesg)

authorHeadStr = "Author: "
timeHeadStr = "Experimental Time: "
folderHeadStr = "Record Folder: "
describeHeadStr = "Describe: "
endTimeHeadStr = "End Time: "

# a base class, you should never use it
class record(object):
    def __init__(self, basePath, expName, author, describe, filename):
        if basePath[-1] != '/':
            basePath = basePath + '/'
        self.dirPath = str(basePath) + str(expName) + '/'
        self.recordTxtName = self.dirPath + filename
        self.author = str(author)
        self.time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.describe = str(describe)
        self.isOpen = False

    def open(self):
        self.recordTxt = open(self.recordTxtName, 'a')
        self.isOpen = True
        self.write_record_head()

    def add_message(self, message):
        assert type(message) == str
        if message[-1] != '\n':
            message = message + '\n'
        if self.isOpen:
            self.recordTxt.write(message)
        else:
            raise FileError('Please run record.open() to open the record file.')

    def write_record_head(self):
        self.add_message(authorHeadStr + self.author)
        self.add_message(timeHeadStr + self.time)
        self.add_message(folderHeadStr + self.dirPath)
        self.add_message(describeHeadStr + self.describe)
        self.add_message("\n")

    def close(self):
        self.add_message('\n')
        self.add_message(endTimeHeadStr + time.strftime("%H:%M:%S"))
        self.recordTxt.close()
        self.isOpen = False


saveImageHeadStr = "    [Save Image]  "

class experimentalRecord(record):
    def __init__(self, basePath='./',
                       expName='WHO KNOWS',
                       author='Bruce Wayne',
                       describe='',
                       fileName='Experimental Record.txt'):
        super(experimentalRecord, self).__init__(basePath, expName, author, describe, fileName)
        if not os.path.exists(self.dirPath):
            os.mkdir(self.dirPath)
        else:
            raise PathError('The folder is already exist.')

    def add_image(self, image, imageName,
                  message="This is a lazy person, leave no message for the image.",
                  mode="TORCH_GPU_TENSOR"):
        if mode == "TORCH_GPU_TENSOR":
            tensor = image.view(image.shape[1:])
            tensor = tensor.cpu()
            tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
            pil = tensor_to_pil(tensor)
        elif mode == "NP":
            image_uint8 = (np.squeeze(image).transpose(1, 2, 0) * 255).astype(np.uint8)
            pil = Image.fromarray(image_uint8, mode='RGB')

        filename = self.dirPath + imageName
        pil.save(filename)

        self.add_message(message)
        self.add_message(saveImageHeadStr + imageName)

class measureRecord(record):
    def __init__(self, basePath='./',
                       expName='WHO KNOWS',
                       author='Bruce Wayne',
                       describe='',
                       fileName='Measure Record.txt'):
        super(measureRecord, self).__init__(basePath, expName, author, describe, fileName)
        if not os.path.exists(self.dirPath):
            raise PathError('The folder is not exist.')
        if os.path.exists(self.recordTxtName):
            raise PathError('The file is already exist.')