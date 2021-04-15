import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import math
import random
import sys
from timesformer_pytorch import TimeSformer


# In[217]:


#ucfPath = 'D:/Files/Datasets/UCF-101'
ucfSplitNumber = 1
ucfPath = 'UCF-101'
framesPerVideo = 8
maxVideoPerClass = 100
maxClasses = 3
valPerClass = 5
numEpochs = 80
embeddingsSize = 256
batchSize = 8


# In[218]:


class AdvancedTimeSformer(TimeSformer):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        num_classes,
        image_width = 320,
        image_height = 240,
        patch_size = 16,
        channels = 3,
        depth = 12,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__(dim = dim, 
                         num_frames = num_frames, 
                         num_classes = num_classes, 
                         image_width = image_width, 
                         image_height = image_height,
                         patch_size = patch_size,
                         channels = channels, 
                         depth = depth,
                         heads = heads,
                         dim_head = dim_head,
                         attn_dropout = attn_dropout,
                         ff_dropout = ff_dropout)
        self.to_out = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, num_classes),
            #torch.nn.Softmax(num_classes)
        )


# In[219]:


class DataStorage():
    def __init__(self, ucfDataPath, framesPerVideo, ucfSplitNumber = 1, maxVideoPerClass = None, maxClasses = None):
        
        ucfFullSize = 13320
        self.trainLabelsNames = {}
        lastTrainIndex = 0
        lastTestIndex = 0
        
        if maxVideoPerClass is None:
            self.trainData = np.zeros((ucfFullSize - 101 * valPerClass, framesPerVideo, 3, 240, 320), dtype=np.uint8)
            self.trainLabels = np.zeros((ucfFullSize - 101 * valPerClass), dtype=int)
            self.testData = np.zeros((101 * valPerClass, framesPerVideo, 3, 240, 320), dtype=np.uint8)
            self.testLabels = np.zeros((101 * valPerClass), dtype=int)
        else:
            if maxClasses is None:
                self.trainData = np.zeros((101 * (maxVideoPerClass - valPerClass), framesPerVideo, 3, 240, 320), dtype=np.uint8)
                self.trainLabels = np.zeros((101 * (maxVideoPerClass - valPerClass)), dtype=int)
                self.testData = np.zeros((101 * valPerClass, framesPerVideo, 3, 240, 320), dtype=np.uint8)
                self.testLabels = np.zeros((101 * valPerClass), dtype=int)
            else:
                self.trainData = np.zeros((maxClasses * (maxVideoPerClass - valPerClass), framesPerVideo, 3, 240, 320), dtype=np.uint8)
                self.trainLabels = np.zeros((maxClasses * (maxVideoPerClass - valPerClass)), dtype=int)
                self.testData = np.zeros((maxClasses * valPerClass, framesPerVideo, 3, 240, 320), dtype=np.uint8)
                self.testLabels = np.zeros((maxClasses * valPerClass), dtype=int)                
        
        for k, classFolderName in enumerate(sorted(os.listdir(ucfPath))):
            if maxClasses is not None and k >= maxClasses:
                break
            
            print('Process class ' + classFolderName)
            self.trainLabelsNames[classFolderName] = k
            for i, videoName in enumerate(sorted(os.listdir(os.path.join(ucfPath, classFolderName)))):
                if maxVideoPerClass is not None and i >= maxVideoPerClass:
                    break

                if i < valPerClass:
                    self.testLabels[lastTestIndex] = k
                else:
                    self.trainLabels[lastTrainIndex] = k

                count = 0
                video = cv2.VideoCapture(os.path.join(ucfPath, classFolderName, videoName))
                numberOfFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

                for j in range(framesPerVideo):
                    video.set(cv2.CAP_PROP_POS_FRAMES, count)
                    success, image = video.read()
                    if success:
                        if image.shape != (240, 320, 3):
                            image = cv2.resize(image, (320, 240))
                        if i < valPerClass:
                            self.testData[lastTestIndex][j] = np.swapaxes(
                                                np.swapaxes(image, 
                                                    0, 2),
                                                1, 2)
                        else:
                            self.trainData[lastTrainIndex][j] = np.swapaxes(
                                                np.swapaxes(image, 
                                                    0, 2),
                                                1, 2)
                    count += numberOfFrames // framesPerVideo
                    
                if i < valPerClass:
                    lastTestIndex += 1
                else:
                    lastTrainIndex += 1
            
        assert lastTrainIndex == self.trainData.shape[0], "Error in train data length"
        assert lastTestIndex == self.testData.shape[0], "Error in test data length"
        


# In[221]:


tsf_dim = 512
tsf_patch_size = 16
tsf_depth = 12
tsf_heads = 8
tsf_dim_head = 64


# In[222]:


if __name__ == '__main__':
    if len(sys.argv) > 1:
        tsf_dim = int(sys.argv[1])
        tsf_patch_size = int(sys.argv[2])
        tsf_depth = int(sys.argv[3])
        tsf_heads = int(sys.argv[4])
        tsf_dim_head = int(sys.argv[5])


# In[223]:


dataStorage = DataStorage(ucfPath, framesPerVideo)


# In[224]:
'''

model = AdvancedTimeSformer(
    dim = tsf_dim,
    image_width = 320,
    image_height = 240,
    patch_size = tsf_patch_size,
    num_frames = framesPerVideo,
    num_classes = 101,
    depth = tsf_depth,
    heads = tsf_heads,
    dim_head = tsf_dim_head,
    attn_dropout = 0.1,
    ff_dropout = 0.1
)


# In[ ]:


print((dataStorage.trainData).shape)
print((dataStorage.testData).shape)


# In[225]:

# In[226]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.device_count())

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

# In[227]:

model.to(device)

lossFunc = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# In[228]:


print('Learnable params: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

trainAccs = []
trainLosses = []
valAccs = []
valLosses = []

for epoch in range(numEpochs):  # loop over the dataset multiple times
    start_time = time.time()
    indices = [i for i in range(len(dataStorage.trainData))]
    random.shuffle(indices)
    
    #Train
    print('Start training')
    model.train()
    train_correct = 0
    train_loss = 0.0
    for batchNumber in range(len(dataStorage.trainData) // batchSize):
        inputs = torch.tensor([dataStorage.trainData[i] for i in indices[batchNumber * batchSize : (batchNumber + 1) * batchSize]], dtype = torch.float32)
        labels = torch.tensor([dataStorage.trainLabels[i] for i in indices[batchNumber * batchSize : (batchNumber + 1) * batchSize]], dtype = torch.long)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = lossFunc(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        
        del inputs
        del labels
        del outputs

    indices = [i for i in range(len(dataStorage.testData))]
    random.shuffle(indices)
    
    print('Epoch: %d Batches: %d Train loss: %.3f Train acc: %.3f' %
          (epoch + 1, len(dataStorage.trainData) // batchSize, train_loss / len(dataStorage.trainData), train_correct / len(dataStorage.trainData)))
    trainAccs.append(train_correct / len(dataStorage.trainData))
    trainLosses.append(train_loss / len(dataStorage.trainData))

    #Validation
    print('Start validation')
    model.train(False)
    val_correct = 0
    val_loss = 0.0
    for batchNumber in range(len(dataStorage.testData) // batchSize):
        inputs = torch.tensor([dataStorage.testData[i] for i in indices[batchNumber * batchSize : (batchNumber + 1) * batchSize]], dtype = torch.float)
        labels = torch.tensor([dataStorage.testLabels[i] for i in indices[batchNumber * batchSize : (batchNumber + 1) * batchSize]], dtype = torch.long)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = lossFunc(outputs, labels)

        val_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        val_correct += (predicted == labels).sum().item()
                
        del inputs
        del labels
        del outputs
    
    print('Epoch: %d Val loss: %.3f Val acc: %.3f' %
          (epoch + 1, val_loss / len(dataStorage.testData), val_correct / len(dataStorage.testData)))
    valAccs.append(val_correct / len(dataStorage.testData))
    valLosses.append(val_loss / len(dataStorage.testData))
    
    print("%s seconds for epoch" % (time.time() - start_time))

print('Finished Training')

torch.save(model.state_dict(), "timesformerOnUCF")'''


# In[234]:
model = AdvancedTimeSformer(
    dim = tsf_dim,
    image_width = 320,
    image_height = 240,
    patch_size = tsf_patch_size,
    num_frames = framesPerVideo,
    num_classes = embeddingsSize,
    depth = tsf_depth,
    heads = tsf_heads,
    dim_head = tsf_dim_head,
    attn_dropout = 0.1,
    ff_dropout = 0.1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.device_count())

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
model.to(device)

lossFunc = torch.nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters())

def getTripletsForBatch(batchIndices, isTrain = True):
    posClassIndices = []
    for b in batchIndices:
        if isTrain:
            index = random.randint(0, len(dataStorage.trainDict[dataStorage.trainLabels[b]]) - 2)
            elem = dataStorage.trainDict[dataStorage.trainLabels[b]][index]
            if elem >= b:
                elem = dataStorage.trainDict[dataStorage.trainLabels[b]][index + 1]
            posClassIndices.append(elem)
        else:
            index = random.randint(0, len(dataStorage.testDict[dataStorage.testLabels[b]]) - 2)
            elem = dataStorage.testDict[dataStorage.testLabels[b]][index]
            if elem >= b:
                elem = dataStorage.testDict[dataStorage.testLabels[b]][index + 1]
            posClassIndices.append(elem)
    negClassIndices = []
    for b in batchIndices:
        if isTrain:
            classNum = random.randint(0, len(dataStorage.trainDict.keys()) - 2)
            if classNum >= dataStorage.trainLabels[b]:
                classNum += 1
            index = random.randint(0, len(dataStorage.trainDict[classNum]) - 1)
            negClassIndices.append(dataStorage.trainDict[classNum][index])
        else:
            classNum = random.randint(0, len(dataStorage.testDict.keys()) - 2)
            if classNum >= dataStorage.testLabels[b]:
                classNum += 1
            index = random.randint(0, len(dataStorage.testDict[classNum]) - 1)
            negClassIndices.append(dataStorage.testDict[classNum][index])
            
    return batchIndices + posClassIndices + negClassIndices

print('Learnable params: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

trainAccs = []
trainLosses = []
valAccs = []
valLosses = []

for epoch in range(numEpochs):  # loop over the dataset multiple times
    start_time = time.time()
    indices = [i for i in range(len(dataStorage.trainData))]
    random.shuffle(indices)
    
    #Train
    print('Start training')
    model.train()
    train_correct = 0
    train_loss = 0.0
    for batchNumber in range(len(dataStorage.trainData) // batchSize):
        batchIndices = getTripletsForBatch(indices[batchNumber * batchSize : (batchNumber + 1) * batchSize])
        
        inputs = torch.tensor([dataStorage.trainData[i] for i in batchIndices], dtype = torch.float32)
        inputs = inputs.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = lossFunc(outputs[:batchSize], outputs[batchSize:2*batchSize], outputs[2*batchSize:])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        for i in range(batchSize):
            if torch.dist(outputs[i], outputs[batchSize + i]) < torch.dist(outputs[batchSize + i], outputs[2 * batchSize + i]):
                train_correct += 1
        
        del inputs
        del outputs

    indices = [i for i in range(len(dataStorage.testData))]
    random.shuffle(indices)
    
    print('Epoch: %d Batches: %d Train loss: %.3f Train acc: %.3f' %
          (epoch + 1, len(dataStorage.trainData) // batchSize, train_loss / len(dataStorage.trainData), train_correct / len(dataStorage.trainData)))
    trainAccs.append(train_correct / len(dataStorage.trainData))
    trainLosses.append(train_loss / len(dataStorage.trainData))

    #Validation
    print('Start validation')
    model.train(False)
    val_correct = 0
    val_loss = 0.0
    for batchNumber in range(len(dataStorage.testData) // batchSize):
        batchIndices = getTripletsForBatch(indices[batchNumber * batchSize : (batchNumber + 1) * batchSize], isTrain=False)
        
        inputs = torch.tensor([dataStorage.testData[i] for i in batchIndices], dtype = torch.float)
        inputs = inputs.to(device)
        
        outputs = model(inputs)
        loss = lossFunc(outputs[:batchSize], outputs[batchSize:2*batchSize], outputs[2*batchSize:])

        val_loss += loss.item()
        
        for i in range(batchSize):
            if torch.dist(outputs[i], outputs[batchSize + i]) < torch.dist(outputs[batchSize + i], outputs[2 * batchSize + i]):
                val_correct += 1
                
        del inputs
        del outputs
    
    print('Epoch: %d Val loss: %.3f Val acc: %.3f' %
          (epoch + 1, val_loss / len(dataStorage.testData), val_correct / len(dataStorage.testData)))
    valAccs.append(val_correct / len(dataStorage.testData))
    valLosses.append(val_loss / len(dataStorage.testData))
    
    print("%s seconds for epoch" % (time.time() - start_time))

print('Finished Training')

torch.save(model.state_dict(), "timesformerOnUCFSiamese")