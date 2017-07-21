import os
import sys
import numpy as np
from skvideo.io import vread, vreader, FFmpegReader
import skimage.io as io
import skimage
from skimage.transform import resize

def GetFrames(fileName, skipLength = 1, debug = False):
    '''
    Get video frames after skipping
    Args:
        fileName: full fileName to read
        skipLength: Number of skips to perform
    Returns:
        Numpy array of frames
    '''

    if debug:
        print "Started creating Frame List for file", fileName

    try:
        frameList = vread(fileName)
    except:
        return None

    if debug:
        print 'The video shape is', frameList.shape
        print 'The array type is', frameList.dtype

    frameList = frameList[range(0, frameList.shape[0], skipLength), :, :, :]
    # Skip frames according to skipLength

    if debug:
        print 'The new shape after skipping', skipLength, 'is', frameList.shape
        print "Finished creating Frame List"

    return frameList


def ResizeFrameList(frameList, row = 280, col = 280):
    '''
    Resizes a frameList to (row, col)
    Args: 
        frameList: height x width x channels
        row, col: Dimensions to resize into
    Returns:
        Numpy array (row x col x channels)
    '''

    origType = frameList.dtype

    if ((frameList.shape[1] < row) or (frameList.shape[2] < col)):
        ratio = min( ((frameList.shape[1] * (1.0))/row), ((frameList.shape[2] * (1.0))/col))
        ind1 = int(ratio * row)
        ind2 = int(ratio * col)
        frameList = frameList[:, :ind1, :ind2, :]
    elif ((frameList.shape[1] == row) and (frameList.shape[2] == col)):
        return frameList

    newFrameList = []
    for i in range(frameList.shape[0]):
        newImg = resize(frameList[i], (row, col, frameList.shape[3]))
        newImg = (newImg * 255)
        newFrameList.append(newImg)
    newFrameList = np.array(newFrameList, dtype = origType)

    return newFrameList


def GetSpacedBlocks(frameList, numBlocks = 25, numFrames = 25, spaceLength = 1, debug = False, wellSpaced = False):
    '''
    Extract multiple spaced blocks, returns the first numBlocks such samples
    Args: 
        numBlocks: Number of blocks to sample
        numFrames: Number of frames in a sample
        spaceLength: Distance to maintain between consecutive frames
        wellSpaced: Sets skipLength such that the computed samples are well spaced
    Returns:
        Numpy array (numBlocks x numFrames x height x width x channels)
    '''

    origType = frameList.dtype

    if (spaceLength < numBlocks):
        if debug:
            print 'WARNING: Not enough size, returned blocks will be smaller than desired!'
        numBlocks = spaceLength

    if (numFrames > (frameList.shape[0]/spaceLength)):
        if debug:
            print 'WARNING: spaceLength too large!'
        indList = [i for i in range(frameList.shape[0]) for _ in range( ((numFrames * spaceLength)/frameList.shape[0]) + 1 )]
        frameList = frameList[ indList, :, :, :]

    skipLength = 1
    if (wellSpaced):
        skipLength = max(((frameList.shape[0] - (spaceLength * numFrames) + 1)/numBlocks), 1)

    blockList = []
    cnt = 0
    for i in range(0, frameList.shape[0] - (spaceLength * numFrames) + 1, skipLength):
        indList = range(i, frameList.shape[0], spaceLength)[: numFrames]
        while (len(indList) < numFrames):
            indList.append(indList[-1])
        blockList.append(frameList[indList])

        if (len(indList) != numFrames):
            print 'numFrame mismatch in current file! Note file name.'

        cnt += 1
        if (cnt >= numBlocks):
            break

    blockList = np.array(blockList, dtype = origType)

    if debug:
        print "The final size is", blockList.shape
        print "The final type is", blockList.dtype

    return blockList

def GetAdaptiveSpacedBlocks(frameList, numBlocks = 25, numFrames = 25, debug = False):
    '''
    Extract multiple spaced blocks, returns the first numBlocks such samples. Spacing between frames depends
    Args: 
        numBlocks: Number of blocks to sample
        numFrames: Number of frames in a sample
    Returns:
        Numpy array (numBlocks x numFrames x height x width x channels)
    '''

    spaceLength = max((frameList.shape[0]/(numFrames+1)), 1)

    if (numBlocks > numFrames):
        print 'WARNING: numBlocks too large!'
        numBlocks = numFrames - 1

    if (debug):
        print 'Frame Shape:', frameList.shape
        print 'Adaptive Spacing:', numBlocks, numFrames, spaceLength

    return GetSpacedBlocks(frameList, numBlocks, numFrames, spaceLength, debug = debug, wellSpaced = True)


'''
Flow sampling functions
'''

#def flowList(flowPath, xFileNames, yFileNames):
def flowList(xFileNames, yFileNames):
    '''
    (x/y)fileNames: List of the fileNames in order to get the flows from
    '''

    frameList = []

    if (len(xFileNames) != len(yFileNames)):
        print 'XFILE!=YFILE ERROR: In', xFileNames[0]

    for i in range(0, min(len(xFileNames), len(yFileNames))):
        imgX = io.imread(xFileNames[i])
        imgY = io.imread(yFileNames[i])
        frameList.append(np.dstack((imgX, imgY)))

    frameList = np.array(frameList)
    return frameList

def getStackFlowList(flowList, window = 3, altFlag = False):
    '''
    flowList: Array of size (numFrame x row x col x channel)
    window: Window to concatenate, both left and right frames are taken
    altFlag: Consider only alternate frames
    '''

    frameList = []
    winFact = 1 if (not altFlag) else 2
    for i in range((window*winFact), flowList.shape[0] - (window*winFact)):
        indList = range(i - (window*winFact), i + (window*winFact), winFact)
        tmpCut = [tuple([flowList[j] for j in indList]), 2]
        # tmpCut is the argument to np concatenate
        tmpCut = np.concatenate(*tmpCut)
        frameList.append(tmpCut)

    frameList = np.array(frameList)
    return frameList

def GetFlowSample(frameList, savePath = 'vidData/'):

    blockList = GetAdaptiveSpacedBlocks(frameList, numBlocks = 2, numFrames = 25, debug = False)

    row, col = 240, 320
    newFrameList = []
    if ((blockList.shape[1] != row) or (blockList.shape[2] != col)):
        for i in range(blockList.shape[0]):
            tmpFrameList = ResizeFrameList(blockList[i], row = row, col = col)
            newFrameList.append(tmpFrameList)
    else:
        newFrameList = blockList

    newFrameList = np.array(newFrameList, dtype = np.uint8)
    return newFrameList

'''
RGB sampling functions
'''

def GetRGBSample(frameList, savePath = 'vidData/'):

    blockList = GetAdaptiveSpacedBlocks(frameList, numBlocks = 2, numFrames = 25, debug = False)

    newFrameList = []
    for i in range(blockList.shape[0]):
        tmpFrameList = ResizeFrameList(blockList[i], row = 240, col = 320)
        newFrameList.append(tmpFrameList)

    newFrameList = np.array(newFrameList, dtype = np.uint8)
    return newFrameList

'''
Function to sample rgb and flow
'''

def SampleUCFV1(classNames, savePath, rgbPath, flowPath, type = 'dual'):
    '''
    Create train test data for the new flow data in the form of subdirectories
    savePath: Path to save to
    flowPath: Path to read the flow files (The parent containing the subdirs)
    rgbPath: Path to read the rgb video files
    type: rgb, flow, dual
    '''

    for classDir in classNames:

        tmpPath = flowPath + classDir + '/'
        vidNames = os.listdir(tmpPath)
        vidNames = [x for x in vidNames if os.path.isdir(tmpPath + x)]

        cnt = 0

        for vidDir in vidNames:

            totPath = tmpPath + vidDir + '/'
            vidNameListX, vidNameListY = [], []
            imgNames = [x for x in os.listdir(totPath) if x.endswith(".jpg")]

            if ((type == 'flow') or (type == 'dual')):
                for i in range(0, len(imgNames)):
                    if ('flow_x' in imgNames[i]):
                        vidNameListX.append(imgNames[i]);
                    else:
                        vidNameListY.append(imgNames[i]);

                # File Format: flow_x_int.jpg
                vidNameListX = sorted(vidNameListX, key=lambda x: int((x.strip('.jpg')).split('_')[2]))
                vidNameListY = sorted(vidNameListY, key=lambda x: int((x.strip('.jpg')).split('_')[2]))

                vidNameListX = [totPath + x for x in vidNameListX]
                vidNameListY = [totPath + x for x in vidNameListY]

            fileName = str(vidDir)
            fileName.strip('/')

            # If file already exists, no need to compute it again
            if (os.path.isfile(savePath + fileName.replace('.avi', '') + '.npz')):
                continue

            window, winFact = 5, 1

            if ((type == 'flow') or (type == 'dual')):
                flowsList = flowList(vidNameListX, vidNameListY)
                stackFlow = getStackFlowList(flowsList, window = window, altFlag = False)
                flowData = GetFlowSample(stackFlow)

            if ((type == 'rgb') or (type == 'dual')):
                frameList = GetFrames(rgbPath + fileName +'.avi', 1, False)
                if (frameList is None):
                    print 'SKIPPED... at file', fileName
                    continue

                frameList = frameList[(window*winFact):(flowsList.shape[0] - (window*winFact))]
                rgbData = GetRGBSample(frameList)

            if ((type == 'dual')):
                if (rgbData.shape[0] != flowData.shape[0]):
                    print "ERROR! at file", fileName

            if ((type == 'dual')):
                np.savez_compressed(savePath + fileName.replace('.avi', ''), rgb = rgbData, flow = flowData)
            elif ((type == 'rgb')):
                np.savez_compressed(savePath + fileName.replace('.avi', ''), rgbData)
            elif ((type == 'flow')):
                np.savez_compressed(savePath + fileName.replace('.avi', ''), flowData)
            else:
                print "Invalid type! ERROR!"
                return None

            cnt += 1

            if ((type == 'rgb') or (type == 'dual')):
                print 'RGB: rgbData shape', rgbData.shape, 
            if ((type == 'flow') or (type == 'dual')):
                print 'FLOW: flowData shape', flowData.shape,

            print 'Currently at file:', fileName
            sys.stdout.flush()

    print '\n'


'''
Function to sample rgb and flow (Version 2, for flow downloaded later)
'''

def SampleUCFV2(vidNames, savePath, rgbPath, flowPath, type = 'dual'):
    '''
    Create train test data for the new flow data in the form of subdirectories
    savePath: Path to save to
    flowPath: Path to read the flow files
    rgbPath: Path to read the rgb video files
    type: rgb, flow, dual
    '''

    flowXPath = flowPath + 'u/'
    flowYPath = flowPath + 'v/'

    cnt = 0

    for vidDir in vidNames:

        totXPath = flowXPath + vidDir + '/'
        totYPath = flowYPath + vidDir + '/'

        if ((type == 'flow') or (type == 'dual')):
            vidNameListX = [x for x in os.listdir(totXPath) if x.endswith(".jpg")]
            vidNameListY = [x for x in os.listdir(totYPath) if x.endswith(".jpg")]

            # File Format: frame(int).jpg
            vidNameListX = sorted(vidNameListX, key=lambda x: int((x.replace('.jpg', '')).replace('frame', '')))
            vidNameListY = sorted(vidNameListY, key=lambda x: int((x.replace('.jpg', '')).replace('frame', '')))

            vidNameListX = [totXPath + x for x in vidNameListX]
            vidNameListY = [totYPath + x for x in vidNameListY]

        fileName = str(vidDir)
        fileName.strip('/')

        if (os.path.isfile(savePath + fileName.replace('.avi', '') + '.npz')):
            #print fileName, ': Already Exists'
            continue

        window = 5
        winFact = 1

        if ((type == 'flow') or (type == 'dual')):
            flowsList = flowList(vidNameListX, vidNameListY)
            stackFlow = getStackFlowList(flowsList, window = window, altFlag = False)
            flowData = GetFlowSample(stackFlow)

        if ((type == 'rgb') or (type == 'dual')):
            frameList = GetFrames(rgbPath + fileName.split('_')[1] + '/' + fileName +'.avi', 1, False)
            if (frameList is None):
                print 'SKIPPED... at file', fileName
                continue

            frameList = frameList[(window*winFact):(frameList.shape[0] - (window*winFact))]
            rgbData = GetRGBSample(frameList)

        if ((type == 'dual')):
            if (rgbData.shape[0] != flowData.shape[0]):
                print "WRONG! WRONG! at file", fileName

        if ((type == 'dual')):
            np.savez_compressed(savePath + fileName.replace('.avi', ''), rgb = rgbData, flow = flowData)
        elif ((type == 'rgb')):
            np.savez_compressed(savePath + fileName.replace('.avi', ''), rgbData)
        elif ((type == 'flow')):
            np.savez_compressed(savePath + fileName.replace('.avi', ''), flowData)
        else:
            print "Invalid type! ERROR!"
            return None

        cnt += 1

        if ((type == 'rgb') or (type == 'dual')):
            print 'RGB: rgbData shape', rgbData.shape, 
        if ((type == 'flow') or (type == 'dual')):
            print 'FLOW: flowData shape', flowData.shape,

        print 'Currently at file:', fileName

        sys.stdout.flush()

    print '\n'


'''
Function to sample rgb and flow with multiple processes
'''

def SampleUCFInParallel(savePath, rgbPath, flowPath, type = 'dual'):
    '''
    Create train test data for the new flow data in the form of subdirectories
    '''
    
    import multiprocessing

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    dirNames = os.listdir(flowPath + 'u/')
    dirNames = [x for x in dirNames if os.path.isdir(flowPath +'u/' + x)]
    dirNames.sort()

    NUM_PROCESSES = 25
    blockLen = int((len(dirNames)*1.0)/NUM_PROCESSES)+1
    jobs = []

    for i in range(NUM_PROCESSES):
        p = multiprocessing.Process(target = SampleUCFV2, args = (dirNames[i*blockLen:(i+1)*blockLen], savePath, rgbPath, flowPath, type))
        jobs.append(p)
        p.start()

def GetVideoNames(videoPath):
    vidNames = os.listdir(videoPath)
    vidNames = [x for x in vidNames if x.endswith(".avi")]
    return vidNames

if __name__ == "__main__":
    saveRGBPath = 'path/to/save/rgb/to/'
    saveFlowPath = 'path/to/save/flow/to'
    rgbPath = 'path/to/read/rgb/from/'
    flowPath = 'path/to/read/flow/from/'

    SampleUCFInParallel(saveRGBPath, rgbPath, flowPath, type = 'rgb')
    SampleUCFInParallel(saveFlowPath, rgbPath, flowPath, type = 'flow')
