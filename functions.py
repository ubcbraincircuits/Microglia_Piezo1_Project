import numpy as np
import os
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.util as util
import tifffile


def returnFilePaths(dir_path, type=None):
    ls = []
    for file in os.listdir(dir_path):
        if type is not None:
            if file.endswith(type):
                ls.append(file)
        else:
            ls.append(file)
    ls = sorted(ls)
    return ls

# This function will load the the lsm file
# Increase the contrast of the bright feild image and inver it

def lsm2tif(image_path, filename):
    name = os.path.basename(filename)
    raw_image = tifffile.imread(image_path+'/'+filename)
    print(raw_image.shape)
    channel1 = raw_image[0, :, 0, :, :]
    channel2 = raw_image[0, :, 1, :, :]
    #output = np.zeros((raw_image.shape[1],raw_image.shape[3],raw_image.shape[4]),np.uint16)
    contrast = util.invert(channel2)
    for i in range(contrast.shape[0]):
        contrast[i, :, :] = filters.rank.enhance_contrast(contrast[i, :, :], selem=morphology.disk(3))
    output = filters.gaussian(contrast, sigma=1)
    wd = os. getcwd()
    if not os.path.exists(wd+'/scratch/tif'):
        os.makedirs(wd+'/scratch/tif')
    filename = wd+'/scratch/tif/prepos'+name+'.tif'
    print(output.shape)

    tifffile.imsave(filename, output)


def deltaFOverF0(data, hz, t0=0.2, t1=0.75, t2=3.0, iterFunc=None):
    t0ratio = None if t0 is None else np.exp(-1 / (t0 * hz))
    t1samples, t2samples = round(t1 * hz), round(t2*hz)

    def _singeRowDeltaFOverF(samples):
        fBar = _windowFunc(np.mean, samples, t1samples, mid=True)
        f0 = _windowFunc(np.min, fBar, t2samples)
        result = (samples - f0) / f0
        if t0ratio is not None:
            result = _ewma(result, t0ratio)
        return result
    return _forEachTimeseries(data, _singeRowDeltaFOverF, iterFunc)


def _windowFunc(f, x, window, mid=False):
    n = len(x)
    startOffset = (window - 1) // 2 if mid else window - 1
    result = np.zeros(x.shape)
    for i in range(n):
        startIdx = i - startOffset
        endIdx = startIdx + window
        startIdx, endIdx = max(0, startIdx), min(endIdx, n)
        result[i] = f(x[startIdx:endIdx])
    return result


def _ewma(x, ratio):
    result = np.zeros(x.shape)
    weightedSum, sumOfWeights = 0.0, 0.0
    for i in range(len(x)):
        weightedSum = ratio * weightedSum + x[i]
        sumOfWeights = ratio * sumOfWeights + 1.0
        result[i] = weightedSum / sumOfWeights
    return result

# Input is either 1d (timeseries), 2d (each row is a timeseries) or 3d (x, y, timeseries)
def _forEachTimeseries(data, func, iterFunc=None):
    if iterFunc is None:
        iterFunc = lambda x: x
    dim = len(data.shape)
    result = np.zeros(data.shape)
    if dim == 1: # single timeseries
        result = func(data)
    elif dim == 2: # (node, timeseries)
        for i in iterFunc(range(data.shape[0])):
            result[i] = func(data[i])
    elif dim == 3: # (x, y, timeseries)
        for i in iterFunc(range(data.shape[0])):
            for j in iterFunc(range(data.shape[1])):
                result[i, j] = func(data[i, j])
    return result
