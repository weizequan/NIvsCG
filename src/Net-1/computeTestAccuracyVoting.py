# Computing average accuracy on cropped patch (240 x 240) and full-sized image after voting
# This file can also be modified for other patch sizes, i.e., 180 x 180, 120 x 120, etc. 

import numpy as np
import sys,os,caffe

caffe_root = ''  # the root of caffe
kProjectRoot = ''  # the directory includes trained caffemodel and deploy.prototxt
kDataRoot = ''  # the directory includes file of the mean value of training data 
kCaffeModelName = ''  # the trained caffemodel name
kPrcgNum = 200  # the number of cg images

sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)
if not os.path.isfile(caffe_root + kProjectRoot+ kCaffeModelName):
    print("caffemodel is not exist...")

caffe.set_device(1)
caffe.set_mode_gpu()

model_def = caffe_root + kProjectRoot + 'deploy.prototxt'
model_weights = caffe_root + kProjectRoot + kCaffeModelName
mu = np.load(caffe_root + kDataRoot + 'imagenet_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print('mean-subtracted values:', zip('BGR', mu))

net = caffe.Classifier(model_def, model_weights, image_dims=(240, 240),
                    mean=mu, raw_scale=255,
                    channel_swap=(2,1,0))

imageLabel = []
testLabel = []
imageTmp = []
testTmp = []
oriImageLabel = []  # one dimension list
oriTestLabel = []  # one dimension list

# test240_30_num.txt records the name, label of image patch and the number of cropped patches for each test image (i.e., 30)
# for example:
# prcg_images/set1-arch-11-1.bmp(the name of 1-th patch) 0(label)
# prcg_images/set1-arch-11-2.bmp(the name of 2-th patch) 0(label)
# ...
# prcg_images/set1-arch-11-30.bmp(the name of 30-th patch) 0(label)
# 30 (the number of cropped patches for each test image)
# ...

# Note that, [1] and [2] need to be refined for your own data
testImageDir = caffe_root + 'data/columbia-prcg-datasets/test240_30_num.txt'  # [1]the info of test image patch
testImageFile = open(testImageDir, 'r')
for line in testImageFile:
    twoTuple = line.split()
    if len(twoTuple) == 2:
        image = caffe.io.load_image(caffe_root + 'data/columbia-prcg-datasets/test_images_240_30/' + twoTuple[0])  # [2]the test image dir
        imageTmp.append(int(twoTuple[1]))
        output = net.predict([image], oversample=True)
        output_prob = output[0]
        testTmp.append(output_prob.argmax())

    else:
        oriImageLabel.extend(imageTmp)
        oriTestLabel.extend(testTmp)
        imageLabel.append(imageTmp)
        testLabel.append(testTmp)
        imageTmp = []
        testTmp = []

print('The number of full-sized testing images is:', len(imageLabel))

imageCropNum = [len(x) for x in imageLabel]
imageCropNumNp = np.array(imageCropNum)
imageLabelNp = np.array(imageLabel)
testLabelNp = np.array(testLabel)

#  Computing average accuracy on patches
result = np.array(oriImageLabel) == np.array(oriTestLabel)

prcg_result = result[:kPrcgNum*30]
google_result = result[kPrcgNum*30:]
print('The number of patches:', len(oriImageLabel), len(prcg_result), len(google_result))
print('The average accuracy on patches:')
print('The google (NI) accuracy is:', google_result.sum()*1.0/len(google_result))
print('The prcg (CG) accuracy is:', prcg_result.sum()*1.0/len(prcg_result))
print('CG patches misclassified as natural patches (CGmcNI) is:', (len(prcg_result) - prcg_result.sum())*1.0/len(prcg_result))
print('natural patches misclassified as CG patches (NImcCG) is:', (len(google_result) - google_result.sum())*1.0/len(google_result))
print('The average accuracy is:', result.sum()*1.0/len(result))

#  Computing average accuracy on full-sized images (29 patches and majority voting)
result = np.arange(len(imageLabel))
for x in range(len(imageLabel)):
    tmp = np.array(imageLabelNp[x]) == np.array(testLabelNp[x])
    result[x] = np.sum(tmp[:-1]) > imageCropNumNp[x]//2 - 1

prcg_result = result[:kPrcgNum]
google_result = result[kPrcgNum:]
print('The average accuracy on full-sized images after majority voting: ', len(prcg_result), len(google_result))
print('The google (NI) accuracy is:', google_result.sum()*1.0/len(google_result))
print('The prcg (CG) accuracy is:', prcg_result.sum()*1.0/len(prcg_result))
print('CG images misclassified as natural images (CGmcNI) is:', (len(prcg_result) - prcg_result.sum())*1.0/len(prcg_result))
print('natural images misclassified as CG images (NImcCG) is:', (len(google_result) - google_result.sum())*1.0/len(google_result))
print('The average accuracy is:', result.sum()*1.0/len(result))
