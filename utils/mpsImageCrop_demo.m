%----------------------------------------------------------------
%  Patch augmentation by cropping a certain number of image patches 
%  from each image using Maximal Poisson-disk Sampling (MPS)
%----------------------------------------------------------------
clc;
clear all;
kCropHeight = 240;  % the height of patch
kCropWidth = 240;  % the width of patch
kCropNum = 200;  % the number of cropped patches for each image
mps_data = load('mps/maxmps_200.pts');
rng('shuffle');  % used for randi
image_name = 'set2-arch-27.bmp';
img = imread(image_name);
pos_flag = zeros(4,1);  % four corners
for x = 1:kCropNum
    [cut pos_flag]= imageMpsCrop(img, pos_flag, mps_data, x, kCropWidth, kCropHeight); % mps sampling
    image_name_split = strsplit(image_name, '.');
    imwrite(cut, strcat(image_name_split{1}, '-', num2str(x), '.bmp'));
end

