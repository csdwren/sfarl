
run(fullfile('matlab', 'vl_layers', 'vl_setupnn.m'));

opts.cuda = false;

opts.filePath = 'datasets/deconvolution/disk/1.jpg';
opts.model = 'sfarlnet';
opts.netPath = 'models/deconvolution/net-disk.mat';

blur_kernel = fspecial('disk',7);
opts.blur_kernel = single(blur_kernel);

y = imread(opts.filePath);
blurry = y;
tmpy = y;
if size(tmpy,3) == 3
    tmpy = rgb2ycbcr(y);
    y = tmpy(:,:,1);
end
y = single(y);

if opts.cuda
    y = gpuArray(y);
end

x = sfarlnet_deblur_den(y,'blur_kernel',opts.blur_kernel,'netPath',opts.netPath);


if opts.cuda
    x = gather(x);
end

x = uint8(x);
if size(tmpy,3) == 3
    tmpy(:,:,1) = x;
end

x = ycbcr2rgb(tmpy);

if opts.cuda
    x = gather(x);
end

subplot(1,2,1); imshow(blurry); title('blurry image')
subplot(1,2,2); imshow(x); title('deblurring result')
