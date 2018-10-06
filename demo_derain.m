
run(fullfile('matlab', 'vl_layers', 'vl_setupnn.m'));

opts.cuda = false;

opts.filePath = 'datasets/derain/1.png';
opts.model = 'sfarlnet';
opts.netPath = 'models/derain/net-derain_1.mat';
% opts.netPath = 'models/derain/net-derain_lr.mat';


y = imread(opts.filePath);
% y = fliplr(y);
rainy = y;
tmpy = y;
if size(tmpy,3) == 3
    tmpy = rgb2ycbcr(y);
    y = tmpy(:,:,1);
end
y = single(y);

if opts.cuda
    y = gpuArray(y);
end

x = sfarlnet_den(y,'netPath',opts.netPath);

x = uint8(x);
if size(tmpy,3) == 3
    tmpy(:,:,1) = x;
end

x = ycbcr2rgb(tmpy);

if opts.cuda
    x = gather(x);
end

subplot(1,2,1); imshow(rainy); title('rainy image')
subplot(1,2,2); imshow(x); title('deraining result')


