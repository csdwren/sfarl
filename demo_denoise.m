
run(fullfile('matlab', 'vl_layers', 'vl_setupnn.m'));

opts.cuda = true;

opts.filePath = 'datasets/denoise/BSD68/test001.png'; 
opts.noise_std = 50; %% 15, 25 or 50
opts.model = 'sfarlnet';
opts.netPath = ['models/denoise/net-sigma',num2str(opts.noise_std),'.mat'];

f = single(imread(opts.filePath));

y = f + opts.noise_std * randn(size(f),'like',f);

if opts.cuda
    y = gpuArray(y);
end

tic
x = sfarlnet_den(y,'stdn',opts.noise_std,'netPath',opts.netPath);
toc

if opts.cuda
    x = gather(x);
end

subplot(1,2,1); imshow(im2uint8(y/255)); title('noisy image')
subplot(1,2,2); imshow(im2uint8(x/255)); title('denoising result')


