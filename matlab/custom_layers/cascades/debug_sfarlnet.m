x = single(imread('/media/r/BC580A85580A3F20/dataset/BSDS500/gray/2018.jpg'));
support = [7 7]; numFilters=48; padSize = [2,2,2,2]; stride=[1,1];
zeroMeanFilters = true; weightNormalization = true;
stdn = 25; 
y = x+stdn*randn(size(x),'like',x);
alpha = 0;
cid = 'single';
h = misc.gen_dct3_kernel(support,'classType',cid,'gpu',false);
h = h(:,:,:,1:end-1);
s = ones(1,numFilters,cid);
rbf_means=cast(-310:10:310,cid); rbf_precision = 10;
rbf_weights = randn(numFilters,numel(rbf_means),cid);

obs = randn(size(x),'like',x);

%opts.rbf_means = -100:4:100;
step = 0.1;
origin = -104;
data_mu=cast(origin:step:-origin,cid);
data_mu=bsxfun(@minus,data_mu,cast(rbf_means(:),cid));

%%
support_data= [5 5]; numFilters_data=25; padSize_data = [3,3,3,3]; stride_data=[1,1];
zeroMeanFilters = true; weightNormalization = true;
alpha = 0;
lambda = 0;
lambda = cast(lambda,cid);
h_data = misc.gen_dct3_kernel(support_data,'classType',cid,'gpu',false);
s_data = ones(1,numFilters_data,cid);
rbf_means_data=cast(-310:10:310,cid); rbf_precision_data = 10;
rbf_weights_data = randn(numFilters_data,numel(rbf_means_data),cid);
step = 0.1;
origin_data = -104;
data_mu_data=cast(origin:step:-origin,cid);
data_mu=bsxfun(@minus,data_mu,cast(rbf_means(:),cid));
%%

padType = 'symmetric';
[y,~,~,~,~,~,~,~,~,~,~,~,~,J,J_data,M] = sfarlnet(x,obs,h,h_data,[],[],s,s_data,[],[],rbf_weights,rbf_weights_data,rbf_means,...
  rbf_precision,stdn,alpha,lambda,[],'stride',stride,'padSize',padSize,'padSize_data',padSize_data,...
  'padType',padType,'zeroMeanFilters',zeroMeanFilters,...
  'weightNormalization',weightNormalization,'data_mu',data_mu,'conserveMemory',false);
dzdy = randn(size(y{end}),'like',x);
input = {y{6},y{5},y{4},y{3},y{2},y{1},x};
[y,dh,dh_data,~,~,ds,ds_data,~,~,dw,dw_data,da,dl] = sfarlnet(input,obs,h,h_data,[],[],s,s_data,[],[],rbf_weights,rbf_weights_data,rbf_means,...
  rbf_precision,stdn,alpha,lambda,dzdy,'stride',stride,'padSize',padSize,...
  'padType',padType,'padSize_data',padSize_data,'zeroMeanFilters',zeroMeanFilters,'data_mu',data_mu,...
  'weightNormalization',weightNormalization);