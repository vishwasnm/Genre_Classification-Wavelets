function features = helperscatfeatures(x,sf)
% This function is in support of wavelet scattering examples only. It may
% change or be removed in a future release.

features = featureMatrix(sf,x(1:2^19),'Transform','log');
features = features(:,1:8:end)';
end