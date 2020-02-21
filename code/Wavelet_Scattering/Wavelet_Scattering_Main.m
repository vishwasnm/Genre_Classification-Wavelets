%% Music Genre Classification Using Wavelet Time Scattering
% This example shows how to classify the genre of a musical excerpt using
% wavelet time scattering and the audio datastore. In wavelet scattering,
% data is propagated through a series of wavelet transforms,
% nonlinearities, and averaging to produce low-variance representations of
% the data. These low-variance representations are then used as inputs to a
% classifier.
%% GTZAN Dataset
% The data set used in this example is the GTZAN Genre Collection [7][8].
% The data is provided as a zipped tar archive which is approximately 1.2
% GB. The uncompressed data set requires about 3 GB of disk space.
% Extracting the compressed tar file from the link provided in the
% references creates a folder with ten subfolders. Each subfolder is named
% for the genre of music samples it contains. The genres are: blues,
% classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.
% There are 100 examples of each genre and each audio file consists of
% about 30 seconds of data sampled at 22050 Hz. In the original paper, the
% authors used a number of time-domain and frequency-domain features
% including mel-frequency cepstral (MFC) coefficients extracted from each
% music example and a Gaussian mixture model (GMM) classification to
% achieve an accuracy of 61 percent [7]. Subsequently, deep learning
% networks have been applied to this data. In most cases, these deep
% learning approaches consist of convolutional neural networks (CNN) with
% the MFC coefficients or spectrograms as the input to the deep CNN. These
% approaches have resulted in performance of around 84% [4]. An LSTM
% approach with spectrogram time slices resulted in 79% accuracy and
% time-domain and frequency-domain features coupled with an ensemble
% learning approach (AdaBoost) resulted in 82% accuracy on a test set
% [2][3]. Recently, a sparse representation machine learning approach
% achieved approximately 89% accuracy [6].
%% Wavelet Scattering Framework
% The only parameters to specify in a wavelet time scattering framework are
% the duration of the time invariance, the number of wavelet filter banks,
% and the number of wavelets per octave. For most applications, cascading
% the data through two wavelet filter banks is sufficient. In this example,
% we use the default scattering framework which uses two wavelet filter
% banks. The first filter bank has 8 wavelets per octave and the second
% filter bank has 1 wavelet per octave. For this example, set the invariant
% scale to be 0.5 seconds, which corresponds to slightly more than 11,000
% samples for the given sampling rate. Create the wavelet time scattering
% decomposition framework.
sf = waveletScattering('SignalLength',2^19,'SamplingFrequency',22050,...
    'InvarianceScale',0.5,...
    'QualityFactors',[8,2,1]);
%%
% To understand the role of the invariance scale, obtain and plot the
% scaling filter in time along with the real and imaginary parts of the
% coarsest-scale wavelet from the first filter bank. Note that the
% time-support of the scaling filter is essentially 0.5 seconds as
% designed. Further, the time support of the coarsest-scale wavelet does
% not exceed the invariant scale of the wavelet scattering decomposition.
[fb,f,filterparams] = filterbank(sf);
phi = ifftshift(ifft(fb{1}.phift));
psiL1 = ifftshift(ifft(fb{2}.psift(:,end)));
dt = 1/22050;
time = -2^18*dt:dt:2^18*dt-dt;
scalplt = plot(time,phi,'linewidth',1.5);
hold on
grid on
ylimits = [-3e-4 3e-4];
ylim(ylimits);
plot([-0.25 -0.25],ylimits,'k--');
plot([0.25 0.25],ylimits,'k--');
xlim([-0.6 0.6]);
xlabel('Seconds'); ylabel('Amplitude');
wavplt = plot(time,[real(psiL1) imag(psiL1)]);
legend([scalplt wavplt(1) wavplt(2)],{'Scaling Function','Wavelet-Real Part','Wavelet-Imaginary Part'});
title({'Scaling Function';'Coarsest-Scale Wavelet First Filter Bank'})
hold off
%% Audio Datastore
% The audio datastore enables you to manage collections of audio data
% files. For machine or deep learning, the audio datastore not only manages
% the flow of audio data from files and folders, the audio datastore also
% manages the association of labels with the data and provides the ability
% to randomly partition your data into different sets for training,
% validation, and testing. In this example, use the audio datastore to
% manage the GTZAN music genre collection. Recall each subfolder of the
% collection is named for the genre it represents. Set the
% |'IncludeSubFolders'| property to |true| to instruct the audio datastore
% to use subfolders and set the |'LabelSource'| property to |'foldernames'|
% to create data labels based on the subfolder names.
% This example assumes the top-level directory is inside your MATLAB
% |tempdir| directory and is called 'genres'. Ensure that
% |location| is the correct path to the top-level data folder on your
% machine. The top-level data folder on your machine should contain ten
% subfolders each named for the ten genres and must only contain audio
% files corresponding to those genres.
location = fullfile(tempdir,'genres');
ads = audioDatastore(location,'IncludeSubFolders',true,...
    'LabelSource','foldernames');
%%
% Run the following to obtain a count of the musical genres in the data
% set.
countEachLabel(ads)
%%
% As previously stated, there are 10 genres with 100 files each.
%% Training and Test Sets
% Create training and test sets to develop and test our classifier. We use
% 80% of the data for training and hold out the remaining 20% for testing.
% The |shuffle| function of the audio datastore randomly shuffles the data.
% Do this prior to splitting the data by label to randomize the data. In
% this example, we set the random number generator seed for
% reproducibility. Use the audio datastore |splitEachLabel| function to
% perform the 80-20 split. |splitEachLabel| ensures that all classes are
% equally represented.
rng(100);
ads = shuffle(ads);
[adsTrain,adsTest] = splitEachLabel(ads,0.8);
countEachLabel(adsTrain)
countEachLabel(adsTest)
%%
% You see that there are 800 records in the training data as expected and
% 200 records in the test data. Additionally, there are 80 examples of each
% genre in the training set and 20 examples of each genre in the test set.
%%
% |audioDatastore| works with MATLAB tall arrays. Create tall arrays for
% both the training and test sets. Depending on your system, the number of
% workers in the parallel pool MATLAB creates may be different.
Ttrain = tall(adsTrain);
Ttest = tall(adsTest);
%%
% To obtain the scattering features, define a helper function,
% |helperscatfeatures|, that obtains the natural logarithm of the
% scattering features for 2^19 samples of each audio file and subsamples
% the number of scattering windows by 8. The source code for
% |helperscatfeatures| is listed in the appendix. We will compute the
% wavelet scattering features for both the training and test data.
scatteringTrain = cellfun(@(x)helperscatfeatures(x,sf),Ttrain,'UniformOutput',false);
scatteringTest = cellfun(@(x)helperscatfeatures(x,sf),Ttest,'UniformOutput',false);
%%
% Compute the scattering features on the training data and bundle all the
% features together in a matrix. This process takes several minutes.
TrainFeatures = gather(scatteringTrain);
TrainFeatures = cell2mat(TrainFeatures);
%%
% Repeat this process for the test data.
TestFeatures = gather(scatteringTest);
TestFeatures = cell2mat(TestFeatures);
%%
% Each row of |TrainFeatures| and |TestFeatures| is one scattering time
% window across the 341 paths in the scattering transform of each audio
% signal. For each music sample, we have 32 such time windows. Accordingly,
% the feature matrix for the training data is 25600-by-341. The number of
% rows is equal to the number of training examples (800) multiplied by the
% number of scattering windows per example (32). Similarly, the scattering
% feature matrix for the test data is 6400-by-341. There are 200 test
% examples and 32 windows per example. Create a genre label for each of the
% 32 windows in the wavelet scattering feature matrix for the training
% data.
numTimeWindows = 32;
trainLabels = adsTrain.Labels;
numTrainSignals = numel(trainLabels);
trainLabels = repmat(trainLabels,1,numTimeWindows);
trainLabels = reshape(trainLabels',numTrainSignals*numTimeWindows,1);
%%
% Repeat the process for the test data.
testLabels = adsTest.Labels;
numTestSignals = numel(testLabels);
testLabels = repmat(testLabels,1,numTimeWindows);
testLabels = reshape(testLabels',numTestSignals*numTimeWindows,1);
%%
% In this example, use a multi-class support vector machine (SVM)
% classifier with a cubic polynomial kernel. Fit the SVM to the training
% data.
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
Classes = {'blues','classical','country','disco','hiphop','jazz',...
    'metal','pop','reggae','rock'};
classificationSVM = fitcecoc(...
    TrainFeatures, ...
    trainLabels, ...
    'Learners', template, ...
    'Coding', 'onevsone','ClassNames',categorical(Classes));
%% Test Set Prediction
% Use the SVM model fit to the scattering transforms of the training data
% to predict the music genres for the test data. Recall there are 32 time
% windows for each signal in the scattering transform. Use a simple
% majority vote to predict the genre. The helper function
% |helperMajorityVote| obtains the mode of the genre labels over all 32
% scattering windows. If there is no unique mode, |helperMajorityVote|
% returns a classification error indicated by |'NoUniqueMode'|. 
% This results in an extra column in the confusion matrix.
% The source code for |helperMajorityVote| is listed in the appendix.
predLabels = predict(classificationSVM,TestFeatures);
[TestVotes,TestCounts] = helperMajorityVote(predLabels,adsTest.Labels,categorical(Classes));
testAccuracy = sum(eq(TestVotes,adsTest.Labels))/numTestSignals*100;
%%
% The test accuracy, |testAccuracy|, is 88 percent. This accuracy is
% comparable with the state of the art of the GTZAN dataset.
%%
% Display the confusion matrix to inspect the genre-by-genre accuracy
% rates. Recall there are 20 examples in each class.
confusionchart(TestVotes,adsTest.Labels);
%%
% The diagonal of the confusion matrix plot shows that the classification
% accuracies for the individual genres is quite good in general. Extract
% these genre accuracies and plot separately.
figure;
cm = confusionmat(TestVotes,adsTest.Labels);
cm(:,end) = [];
genreAccuracy = diag(cm)./20*100;
figure;
bar(genreAccuracy)
set(gca,'XTickLabels',Classes);
xtickangle(gca,30);
title('Percentage Correct by Genre - Test Set');
%% Summary
% This example demonstrated the use of wavelet time scattering and the
% audio datastore in music genre classification. In this example, wavelet
% time scattering achieved an classification accuracy comparable to state
% of the art performance for the GTZAN dataset. As opposed to other
% approaches requiring the extraction of a number of time-domain and
% frequency-domain features, wavelet scattering only required the
% specification of a single parameter, the scale of the time invariant. The
% audio datastore enabled us to efficiently manage the transfer of a large
% dataset from disk into MATLAB and permitted us to randomize the data and
% accurately retain genre membership of the randomized data through the
% classification workflow.
%% References
% # Anden, J. and Mallat, S. 2014. Deep scattering spectrum. IEEE Transactions on Signal Processing, Vol. 62, 16, pp. 4114-4128. 
% # Bergstra, J., Casagrande, N., Erhan, D., Eck, D., and Kegl, B. Aggregate features and AdaBoost for music classification. Machine Learning, Vol. 65, Issue 2-3, pp. 473-484.  
% # Irvin, J., Chartock, E., and Hollander, N. 2016. Recurrent neural networks with attention for genre classification. https://www.semanticscholar.org/paper/Recurrent-Neural-Networks-with-Attention-for-Genre-Irvin-Chartock/bff3eaf5d8ebb6e613ae0146158b2b5346ee7323 
% # Li, T., Chan, A.B., and Chun, A. 2010. Automatic musical pattern feature extraction using convolutional neural network. International Conference Data Mining and Applications.  
% # Mallat. S. 2012. Group invariant scattering. Communications on Pure and Applied Mathematics, Vol. 65, 10, pp. 1331-1398.
% # Panagakis, Y., Kotropoulos, C.L., and Arce, G.R. 2014. Music genre classification via joint sparse low-rank representation of audio features. IEEE Transactions on Audio, Speech, and Language Processing, 22, 12, pp. 1905-1917. 
% # Tzanetakis, G. and Cook, P. 2002. Music genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, Vol. 10, No. 5, pp. 293-302. 
% # _GTZAN Genre Collection_. |http://marsyas.info/downloads/datasets.html|
%% Appendix -- Supporting Functions
% *helperMajorityVote* -- This function returns the mode of the class
% labels predicted over a number of feature vectors. In wavelet time
% scattering, we obtain a class label for each time window. If no unique
% mode is found a label of 'NoUniqueMode' is returned to denote a
% classification error.
%
% <include>helperMajorityVote.m</include>
%
%%
% *helperscatfeatures* - This function returns the wavelet time scattering
% feature matrix for a given input signal. In this case, we use the natural
% logarithm of the wavelet scattering coefficients. The scattering feature
% matrix is computed on 2^19 samples of a signal. The scattering features
% are subsampled by a factor of 8.
%
% <include>helperscatfeatures.m</include>
%
% Copyright 2018 The MathWorks, Inc.