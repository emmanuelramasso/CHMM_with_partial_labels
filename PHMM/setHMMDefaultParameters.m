function parametersAlgorithm = setHMMDefaultParameters
%
% Hidden Markov Models with Multicomponents Gaussian outputs and uncertain and noisy labels
% Also allows a use for estimation of Gaussian Mixture Models with uncertain noisy and noisy labels
% HMM are well known methods to statistically represent time-series. 
% An extension is proposed in [1] which allows to take into account partial
% knowledge on hidden states to correct the posterior distributions and their parameters.
% 
% This function initialises some parameters of the algorithm.
% 
% parametersAlgorithm is a structure with the following fields:
%
%	 .nessai [default=10] (>=1) is the number of trials to help the algorithm to converge to "good" solutions, 
%	 choose the model with the highest likelihood afterwards
% 
%	 .idiag = true or false [default=false], if true then considers diagonal covariance
%
%	 .iltr = true or false [default=false], if true then considers left to right HMM
% 
%	 .visu = true or false  [default=false], if true then display some information
% 
%	 .iplot = true or false  [default=false], if true then plot likelihood
% 
%	 .thresh  [default=1e-6] = threshold for convergence detection 
%
% 	 .init [default=false] = use pl to initialise the parameters
%
%	 .nitermax [default=100] = the number of iterations for convergence, stop
%	 the algorith when reached
%
%	.isHMM [default=true] = HMM or GMM model if not empty.
%
% 	.phmmInit is a structure with initial parameters for the HMM, could be empty [default]. 
%	Below K=nbStates, M=nbComponents):
% 		phmmInit.mu = means of components size KxFxM, mu(k,:,m) means for state k in all features in component m
% 		phmmInit.sig = covariance of components, size FxFxKxM
% 		phmmInit.mix = mixture weights, size KxM
% 		phmmInit.Pi = initial probability, size 1xK, p(s=i at t=1)
% 		phmmInit.A = transition matrix, size KxK, p(s=i at t-1 | s=j at t)
% 

parametersAlgorithm.nessai=10;
parametersAlgorithm.idiag = false;
parametersAlgorithm.iltr = false;
parametersAlgorithm.visu = false;
parametersAlgorithm.iplot = false;
parametersAlgorithm.thresh = 1e-6;
parametersAlgorithm.init = false;
parametersAlgorithm.nitermax = 100;
parametersAlgorithm.isHMM=true;

parametersAlgorithm.phmmInit = [];

