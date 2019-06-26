function [a,b]=param_beta(m,v)
% a and b are parameters of a beta pdf with expectation m and variance v
%
% Defined by 
% [1] Come, E., Oukhellou, L., Denoeux, T., Aknin, P., 2009. Learning from
% partially supervised data using mixture models and belief functions.
% Pattern recognition 42 (3), 334–348.
% Allows to generate uncertain and noisy labels
%
a=m.^2*(1-m)/v -m;
b=m*(1-m)/v -1 - a;

