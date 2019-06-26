function [pl,w1,pl1]=add_noise1(w,perr,r)
% Generate noisy and uncertain labels as defined in [1].
%
% Inputs:
% w = true labels
% perr =  probability of error
% r = number of labels
%
% Outputs:
% pl(t,k) in [0,1] = uncertain labels = plausibility of state k at time t after corruption
% w1 the labels after corruption
% pl1(t,k) in {0,1} the plausibility of label k at time is 1 if w1(t)=1, 0 otherwise
%
%
% Defined by 
% [1] Come, E., Oukhellou, L., Denoeux, T., Aknin, P., 2009. Learning from
% partially supervised data using mixture models and belief functions.
% Pattern recognition 42 (3), 334–348.
% Allows to generate uncertain and noisy labels

[n,p]=size(w);
pl=zeros(n,p,max(r));
pl1=pl;

%rand('twister',5489)

w1=w;
for i=1:n,
    for j=1:p,
        x=rand;
        if x<perr(i,j),
            w1(i,j) = unidrnd(r(j));
        end;
        if isnan(w1(i,j)),
            pl(i,j,1:r(j))=ones(1,r(j));
            pl1(i,j,1:r(j))=ones(1,r(j));
        else,
%           pl(i,j,1:r(j))= perr(i,j)*ones(1,r(j));
%           pl(i,j,w1(i,j))=1;
            pl(i,j,1:r(j))= perr(i,j)*ones(1,r(j))/r(j);
           pl(i,j,w1(i,j))=pl(i,j,w1(i,j))+1-perr(i,j);
           pl1(i,j,w1(i,j))=1;
        end;
    end;
end;

pl=squeeze(pl);
pl1=squeeze(pl1);
