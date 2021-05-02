%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A SIMPLE TEST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, close all

disp('This code will generate a video to show the convergence')
disp(' (press enter) ')
pause

% Model: 3 states, data in 3 dimensions, use in the paper
K=3;
d=3;
MU=2*[1 0 0;0 1 0; 0 0 10]; % means
SIG=[1 1 1 % cov
    1 1 1
    1 1 1];
Pi=ones(K,1)/K; % vector of initial prob in HMM
A=[0.6 0.3 0.1 % matrix of transitions
    0.1 0.6 0.3
    0.1 0.3 0.6];

% Generation of data for learning
T=200;
x=zeros(T,d);
y=zeros(T,1);
y(1)=find(mnrnd(1,Pi));
x(1,:)=mvnrnd(MU(y(1),:),SIG(y(1),:));
for t=2:T;
    y(t)= find(mnrnd(1,A(y(t-1),:)));
    x(t,:)=mvnrnd(MU(y(t),:),SIG(y(t),:));
end;
I=eye(K);
pl0=I(y,:);
plvide=ones(size(pl0));

%testing data
Tt=1000;
xt=zeros(Tt,d);
yt=zeros(Tt,1);
yt(1)=find(mnrnd(1,Pi));
xt(1,:)=mvnrnd(MU(yt(1),:),SIG(yt(1),:));
for t=2:Tt;
    yt(t)= find(mnrnd(1,A(yt(t-1),:)));
    xt(t,:)=mvnrnd(MU(yt(t),:),SIG(yt(t),:));
end;

rho=1;
if rho==0,
    perr=zeros(T,1);
elseif rho==1,
    perr=ones(T,1);
else
    [a,b]=param_beta(rho,(0.2).^2);
    perr=betarnd(a,b,T,1);
end;
[pl,y1,pl1]=add_noise1(y,perr,K);

parametersAlgorithm = setHMMDefaultParameters;
parametersAlgorithm.init = false; % true => use prior to initialize
parametersAlgorithm.nitermax = 500;

K = 3; % states
M = 1; % components

%%%
% CALL
Nit=5; it = 1; 
while it<=Nit
    try
        [parametersHMM, outputsInference] = ...
            phmm_gauss_mix_learn(x, pl, K, M, parametersAlgorithm);
        
        for u=1:K, for m=1:M, [R,err] = cholcov(parametersHMM.Sigf(:,:,u,m),0);
                % SIGMA must be a square, symmetric, positive definite matrix.
                if err~=0, disp('go to catch...'), error('pb of convergence'), end, end
        end
        break
    catch
        if it+1>Nit, error('Impossible to run, look in data (nan ? inf ?) or normalize using zscore ?')
        else it=it+1; % retry...
        end
    end
end
%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% HOW TO USE THE FORWARD BACKWARD ON MULTI STATES
%%%%%%%%%%%%%%%%%%% and MULTICOMPONENTS HMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% see example 2
