%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A SIMPLE TEST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, close all

disp('This code will generate a video to show the convergence (press enter)')
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
parametersAlgorithm.nessai=1;
parametersAlgorithm.init=false; % true => use prior to initialize
parametersAlgorithm.nitermax = 500;

K = 3; % states
M = 1; % components

% will make a video with ground truth in color and the position of centers
% Create an avi file
parametersAlgorithm.VIDEOstruct.doVideo = true;
parametersAlgorithm.VIDEOstruct.colorCenters = rand(K,3);
parametersAlgorithm.VIDEOstruct.groundTruth = y;

parametersAlgorithm.hmmOrgmm = 'hmm';

%%%
% CALL
% will create a video vv.avi
Nit=5; it = 1; 
while it<=Nit
    try
        [parametersHMM, outputsInference] = ...
            phmm_gauss_mix_learn(x, pl, K, M, parametersAlgorithm);
        
        for u=1:K, for m=1:M, [R,err] = cholcov(parametersHMM.Sigf(:,:,u,m),0);
                % SIGMA must be a square, symmetric, positive definite matrix.
                if err~0, disp('go to catch...'), error('pb of convergence'), end, end
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

clear

disp('This code now presents how to use forward backward on multi states and multicomponents (press enter)')
pause

% Model: 3 states, data in 5 dimensions, 3 components per state, large number of points
K=5;
d=5;
M=3;
T=200000;

MU = zeros(K,d,M);
SIG = zeros(d,d,K,M);
for i=1:K, for k=1:M
        MU(i,:,k) = randn(d,1).*randsample(50,d);
        SIG(:,:,i,k) = cov(randn(200,d)).*diag(randsample(20,d));
    end
end
mixmat = mk_stochastic([2 1 1 ; 1 2 3 ; 3 1 1 ; 1 1 6 ; 1 2 3]);

Pi=ones(K,1)/K; % vector of initial prob in HMM
A=[5 3 1 eps eps ; % matrix of transitions
    2 6 3 1 eps ;
    1 4 4 6 1 ;
    eps 1 3 6 4
    eps 2 eps 3 5];
A=mk_stochastic(A)

% Generation of data for learning
T=100;
x=zeros(T,d);
y=zeros(T,1);
y(1)=find(mnrnd(1,Pi));
c=zeros(T,1);
c(1)=find(mnrnd(1,mixmat(y(1),:)));
x(1,:)=mvnrnd(MU(y(1),:,c(1)),SIG(:,:,y(1),c(1)));
for t=2:T;
    y(t)= find(mnrnd(1,A(y(t-1),:)));
    c(t)= find(mnrnd(1,mixmat(y(t),:)));
    x(t,:)=mvnrnd(MU(y(t),:,c(t)),SIG(:,:,c(t)));
end;
I=eye(K);
pl0=I(y,:);
plvide=ones(size(pl0));

%%%%%%%%%%%%%%%%%
% vary this number from 0=supervised to 1=unsupervised
rho=0.1; % uncertain label
[a,b]=param_beta(rho,(0.2).^2);
perr=betarnd(a,b,T,1);
[pl,y1,pl1]=add_noise1(y,perr,K);

parametersAlgorithm = setHMMDefaultParameters;
parametersAlgorithm.init=true;

% will make a video with ground truth in color and the position of centers
% Create an avi file
parametersAlgorithm.VIDEOstruct =  [];
% parametersAlgorithm.VIDEOstruct.doVideo = true;
% parametersAlgorithm.VIDEOstruct.colorCenters = rand(K,3);
% parametersAlgorithm.VIDEOstruct.groundTruth = y;

%%% PARTIALLY SUPERVISED
% CALL
% will create a video vv.avi
Nit=5; it = 1; 
while it<=Nit
    try
        [parametersHMM, outputsInference] = ...
            phmm_gauss_mix_learn(x, pl, K, M, parametersAlgorithm);
        
        for u=1:K, for m=1:M, [R,err] = cholcov(parametersHMM.Sigf(:,:,u,m),0);
                % SIGMA must be a square, symmetric, positive definite matrix.
                if err~0, disp('go to catch...'), error('pb of convergence'), end, end
        end
        break
    catch
        if it+1>Nit, error('Impossible to run, look in data (nan ? inf ?) or normalize using zscore ?')
        else it=it+1; % retry...
        end
    end
end
%%%

%%%
% CALL UNSUPERVISED
% will create a video vv.avi
Nit=5; it = 1; 
plunsup=ones(size(pl)); 
while it<=Nit
    try
        [parametersHMMunsup, outputsInferenceunsup] = ...
            phmm_gauss_mix_learn(x, plunsup, K, M, parametersAlgorithm);
        
        for u=1:K, for m=1:M, [R,err] = cholcov(parametersHMMunsup.Sigf(:,:,u,m),0);
                % SIGMA must be a square, symmetric, positive definite matrix.
                if err~0, disp('go to catch...'), error('pb of convergence'), end, end
        end
        break
    catch
        if it+1>Nit, error('Impossible to run, look in data (nan ? inf ?) or normalize using zscore ?')
        else it=it+1; % retry...
        end
    end
end
%%%


% Inference => all are in outputsInference for training
% on other data

disp('Inference on new data (press enter)'), pause
% Generation of data for testing
T=1000;
xt=zeros(T,d);
yt=zeros(T,1);
yt(1)=find(mnrnd(1,Pi));
ct=zeros(T,1);
ct(1)=find(mnrnd(1,mixmat(yt(1),:)));
xt(1,:)=mvnrnd(MU(yt(1),:,ct(1)),SIG(:,:,yt(1),ct(1)));
for t=2:T;
    yt(t)= find(mnrnd(1,A(yt(t-1),:)));
    ct(t)= find(mnrnd(1,mixmat(yt(t),:)));
    xt(t,:)=mvnrnd(MU(yt(t),:,ct(t)),SIG(:,:,ct(t)));
end;
I=eye(K);
pl0=I(yt,:);

% infere
plforinferenceVacuous = ones(T,K);
[p p2] = computeB(xt, parametersHMM.muf, parametersHMM.Sigf, parametersHMM.mixmatf, K, M, size(xt,1));
[alpha, beta, gamma, loglik, xi, ~, gamma2] = fwdback_phmm_mix(parametersHMM.Pif, ....
    parametersHMM.Af, p, plforinferenceVacuous, p2, parametersHMM.mixmatf);

% compare to truth
yn = viterbi_path_phmm(parametersHMM.Pif, parametersHMM.Af, p', plforinferenceVacuous');
disp(sprintf('Performance (in %%) of partially supervised HMM (value of noise = %d%%)',100*rho))
disp(100*RandIndex(yt,yn))
disp('Vary rho to see the impact on ARI')

% infere UNSUPERVISED learning
clear parametersHMM
plforinferenceVacuous = ones(T,K);
[p p2] = computeB(xt, parametersHMMunsup.muf, parametersHMMunsup.Sigf, parametersHMMunsup.mixmatf, K, M, size(xt,1));
[alpha, beta, gamma, loglik, xi, ~, gamma2] = fwdback_phmm_mix(parametersHMMunsup.Pif, ....
    parametersHMMunsup.Af, p, plforinferenceVacuous, p2, parametersHMMunsup.mixmatf);

% compare to truth
yn = viterbi_path_phmm(parametersHMMunsup.Pif, parametersHMMunsup.Af, p', plforinferenceVacuous');
disp(sprintf('Performance (in %%) unsupervised (standard HMM) learning',100*rho))
disp(100*RandIndex(yt,yn))


disp('the impact of rho on results is illustrated in the second example')

