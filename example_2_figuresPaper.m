%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation made in section IV/A/2: Influence of noise on labels
% Figure 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all

% Model: 3 states, data in 3 dimensions
K=3;
d=3;
M=1;
MU=2*[1 0 0;0 1 0; 0 0 1]; % means
SIG=[1 1 1 % cov
    1 1 1
    1 1 1];
Pi=ones(K,1)/K; % vector of initial prob in HMM
A=[0.6 0.3 0.1 % matrix of transitions
    0.1 0.6 0.3
    0.1 0.3 0.6];

% loop, with different sequences

for iter=1:30,
    
    % Generation of data for learning
    T=100;
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
    
    % unsupervised learning
    parametersAlgorithm = setHMMDefaultParameters;
    parametersAlgorithm.nitermax = 500;
    
    %%%
    % CALL
    it = 1;
    while it<20
        try
            [parametersHMMh, outputsInferenceh] = ...
                phmm_gauss_mix_learn(x, ones(T,K), K, M, parametersAlgorithm);
    
            for u=1:K, for m=1:M, [R,err] = cholcov(parametersHMMh.Sigf(:,:,u,m),0);
                    % SIGMA must be a square, symmetric, positive definite matrix.
                    if err~0, disp('go to catch...'), error('pb of convergence'), end, end
            end
            break
        catch
            if it+1>20, error('Impossible to run, look in data (nan ? inf ?) or normalize using zscore ?')
            else it=it+1; % retry...
            end
        end
    end
    %%%
    
    % inference
    for k=1:K,
        pt(:,k)=mvnpdf(xt,parametersHMMh.muf(k,:),parametersHMMh.Sigf(:,:,k));
    end;
    
    yn = viterbi_path_phmm(parametersHMMh.Pif, parametersHMMh.Af, pt', ones(K,Tt))';
    
    % perf
    [ARn(iter),RIn(iter)]=RandIndex(yt,yn);
    
    
    % Noise on labels
    clear parametersHMMh outputsInferenceh
    rho=(0:0.1:1);
    for ii=1:length(rho),
        
        if rho(ii)==0,
            perr=zeros(T,1);
        elseif rho(ii)==1,
            perr=ones(T,1);
        else,
            [a,b]=param_beta(rho(ii),(0.2).^2);
            perr=betarnd(a,b,T,1);
        end;
        [pl,y1,pl1]=add_noise1(y,perr,K);
        
        
        parametersAlgorithm.init=true; % use pl to initialize
        
        %%%%%%%%%%%%%%%%%%%%
        % Uncertain labels => in pl
        % CALL
        it = 1;
        while it<20
            try
                [parametersHMMh, outputsInferenceh] = ...
                    phmm_gauss_mix_learn(x, pl, K, M, parametersAlgorithm);
                
                for u=1:K, for m=1:M, [R,err] = cholcov(parametersHMMh.Sigf(:,:,u,m),0);
                        % SIGMA must be a square, symmetric, positive definite matrix.
                        if err~0, disp('go to catch...'), error('pb of convergence'), end, end
                end
                break
            catch
                if it+1>20, error('Impossible to run, look in data (nan ? inf ?) or normalize using zscore ?')
                else it=it+1; % retry...
                end
            end
        end
        %%%
        
        %%%%%%%%%%%%%%%
        % Noisy labels in pl1
        %%%
        %%%
        % CALL
        it = 1;
        while it<20
            try
                [parametersHMMh1, outputsInferenceh1] = ...
                    phmm_gauss_mix_learn(x, pl1, K, M, parametersAlgorithm);
                
                for u=1:K, for m=1:M, [R,err] = cholcov(parametersHMMh1.Sigf(:,:,u,m),0);
                        % SIGMA must be a square, symmetric, positive definite matrix.
                        if err~0, disp('go to catch...'), error('pb of convergence'), end, end
                end
                break
            catch
                if it+1>20, error('Impossible to run, look in data (nan ? inf ?) or normalize using zscore ?')
                else it=it+1; % retry...
                end
            end
        end
        %%%
        
        % test inference
        for k=1:K,
            pt(:,k)=mvnpdf(xt,parametersHMMh.muf(k,:),parametersHMMh.Sigf(:,:,k));% uncertain
            pt1(:,k)=mvnpdf(xt,parametersHMMh1.muf(k,:),parametersHMMh1.Sigf(:,:,k));% noisy
        end;
        yth = viterbi_path_phmm(parametersHMMh.Pif, parametersHMMh.Af, pt', ones(K,Tt))';
        yth1 = viterbi_path_phmm(parametersHMMh1.Pif, parametersHMMh1.Af, pt1', ones(K,Tt))';
        
        % perfo
        [AR(ii,iter),RI(ii,iter)]=RandIndex(yt,yth);
        [AR1(ii,iter),RI1(ii,iter)]=RandIndex(yt,yth1);
        
        disp([rho(ii) iter AR(ii,iter) AR1(ii,iter)])
    end;
end;


figure(1)
clf

u=(0:0.1:1)+0.01*randn(1,11);
u(1)=0;u(end)=1;

errorbar(u, mean(AR'),std(AR'),'-ro')
hold on
errorbar(0:0.1:1, mean(AR1'),std(AR1'),'-bs')
plot(rho,repmat(mean(ARn),length(rho),1),'--');
axis([0 1 0 0.8])
xlabel('\rho')
ylabel('adjusted Rand index')
legend('uncertain labels','noisy labels','unsupervised')
grid




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment 2: use of labels for segmentation
% Figure 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

M=1;
K=3;
d=3;
MU=2*[1 0 0;0 1 0; 0 0 1];
SIG=[1 1 1
     1 1 1
     1 1 1];
Pi=ones(K,1)/K;
A=[0.6 0.3 0.1
   0.1 0.6 0.3
   0.1 0.3 0.6];


for iter=1:30
    
    % Generation of data for learning
   T=300;
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
    pl=max(pl0,0.9);
    
    %test    
    Tt=1000;
    xt=zeros(Tt,d);
    yt=zeros(Tt,1);
    yt(1)=find(mnrnd(1,Pi));
    xt(1,:)=mvnrnd(MU(yt(1),:),SIG(yt(1),:));
    for t=2:Tt;
        yt(t)= find(mnrnd(1,A(yt(t-1),:)));
        xt(t,:)=mvnrnd(MU(yt(t),:),SIG(yt(t),:));
    end;
    I=eye(K);
    plt0=I(yt,:);
    
    % unsupervised learning
    parametersAlgorithm = setHMMDefaultParameters;
    parametersAlgorithm.init = true; % makes use of knowledge
    
%     nessai=10;
%     visu=0;
%     iplot=0;
%     idiag=0;
%     iltr=0;
%     init=1;
%     [parametersHMMh.muf,parametersHMMh.Sigf,parametersHMMh.Pif,parametersHMMh.Af,...
%     logLmax,p,gamma]=phmm_gauss(x,pl,nessai,idiag,iltr,visu,iplot,init);
    
    %%%
    clear parametersHMMh
    % CALL
    it = 1;
    while it<20
        try
            [parametersHMMh, outputsInferenceh] = ...
                phmm_gauss_mix_learn(x, pl, K, M, parametersAlgorithm);
    
            for u=1:K, for m=1:M, [R,err] = cholcov(parametersHMMh.Sigf(:,:,u,m),0);
                    % SIGMA must be a square, symmetric, positive definite matrix.
                    if err~0, disp('go to catch...'), error('pb of convergence'), end, end
            end
            break
        catch
            if it+1>20, error('Impossible to run, look in data (nan ? inf ?) or normalize using zscore ?')
            else it=it+1; % retry...
            end
        end
    end
    %%
    
    rho=(0:0.1:1);ns=rho;
    %ns=(0:0.05:1);rho=ns;
    for ii=1:length(rho),
        
        % noise on test labels
        
        if rho(ii)==0,
            perr=zeros(Tt,1);
        elseif rho(ii)==1,
            perr=ones(Tt,1);
        else,
            [a,b]=param_beta(rho(ii),(0.2).^2);
            perr=betarnd(a,b,Tt,1);
        end;
        [plt,y1,plt1]=add_noise1(yt,perr,K);
        
        pt = zeros(Tt,K);
        for k=1:K,
            pt(:,k)=mvnpdf(xt,parametersHMMh.muf(k,:),parametersHMMh.Sigf(:,:,k));
        end;
        
        yn = viterbi_path_phmm(parametersHMMh.Pif, parametersHMMh.Af, pt', plt');
        
        [AR(ii,iter),RI(ii,iter)]=RandIndex(yt,yn);
        
        [AR1(ii,iter),RI1(ii,iter)]=RandIndex(yt,y1);
        disp([ns(ii) iter AR(ii,iter) AR1(ii,iter)])
        
    end;
end;

jj=find(AR(1,:)>0.1);
figure(3)
clf
errorbar(ns, mean(AR(:,jj)'),std(AR(:,jj)'),'-ro')
hold on
errorbar(ns, mean(AR1(:,jj)'),std(AR1(:,jj)'),'-bs')

axis([0 max(ns) 0 1])
xlabel('\rho')
ylabel('adjusted Rand index')
legend('uncertain labels','noisy labels')
grid

