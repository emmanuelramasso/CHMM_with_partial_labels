<<<<<<< Updated upstream:PHMM/phmm_gauss_mix_learn.m
function [parametersHMMGMM, outputsInference] = phmm_gauss_mix_learn(x, pl, nbStates, nbComponents, parametersAlgorithm)
%
% Hidden Markov Models with Multicomponents Gaussian outputs and uncertain and noisy labels
% Also allows a use for estimation of Gaussian Mixture Models with uncertain noisy and noisy labels
% HMM are well known methods to statistically represent time-series.
% An extension is proposed in [1] which allows to take into account partial
% knowledge on hidden states to correct the posterior distributions and their parameters.
%
% This function should be used after phmm_gauss_mix_init.m
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% x is matrix NxF with N data and F features
%
% pl is a matrix of size NxnbStates with N data and nbStates states, made of values in
% [0,1] that allows to give a weight pl(t,k) that data x(t,:) belong more or less to state k.
% pl(t,k)=1 for all t,k then corresponds to unsupervised learning
% If made of 1 and 0 with sum_k pl(t,k)=1, then supervised (and discriminative) learning
%
% nbComponents is the number of components in each state
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
% 	.phmmInit is a structure with initial parameters for the HMM, could be empty [default].
%	Below K=nbStates, M=nbComponents):
% 		phmmInit.mu = means of components size KxFxM, mu(k,:,m) means for state k in all features in component m
% 		phmmInit.sig = covariance of components, size FxFxKxM
% 		phmmInit.mix = mixture weights, size KxM
% 		phmmInit.Pi = initial probability, size 1xK, p(s=i at t=1)
% 		phmmInit.A = transition matrix, size KxK, p(s=i at t-1 | s=j at t)
%
% parametersAlgorithm = setHMMDefaultParameters; allows to set all these values to their default.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs:
% The structure parametersHMMGMM contains the following fields
% 	.muf = means of components size KxFxM after convergence
%
% 	.Sigf =  covariance of components, size FxFxKxM
%
% 	.Pif = initial probability, size 1xK
%
% 	.Af = transition matrix, size KxK
%
% 	.mixmatf = mixture weights, size KxM
%
% outputsInference is a structure with the following fields:
%
% 	.logLmax = likelihood of the model
%
% 	.p = likelihood p(x_t | s_i) for all states s_i
%
% 	.gamma = posterior prob. on states p(s_i | x)
%
% 	.gamma2 = p(s_i | x, m) given component m
%
% 	.p2 = p(x_t|s_i, m)
%
% initmodele = keep in memory the values of initial parameters
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example: used in [1], see example.m
%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: T. Denoeux and E. Ramasso
%
% Reference
% [1] Ramasso, E., & Denoeux, T. (2014).
% Making use of partial knowledge about hidden states in HMMs: an approach
% based on belief functions. Fuzzy Systems, IEEE Transactions on, 22(2), 395-405.
%

nessai = parametersAlgorithm.nessai;
idiag = parametersAlgorithm.idiag;
iltr = parametersAlgorithm.iltr;
visu = parametersAlgorithm.visu;
iplot = parametersAlgorithm.iplot;
init = parametersAlgorithm.init;
thresh = parametersAlgorithm.thresh;
nitermax = parametersAlgorithm.nitermax;
phmmInit = parametersAlgorithm.phmmInit;
K = nbStates;
M = nbComponents;
isHMM = parametersAlgorithm.isHMM;
%%%%%%%%%%%%%%%%%%%%%%%

phmmInitStr = not(isempty(phmmInit));

[T,d]=size(x);

lmax = -inf;

for essai=1:nessai
    
    if parametersAlgorithm.visu
        disp('------------------------------------------')
        disp(sprintf('Essai %d / %d',essai, nessai))
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%% INITIALISATION %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Initialisation using phmmInit
    if phmmInitStr
        
        mu = phmmInit.mu;              
        muf=mu;
        Sig = phmmInit.sig;               
        Sigf=Sig;
        mixmat = phmmInit.mix;       
        mixmatf = mixmat;
        Pi = phmmInit.Pi;                   
        Pif = Pi;
        if isHMM
            A = phmmInit.A;                     
            Af = A;
        else
            A = ones(length(Pi));
        end
        initmodele = phmmInit;
        
    else % init random or make use of prior in pl
        if 1
            [a b]=kmeans(x,K*M); 
            mu=reshape(b,[K,d,M]); %    repmat(mu,[1,1,M]);
            Sigma = zeros(d,d,K*M);
            mixmat = zeros(K*M,1);
            for i=1:K*M
                f=find(a==i);
                Sigma(:,:,i)=cov(x(f,:)) + 100*eye(d);
                mixmat(i) = length(f)/T;
            end
            mixmat = reshape(mixmat,[K,M]);
            Sig = reshape(Sigma,[d,d,K,M]);
        else
            mu=x(randsample(T,K*M),:);
            mu=reshape(mu,[K,d,M]); %    repmat(mu,[1,1,M]);
            mixmat = repmat(ones(1,M)/M,K,1);
        end
                
        u=rand(1,K-1);
        if init
            Pi=pl(1,:)/sum(pl(1,:));
            for k=1:K
                for l=1:K
                    A(k,l)=sum(pl(1:T-1,k).*pl(2:T,l));
                end
                A(k,:)=A(k,:)/sum(A(k,:));
            end
        else
            Pi=diff([0 sort(u) 1]);
            A=zeros(K,K);
            for k=1:K
                u=rand(1,K-1);
                A(k,:)=diff([0 sort(u) 1]);
            end
        end
        
        initmodele.mu = mu;
        initmodele.sig = Sig;
        initmodele.mix = mixmat;
        initmodele.Pi = Pi;
        initmodele.A = A;
        
    end
    if iltr
        A=triu(A);
        A=A./repmat(sum(A,2),1,K);
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%% EM %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    go_on = 1;
    q = 0;
    logL = [];
    err = false; 
    
    while go_on
                
        q=q+1;
        % E step
        try
            
            [p, p2] = computeB(x, mu, Sig, mixmat, K, M, T);
            
            if M>1
                [alpha, beta, gamma, loglik, xi, ~, gamma2] = fwdback_phmm_mix(Pi, A, p, pl, p2, mixmat);
            elseif M==1
                [alpha, beta, gamma, loglik, xi] = fwdback_phmm_mix(Pi, A, p, pl);
                gamma2 = gamma;
            end
                        
            logL=[logL;loglik];
            
            if visu, disp([essai q logL(q)/T]); 
            end
            
            if (q>nitermax) || ((q>1) && ((abs(logL(q)-logL(q-1))/abs(logL(q-1)) < thresh)))
                
                go_on = 0;
            
            else
            
                % M step                
                if isHMM
                    Pi=gamma(1,:);
                    %XI=squeeze(sum(xi,1));
                    XI=xi;
                else
                    A=ones(K,K);
                    Pi=ones(1,K);
                end
                
                S=reshape(sum(gamma2,1),K,M);
                for k=1:K
                    if isHMM
                        A(k,:)=XI(k,:)/sum(XI(k,:));
                    end
                    if M>1 && isHMM, mixmat(k,:) = S(k,:)/sum(S(k,:)); end
                    for m=1:M
                        mu(k,:,m)=sum(x.*repmat(squeeze(gamma2(:,k,m)),1,d))./S(k,m);
                        X=x-repmat(mu(k,:,m),T,1);
                        Sig(:,:,k,m)=(X.*repmat(squeeze(gamma2(:,k,m)),1,d))'*X./S(k,m);
                        if idiag, Sig(:,:,k)=diag(diag(Sig(:,:,k,m))); end
                    end
                end
                
                if iltr
                    A=triu(A);
                    A=A./repmat(sum(A,2),1,K);
                end
                
            end
             
        catch ME
            
            disp(ME)            
            disp('Problem in convergence (possibly nothing, let me continue...)')
            err = true; 
            go_on = false;
            
        end
        
        if parametersAlgorithm.visu
            disp(sprintf('Iteration %d -> Loglik=%f',q,logL(q)))
        end
        
    end      
    
    if err == false
        if logL(end) > lmax
            
            %[p, p2] = computeB(x, mu, Sig, mixmat, K, M, T);            
            muf=mu;
            Sigf=Sig;
            Pif=Pi;
            Af=A;
            mixmatf=mixmat;
        
            lmax=logL(end);
        end        
    end
   
    if iplot, figure;plot(logL);grid; end
    
end


% Set outputs

[p, p2] = computeB(x, muf, Sigf, mixmatf, K, M, T);
if M>1
    [alpha, beta, gamma, loglik, xi, ~, gamma2] = fwdback_phmm_mix(Pif, Af, p, pl, p2, mixmatf);
elseif M==1
    [alpha, beta, gamma, loglik, xi] = fwdback_phmm_mix(Pif, Af, p, pl);
    gamma2 = gamma;
end

parametersHMMGMM.muf = muf; clear muf
parametersHMMGMM.Sigf = Sigf; clear Sigf
parametersHMMGMM.Pif = Pif; clear Pif
parametersHMMGMM.Af = Af; clear Af
parametersHMMGMM.mixmatf = mixmatf; clear mixmatf
parametersHMMGMM.initmodele = initmodele; clear initmodele
parametersHMMGMM.plausibilites = pl;

outputsInference.logLmax = lmax; clear lmax
outputsInference.p = p; clear p
outputsInference.gamma = gamma; clear gamma
outputsInference.alpha = alpha; clear alpha
outputsInference.gamma2 = gamma2; clear gamma2
outputsInference.p2 = p2; clear p2






=======
function [parametersHMMGMM, outputsInference] = phmm_gauss_mix_learn(x, pl, nbStates, nbComponents, parametersAlgorithm)
%
% Hidden Markov Models with Multicomponents Gaussian outputs and uncertain and noisy labels
% Also allows a use for estimation of Gaussian Mixture Models with uncertain noisy and noisy labels
% HMM are well known methods to statistically represent time-series.
% An extension is proposed in [1] which allows to take into account partial
% knowledge on hidden states to correct the posterior distributions and their parameters.
%
% This function should be used after phmm_gauss_mix_init.m
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% x is matrix NxF with N data and F features
%
% pl is a matrix of size NxnbStates with N data and nbStates states, made of values in
% [0,1] that allows to give a weight pl(t,k) that data x(t,:) belong more or less to state k.
% pl(t,k)=1 for all t,k then corresponds to unsupervised learning
% If made of 1 and 0 with sum_k pl(t,k)=1, then supervised (and discriminative) learning
%
% nbComponents is the number of components in each state
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
% 	.phmmInit is a structure with initial parameters for the HMM, could be empty [default].
%	Below K=nbStates, M=nbComponents):
% 		phmmInit.mu = means of components size KxFxM, mu(k,:,m) means for state k in all features in component m
% 		phmmInit.sig = covariance of components, size FxFxKxM
% 		phmmInit.mix = mixture weights, size KxM
% 		phmmInit.Pi = initial probability, size 1xK, p(s=i at t=1)
% 		phmmInit.A = transition matrix, size KxK, p(s=i at t-1 | s=j at t)
%
% parametersAlgorithm = setHMMDefaultParameters; allows to set all these values to their default.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs:
% The structure parametersHMMGMM contains the following fields
% 	.muf = means of components size KxFxM after convergence
%
% 	.Sigf =  covariance of components, size FxFxKxM
%
% 	.Pif = initial probability, size 1xK
%
% 	.Af = transition matrix, size KxK
%
% 	.mixmatf = mixture weights, size KxM
%
% outputsInference is a structure with the following fields:
%
% 	.logLmax = likelihood of the model
%
% 	.p = likelihood p(x_t | s_i) for all states s_i
%
% 	.gamma = posterior prob. on states p(s_i | x)
%
% 	.gamma2 = p(s_i | x, m) given component m
%
% 	.p2 = p(x_t|s_i, m)
%
% initmodele = keep in memory the values of initial parameters
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example: used in [1], see example.m
%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: T. Denoeux and E. Ramasso
%
% Reference
% [1] Ramasso, E., & Denoeux, T. (2014).
% Making use of partial knowledge about hidden states in HMMs: an approach
% based on belief functions. Fuzzy Systems, IEEE Transactions on, 22(2), 395-405.
%

nessai = parametersAlgorithm.nessai;
idiag = parametersAlgorithm.idiag;
iltr = parametersAlgorithm.iltr;
visu = parametersAlgorithm.visu;
iplot = parametersAlgorithm.iplot;
init = parametersAlgorithm.init;
thresh = parametersAlgorithm.thresh;
nitermax = parametersAlgorithm.nitermax;
phmmInit = parametersAlgorithm.phmmInit;
K = nbStates;
M = nbComponents;
isHMM = parametersAlgorithm.isHMM;
%%%%%%%%%%%%%%%%%%%%%%%

phmmInitStr = not(isempty(phmmInit));

[T,d]=size(x);

lmax = -inf;

for essai=1:nessai
    
    if parametersAlgorithm.visu
        disp('------------------------------------------')
        disp(sprintf('Essai %d / %d',essai, nessai))
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%% INITIALISATION %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Initialisation using phmmInit
    if phmmInitStr
        
        mu = phmmInit.mu;              
        muf=mu;
        Sig = phmmInit.sig;               
        Sigf=Sig;
        mixmat = phmmInit.mix;       
        mixmatf = mixmat;
        Pi = phmmInit.Pi;                   
        Pif = Pi;
        if isHMM
            A = phmmInit.A;                     
            Af = A;
        else
            A = ones(length(Pi));
        end
        initmodele = phmmInit;
        
    else % init random or make use of prior in pl
        if 1
            [a b]=kmeans(x,K*M); 
            mu=reshape(b,[K,d,M]); %    repmat(mu,[1,1,M]);
            Sigma = zeros(d,d,K*M);
            mixmat = zeros(K*M,1);
            for i=1:K*M
                f=find(a==i);
                Sigma(:,:,i)=cov(x(f,:)) + 100*eye(d);
                mixmat(i) = length(f)/T;
            end
            mixmat = reshape(mixmat,[K,M]);
            Sig = reshape(Sigma,[d,d,K,M]);
        else
            mu=x(randsample(T,K*M),:);
            mu=reshape(mu,[K,d,M]); %    repmat(mu,[1,1,M]);
            mixmat = repmat(ones(1,M)/M,K,1);
        end
                
        u=rand(1,K-1);
        if init
            Pi=pl(1,:)/sum(pl(1,:));
            for k=1:K
                for l=1:K
                    A(k,l)=sum(pl(1:T-1,k).*pl(2:T,l));
                end
                A(k,:)=A(k,:)/sum(A(k,:));
            end
        else
            Pi=diff([0 sort(u) 1]);
            A=zeros(K,K);
            for k=1:K
                u=rand(1,K-1);
                A(k,:)=diff([0 sort(u) 1]);
            end
        end
        
        initmodele.mu = mu;
        initmodele.sig = Sig;
        initmodele.mix = mixmat;
        initmodele.Pi = Pi;
        initmodele.A = A;
        
    end
    if iltr
        A=triu(A);
        A=A./repmat(sum(A,2),1,K);
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%% EM %%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    go_on = 1;
    q = 0;
    logL = [];
    err = false; 
    
    while go_on
                
        q=q+1;
        % E step
        try
            
            [p, p2] = computeB(x, mu, Sig, mixmat, K, M, T);
            
            if M>1
                [alpha, beta, gamma, loglik, xi, ~, gamma2] = fwdback_phmm_mix(Pi, A, p, pl, p2, mixmat);
            elseif M==1
                [alpha, beta, gamma, loglik, xi] = fwdback_phmm_mix(Pi, A, p, pl);
                gamma2 = gamma;
            end
                        
            logL=[logL;loglik];
            
            if visu, disp([essai q logL(q)/T]); 
            end
            
            if (q>nitermax) || ((q>1) && ((abs(logL(q)-logL(q-1))/abs(logL(q-1)) < thresh)))
                
                go_on = 0;
            
            else
            
                % M step                
                if isHMM
                    Pi=gamma(1,:);
                    %XI=squeeze(sum(xi,1));
                    XI=xi;
                else
                    A=ones(K,K);
                    Pi=ones(1,K);
                end
                
                S=reshape(sum(gamma2,1),K,M);
                for k=1:K
                    if isHMM
                        A(k,:)=XI(k,:)/sum(XI(k,:));
                    end
                    if M>1 && isHMM, mixmat(k,:) = S(k,:)/sum(S(k,:)); end
                    for m=1:M
                        mu(k,:,m)=sum(x.*repmat(squeeze(gamma2(:,k,m)),1,d))./S(k,m);
                        X=x-repmat(mu(k,:,m),T,1);
                        Sig(:,:,k,m)=(X.*repmat(squeeze(gamma2(:,k,m)),1,d))'*X./S(k,m);
                        if idiag, Sig(:,:,k)=diag(diag(Sig(:,:,k,m))); end
                    end
                end
                
                if iltr
                    A=triu(A);
                    A=A./repmat(sum(A,2),1,K);
                end
                
            end
             
        catch ME
            
            disp(ME)            
            disp('Problem in convergence (possibly nothing, let me continue...)')
            err = true; 
            go_on = false;
            
        end
        
        if parametersAlgorithm.visu
            disp(sprintf('Iteration %d -> Loglik=%f',q,logL(q)))
        end
        
    end      
    
    if err == false
        if logL(end) > lmax
            
            %[p, p2] = computeB(x, mu, Sig, mixmat, K, M, T);            
            muf=mu;
            Sigf=Sig;
            Pif=Pi;
            Af=A;
            mixmatf=mixmat;
        
            lmax=logL(end);
        end        
    end
   
    if iplot, figure;plot(logL);grid; end
    
end


% Set outputs

[p, p2] = computeB(x, muf, Sigf, mixmatf, K, M, T);
if M>1
    [alpha, beta, gamma, loglik, xi, ~, gamma2] = fwdback_phmm_mix(Pif, Af, p, pl, p2, mixmatf);
elseif M==1
    [alpha, beta, gamma, loglik, xi] = fwdback_phmm_mix(Pif, Af, p, pl);
    gamma2 = gamma;
end

parametersHMMGMM.muf = muf; clear muf
parametersHMMGMM.Sigf = Sigf; clear Sigf
parametersHMMGMM.Pif = Pif; clear Pif
parametersHMMGMM.Af = Af; clear Af
parametersHMMGMM.mixmatf = mixmatf; clear mixmatf
parametersHMMGMM.initmodele = initmodele; clear initmodele
parametersHMMGMM.plausibilites = pl;

outputsInference.logLmax = lmax; clear lmax
outputsInference.p = p; clear p
outputsInference.gamma = gamma; clear gamma
outputsInference.alpha = alpha; clear alpha
outputsInference.gamma2 = gamma2; clear gamma2
outputsInference.p2 = p2; clear p2






>>>>>>> Stashed changes:phmm_gauss_mix_learn.m
