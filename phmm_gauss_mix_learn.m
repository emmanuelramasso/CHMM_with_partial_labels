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
%	.hmmOrgmm [default='hmm'] = 'hmm' or 'gmm' for HMM or GMM model
%
% 	VIDEOstruct is a structure for creating a video to display the
% 	convergence, will plot components in each state, empty if not used [default=empty]
% 		.doVideo = true or false
% 		.groundTruth = vector of labels, the same size as x
% 		.colorCenters = colors we want to see, size Kx3
%	all fields must be given for VIDEOstruct if not empty.
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
hmmOrgmm = parametersAlgorithm.hmmOrgmm;
VIDEOstruct = parametersAlgorithm.VIDEOstruct;
phmmInit = parametersAlgorithm.phmmInit;
K = nbStates;
M = nbComponents;
%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(upper(hmmOrgmm),'HMM'), isHMM=true;
elseif strcmp(upper(hmmOrgmm),'GMM'), isHMM=false;
else error('Model unknown')
end

phmmInitStr = not(isempty(phmmInit));

[T,d]=size(x);

logLmax=-inf;

if isempty(VIDEOstruct)
    VIDEOstruct.doVideo = false;
end

VIDEO = VIDEOstruct.doVideo;
if VIDEO, ccl = VIDEOstruct.colorCenters; end

for essai=1:nessai,
    
    if parametersAlgorithm.visu
        disp('------------------------------------------')
        disp(sprintf('Essai %d / %d',essai, nessai))
    end
    
    if VIDEO
        vidObj = VideoWriter('vv.avi');%,'quality',70,'FrameRate',25);
        open(vidObj);
    end
    
    % Initialisation using phmmInit
    if phmmInitStr
        mu = phmmInit.mu;              muf=mu;
        Sig = phmmInit.sig;               Sigf=Sig;
        mixmat = phmmInit.mix;       mixmatf = mixmat;
        Pi = phmmInit.Pi;                   Pif = Pi;
        if isHMM
            A = phmmInit.A;                     Af = A;
        else
            A = ones(length(Pi));
        end
        %gamma = phmmInit.gamma;
        logLmax = -inf;
        %gamma2 = phmmInit.gamma2;
        initmodele = phmmInit;
        
    else % init random or make use of prior in pl
        mu=x(randsample(T,K*M),:);
        mu=reshape(mu,[K,d,M]); %    repmat(mu,[1,1,M]);
        mixmat = repmat(ones(1,M)/M,K,1);
        %         for i=1:K, for j=1:M,
        %                 nn = randsample(T,min(T,25));
        %                 Sig(:,:,i,j)=cov(x(nn,:));
        %                 if idiag, Sig(:,:,i,j)=diag(diag(Sig(:,:,i,j)));
        %                 end
        %             end,
        %         end
        Sigma=cov(x);
        %if idiag, Sigma=diag(diag(Sigma)); end
        Sig=repmat(Sigma,[1 1 K M]);
        
        u=rand(1,K-1);
        if init,
            Pi=pl(1,:)/sum(pl(1,:));
            for k=1:K
                for l=1:K,
                    A(k,l)=sum(pl(1:T-1,k).*pl(2:T,l));
                end;
                A(k,:)=A(k,:)/sum(A(k,:));
            end;
        else
            Pi=diff([0 sort(u) 1]);
            A=zeros(K,K);
            for k=1:K,
                u=rand(1,K-1);
                A(k,:)=diff([0 sort(u) 1]);
            end
        end;
        
        initmodele.mu = mu;
        initmodele.sig = Sig;
        initmodele.mix = mixmat;
        initmodele.Pi = Pi;
        initmodele.A = A;
        
    end
    if iltr,
        A=triu(A);
        A=A./repmat(sum(A,2),1,K);
    end;
    
    go_on=1;
    q=0;
    logL=[];
    
    while go_on,
                
        q=q+1;
        % E step
        try
            
            [p p2] = computeB(x, mu, Sig, mixmat, K, M, T);
            
            if M>1
                [alpha, beta, gamma, loglik, xi, ~, gamma2] = fwdback_phmm_mix(Pi, A, p, pl, p2, mixmat);
            elseif M==1
                [alpha, beta, gamma, loglik, xi] = fwdback_phmm_mix(Pi, A, p, pl);
                gamma2 = gamma;
            end
            
            %             if ~isHMM
            %                 post = (ones(T, 1)*mixmat').*gamma;
            %                 s = sum(post, 2);
            %                 if any(s==0)
            %                     warning('Some zero posterior probabilities')
            %                     zero_rows = find(s==0);
            %                     s = s + (s==0);
            %                     post(zero_rows, :) = 1/K;
            %                 end
            %                 gamma = post./(s*ones(1, K));
            %             end
            
            logL=[logL;loglik];
            
            if visu, disp([essai q logL(q)/T]); end;
            
            if (q>nitermax) || ((q>1) && ((abs(logL(q)-logL(q-1))/abs(logL(q-1)) < thresh))),
                go_on=0;
            
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
                for k=1:K,
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
                
                
                if iltr,
                    A=triu(A);
                    A=A./repmat(sum(A,2),1,K);
                end;
                
                if VIDEO
                    if isfield(VIDEOstruct,'groundTruth')
                        yy = VIDEOstruct.groundTruth;
                        f=plotmatrix_mine(x,yy,1,1);
                        title(sprintf('Trial %d -> Iteration %d',essai,q))
                    else f=figure; plot(x(:,1),x(:,2),'.','MarkerSize',14);
                    end
                    hold on,
                    for i=1:size(mu,1),
                        for m=1:M
                            plot(mu(i,1,m),mu(i,2,m),'s','Color',ccl(i,:),'MarkerSize',14,'LineWidth',4),
                            h = draw_ellipse(mu(i,1:2,m)', Sig(1:2,1:2,i,m), [.2 .2 .2]);
                        end, end
                    %improve_figure
                    %figure_pdf_cropped(gcf,sprintf('phmm_%d',q))
                    currFrame = getframe;
                    writeVideo(vidObj,currFrame);
                    %close(f)
                    %pause(0.5)
                end
                
                if logL(q)>logLmax,
                    [p p2] = computeB(x, mu, Sig, mixmat, K, M, T);
                    
                    muf=mu;
                    Sigf=Sig;
                    Pif=Pi;
                    Af=A;
                    mixmatf=mixmat;
                    
                    logLmax=logL(q);
                    
                end;
            end
        
        catch ME
            %ME
            go_on=0;
            logL(q)=logLmax;
            if parametersAlgorithm.visu
                disp('Problem in convergence (possibly nothing, let me continue...)')
            end
        end
        
        if parametersAlgorithm.visu
            disp(sprintf('Iteration %d -> Loglik=%f',q,logL(q)))
        end
        
    end
      
    
    if VIDEO, close(vidObj); end
    
    
    if iplot, figure;plot(logL);grid; end
    
end;


% Set outputs
parametersHMMGMM.muf = muf; clear muf
parametersHMMGMM.Sigf = Sigf; clear Sigf
parametersHMMGMM.Pif = Pif; clear Pif
parametersHMMGMM.Af = Af; clear Af
parametersHMMGMM.mixmatf = mixmatf; clear mixmatf
parametersHMMGMM.initmodele = initmodele; clear initmodele
parametersHMMGMM.plausibilites = pl;

outputsInference.logLmax = logLmax; clear logLmax
outputsInference.p = p; clear p
outputsInference.gamma = gamma; clear gamma
outputsInference.alpha = alpha; clear alpha
outputsInference.gamma2 = gamma2; clear gamma2
outputsInference.p2 = p2; clear p2






