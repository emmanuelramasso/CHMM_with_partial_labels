function [alpha, beta, gamma, loglik, xi, C, gamma2] = ...
    fwdback_phmm_mix(Pi, A, p, pl, B2, mixmat)

% inputs :
% p [T Q] ;  p(t,k)=p(x_t | Y_t=k)
% Pi [1 Q] : Pi(k)=p(Y_1=k)
% A [Q Q] ; A(k,ell)=P(Y_t=ell  Y_{t-1}=k)
% pl [T Q] ; pl(t,k)=Pl(y_t=k)


if nargin<7, fwdonly=false; end


% Alloc
[T,Q]=size(p);% sizes
C = ones(T,1);% normalization coefficients
alpha = zeros(T,Q);% forward
if ~fwdonly
    beta = zeros(T,Q); % backward
    gamma = zeros(T,Q);% gamma(t,k)=P(Y_t=k | x,pl)
    % xi = zeros(T,Q,Q); % xi(t,k,ell)=P(Y_{t}=k, Y_{t+1}=ell | x,pl)
    xi = zeros(Q,Q); % xi(k,ell)=P(Y_{t}=k, Y_{t+1}=ell | x,pl)
end

p_pl=p.*pl; clear p

% Compute gamma2 ?
compGamma2 = false;
if nargin>4 && ndims(B2)==3 && nargout==7
    [~,~,M]=size(B2);
    B2 = repmat(pl,[1 1 M]).*B2;
    denom = p_pl + (p_pl==0); % replace 0s with 1s before dividing
    gamma2 = zeros(T,Q, M);% gamma(t,k,m)=P(Y_t=k, c_t=m | x,pl)
    compGamma2 = true;
else gamma2=[];
end


% ################################
%            Forward
% ################################

% --------------------------------
% init
%keyboard
alpha(1,:) = Pi .* p_pl(1,:); % (13.37)
C(1)=sum(alpha(1,:));
alpha(1,:)=alpha(1,:)/C(1);


% --------------------------------
% Propagate
for t=2:T
    alpha(t,:) =  (p_pl(t,:)) .* (alpha(t-1,:)*A) ; % (13.36)
    C(t)=sum(alpha(t,:));
    alpha(t,:)=alpha(t,:)/C(t);%(C(t)+(C(t)==0)); % (13.59)
end


loglik = sum(log(C)); % (13.63)


% ################################
%            Backward
% ################################


if fwdonly
    beta = [];
    gamma = [];
    xi = [];
    return
end


% --------------------------------
% Init
% beta
beta(T,:) = ones(1,Q);
gamma(T,:) = alpha(T,:);% .* beta(T,:);

% --------------------------------
for t=T-1:-1:1
    
    beta(t,:) = (beta(t+1,:) .* p_pl(t+1,:)) * A' ; % (13.38)
    beta(t,:)=beta(t,:)/C(t+1); % (13.62)
    
    gamma(t,:) = alpha(t,:) .* beta(t,:);   % (13.64)
    
    %xi(t,:,:) = (alpha(t,:)' * (p_pl(t+1,:) .* beta(t+1,:))) .* A; % (13.65)
    %xi(t,:,:) = xi(t,:,:)/C(t+1);
    xi = xi + ((alpha(t,:)' * (p_pl(t+1,:) .* beta(t+1,:))) .* A)/C(t+1); % (13.65)
     
    % Rabiner Eq. post 54
    if compGamma2
        gamma2(t,:,:) = reshape(reshape(B2(t,:,:),Q,M) .* mixmat .* repmat(gamma(t,:)', [1 M]) ./ repmat(denom(t,:)',  [1 M]),[1 Q M]);
    end
end
% --------------------------------

if nargout>6 && isempty(gamma2), gamma2=gamma; end