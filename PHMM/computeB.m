function [p p2] = computeB(x, mu, Sig, mixmat, K, M, T)

if M>1
    % Mixture per state, Eq. post 54 Rabiner
    p2=zeros(T,K,M);
    for k=1:K
        for m=1:M
            try
                p2(:,k,m) = mvnpdf(x,mu(k,:,m),Sig(:,:,k,m));
            catch ME
                disp([k,m]), error(ME);
            end
        end
    end
    p=zeros(T,K);
    for t=1:T
        p(t,:) = sum(mixmat .* reshape(p2(t,:,:),K,M), 2);
    end
else
    % single component
    p=zeros(T,K);
    for k=1:K
        try
            p(:,k) = mvnpdf(x,mu(k,:),Sig(:,:,k));
        catch ME
            disp([k]), error(ME);
        end
    end
    p2 = [];
end