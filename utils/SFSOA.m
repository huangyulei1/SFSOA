function [X_new,W,G,P,F,i,obj,score1] = SFSOA(X,Niter,k,alpha,lamda,gamma,l)
% SFSOA: Soft-label Guided Unsupervised Feature Selection with Orthogonal Anchor Basis.
% 
% Input:
%   X       - Data matrix (d*n). Each column vector of data is a sample vector.
%   Niter   - The maximum number of iterations
%   k       - The parameter for generating the original anchor graph G. [4 5 6 7 8]
%   alpha   - The orthogonal constraint parameter. [1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3 1e4]
%   lamda   - The sparse constraint parameter. [1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3 1e4]
%   gamma   - The scale parameter. [1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3 1e4]
%   l       - The number of featuers to be selected
%
% Output:
%   X_new   - The selected subset of features (l*n).
%--------------------------------------------------------------------------
%    Examples:
%       load('COIL20.mat');
%       rng('default');
%       X = NormalizeFea(fea);
%       nClusts = length(unique(gnd)); c = nClusts; Niter = 30; 
%       k = 4; alpha = 1e0; gamma = 1e-3; lamda = 1e-4; mm = 10; l = 80;
%       [X_new,W,G,P,F,iter,obj,score] = SFSOA(X',Niter,k,alpha,lamda,gamma,l);
%       X_new is the final feature subset.


% Initialization
[N_row,N_column] = size(X);
K = 2^k-1;
[~,locAnchor] = hKM(X,[1:N_column],k,1);
G = ConstructA_NP(X, locAnchor,K); G = G'; [m,~] = size(G);
P = G ;
W = abs(rand(N_row,m));
F = orth(rand(m,m));
[N_rowF,N_columnF] = size(F);
if N_rowF < m
    F = abs(rand(m,m));
end
if N_columnF < m
    F = abs(rand(m,m));
end

% Optimization
for i = 1:Niter
    L_k = diag(sum(P,1));
    L_i = diag(sum(P,2));
    
    Pi = sqrt(sum(W.*W,2)+eps);
    d = 0.5./Pi;
    Q = diag(d);
    
%     Update F
    F = F.*((1+gamma)*W'*X*P' + 2*alpha*F)./(F*P*P' + gamma*F*L_i + 2*alpha*F*F'*F + eps);
    
%     Update W
    W = W.*((1+gamma)*X*P'*F')./(X*X'*W + gamma*X*L_k*X'*W + lamda*Q*W);

%     Update P
    M = W'*X;
    distx = L2_distance_subfun((M),(F));
    distx = distx';
    r_i = F'*M;
    for ii = 1:N_column
        idxa0 = 1:m;
        dxi = distx(idxa0,ii);
        ad = (r_i(idxa0,ii) + G(idxa0,ii) - gamma*(dxi)/(2))/(2);
        P(idxa0,ii) = EProjSimplex_new(ad,1);
    end
    col_sum = sum(P,1);  
    P = P./repmat(col_sum,m,1);
    
%     Convergence of judgment
    obj(i) = trace(M*M' - 2*M*P'*F' + F*P*P'*F');
    if i > 2
        if abs(obj(i) - obj(i - 1))/obj(i - 1) < 0.01
            break
        end
    end

end
score1 = sqrt(sum(W.*W,2));
[score, idx] = sort(score1,'descend');
X_new = X (idx(1:l),:);
end