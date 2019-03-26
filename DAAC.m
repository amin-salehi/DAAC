%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.

function [U, H, loss] = DAAC(X, R, k, lambda, niter)
% Inputs:
%   X: Attitude matrix
%   R: Social interaction matrix
%   k: The number of communities
%   lambda: The regularization parameter controlling the contribution of the
%   graph regularizer
%   niter: The number of iterations
% Return:
%   U: Community membership matrix
%   V: Community profile matrix
%   loss: The loss


% default configuration for line search
a = 0.4;
b = 0.9;
beta = 0;


minVal = 1e-1000;
[n, n] = size(R);
U = rand(n,k);
H = rand(k,k);
I_k = rand(k,k);

D = diag(full(sum(R,2)));
D_ = D.^(-0.5);
D_(isinf(D_)) = 0;
Z = D_*R*D_;

W = abs(X)>0;

W2 = W' .* W;
W2M = W2 .* X;
W2M_T = (W2M)';

lambda_Z = lambda * Z;


for it = 1:niter
    
    U_T = U';
    UH = U*H;
    UH_T = U*H';
    UHU_T = U*H*U_T;
    
    W2MUH_T = - W2M * UH_T;
    W2M_TUH = - W2M_T * UH;
    W2UHU_T = W2 .* UHU_T;
    W2UHU_TUH_T = W2UHU_T * UH_T;
    W2UHU_T_TUH = (W2UHU_T)' * UH;
    
    lambda_ZU = lambda_Z * U;
    
    [lag_pos, lag_neg] = PosNegSeperation(-U_T * W2MUH_T - U_T * W2M_TUH - U_T * W2UHU_TUH_T - U_T * W2UHU_T_TUH + U_T * lambda_ZU);
    
    [W2MUH_T_pos, W2MUH_T_neg] = PosNegSeperation(W2MUH_T);
    [W2M_TUH_pos, W2M_TUH_neg] = PosNegSeperation(W2M_TUH);
    [W2UHU_TUH_T_pos, W2UHU_TUH_T_neg] = PosNegSeperation(W2UHU_TUH_T);
    [W2UHU_T_TUH_pos, W2UHU_T_TUH_neg] = PosNegSeperation(W2UHU_T_TUH);

    U = U.*(((W2MUH_T_pos + W2M_TUH_pos + W2UHU_TUH_T_neg + W2UHU_T_TUH_neg + lambda_ZU + U*lag_neg)./max((W2MUH_T_neg + W2M_TUH_neg + W2UHU_TUH_T_pos + W2UHU_T_TUH_pos + U*lag_pos),minVal)).^(0.5)); 
    U(isnan(U))=0;
    U(isinf(U))=0;
    

     U_T = U';
     H_deriv =  -2 * U_T * W2M * U + 2 * U_T * (W2 .* (U*H*U_T)) * U;
     alpha = LineSearch(X, Z, W, U, H, lambda, beta, -1*H_deriv, H_deriv, a, b,I_k);
	H = H - alpha * H_deriv;
    
    H(isnan(H))=0;
    H(isinf(H))=0;
end

loss = norm(W.* (Z - U*H*U'), 'fro')^2 + lambda*trace(U'*Z*U)
 
function [ A_pos, A_neg ] = PosNegSeparation( A )
    A_abs = abs(A);
    A_pos = (A_abs + A) / 2;
    A_neg = (A_abs - A) / 2;
end

function t = LineSearch(M, Z, W, U, V, alpha, beta, searchDir, deriv, a, b, I_k)

    function result = f(M, Z,W, U, V, alpha, beta, I_k)
        result = norm(W.*(M - U*V*U'), 'fro')^2 - alpha*trace((U'*(Z)*U)) + trace(U'*U - I_k);
    end

    t = 1;
    
    fx = f(M,Z,W,U,V, alpha, beta,I_k);
    
    fx_new = f(M,Z,W,U,(V + t * searchDir),alpha,beta,I_k); 
    while fx_new > (fx + (a * t * (deriv.' * searchDir)))
        t = t * b;
        %V = V_x + t * searchDir;
        %fx_new = f(M,Z,W,U,(V + t * searchDir),alpha,beta,I_k);
        fx_new = f(M,Z,W,U,(V + t * searchDir),alpha,beta,I_k);
    end
end

end
