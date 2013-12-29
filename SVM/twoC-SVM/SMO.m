function [alphas, b] = SMO(X, y, C, toler, maxIter, kft, param)
%SMO Training parameters for SVM method
%   [alphas, b] =  SMO(X, y, C, toler, maxIter, kft, param) compute alphas 
%   and b for SVM method.
%   X is training set;
%   y is labels corresponding to training set X;
%   C is cost parameter of SVM;
%   toler is approximation;
%   maxIter is the maximum iteration number;
%   kft is kernel function type number;
%   param is parameter for some kernel functions

if nargin < 5
    maxIter = 10000;
end
if nargin < 6
    kft = 0;
end
if nargin < 7
    param = 0;
end

N = size(X, 1);
alphas = zeros(N, 1);
b = 0;
k = 0;

K = calcK(X, X, kft, param);

while k < maxIter
    
    % update E
    E = K * (alphas.*y) + b - y;
    
    % choose i
    KKT = y .* E;
    p1 = abs(alphas-C/2) < C/2;
    p2 = abs(KKT) > toler;
    if sum(p1&p2) == 0
        p1 = alphas == 0;
        p2 = KKT < -toler;
        if sum(p1&p2) == 0
            p1 = alphas == C;
            p2 = KKT > toler;
            if sum(p1&p2) == 0
                return;
            else
                KKT(~(p1&p2)) = 0;
                [~, i] = max(KKT);
            end
        else
            KKT(~(p1&p2)) = 0;
            [~, i] = min(KKT);
        end
    else
        KKT(~(p1&p2)) = 0;
        [~, i] = max(abs(KKT));
    end
    
    k = k + 1;

    % choose j
    if E(i)>0
        [~, j] = min(E);
        if j==i
            Eibak = E(i);
            E(i) = max(E)+1;
            [~, j] = min(E);
            E(i) = Eibak;
        end
    else
        [~, j] = max(E);
        if j==i
            Eibak = E(i);
            E(i) = min(E)-1;
            [~, j] = max(E);
            E(i) = Eibak;
        end
    end
    
    if y(i) ~= y(j)
        L = max(0, alphas(j)-alphas(i));
        H = min(C(j), C(j)+alphas(j)-alphas(i));
    else
        L = max(0, alphas(j)+alphas(i)-C(j));
        H = min(C(j), alphas(j)+alphas(i));
    end
    
    % update alphas(j)
    ajo = alphas(j);
    ita = K(i, i) - 2 * K(i, j) + K(j, j);
    alphas(j) = alphas(j) + y(j)*(E(i)-E(j))/ita;
    if alphas(j) > H
        alphas(j) = H;
    end
    if alphas(j) < L
        alphas(j) = L;
    end
    
    % update alphas(i)
    aio = alphas(i);
    alphas(i) = alphas(i) + y(i)*y(j)*(ajo-alphas(j));
    
    % update b
    b1 = b - E(i) + ...
        y(i)*K(i, i)*(aio-alphas(i)) + y(j)*K(j, i)*(ajo-alphas(j));
    b2 = b - E(j) + ...
        y(i)*K(i, j)*(aio-alphas(i)) + y(j)*K(j, j)*(ajo-alphas(j));
    if alphas(i)>0 && alphas(i)<C(i)
        b = b1;
    else
        if alphas(j)>0 && alphas(j)<C(j)
            b = b2;
        else
            b = (b1 + b2) / 2;
        end
    end
    
end

end

