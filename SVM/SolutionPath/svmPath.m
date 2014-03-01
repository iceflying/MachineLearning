function [lambda, alpha, elbow] = svmPath(x, y, hkernel, kernelparam)
%SVMPATH compute the entire regularization path for svm
%   [lambda, alpha, elbow] = SVMPATH(x, y, hkernel, kernelparam) compute
%   lambda, alpha and elbow set of the entire regularization path for svm.
%   x               N * p feature vector
%   y               N * 1 label
%   hkernel         kernel function handle
%   kernelparam     kernel parameter

EPS = 1e-10;
lambdamin = 1e-3;

N = length(y);
Np = sum(y==1);
Nn = sum(y==-1);
gamma = Nn/N;

Left = (1:N)';
Elbow = [];
Right = [];

maxIter = 5*N;
lambda = zeros(maxIter, 1);
alpha = zeros(N, maxIter);
elbow = cell(maxIter, 1);

K = hkernel(x, x, kernelparam);
Kt = diag(y)*K*diag(y);
% diag(y)(i, i) == y(i)
% Kt(i, j) == y(i)*y(j)*K(i, j)

k = 1;  % iterator counter
[lambda(k), alpha(:, k), Elbow] = svmPathInit(y, Kt, gamma);
Left = setdiff(Left, Elbow);
Kstar = Kt(Elbow, Elbow);
fl = (K*(alpha(:, k).*y))/lambda(k); % fl(x) = sum_j(alpha(j) * y(j) * K(i, j))/lambda
obs = Elbow;
elbow{k} = Elbow;

while (lambda(k) > lambdamin)
    if (~isempty(Elbow))
        b = Kstar \ ones(length(Elbow), 1);
        hl = K(:, Elbow) * (b .* y(Elbow));
        dl = fl - hl;
        
        immobile = sum(abs(dl)) / N < EPS;
        
        temp = ~(abs(b)<EPS);
        lambdaleft = -ones(size(temp));
        lambdaright = -ones(size(temp));
        if (sum(temp))
            lambdaleft(temp) = ((y(Elbow(temp))==-1) + ...
                y(Elbow(temp))*gamma-alpha(Elbow(temp), k)+...
                lambda(k)*b(temp))./b(temp);
            lambdaright(temp) = (-alpha(Elbow(temp), k)+...
                lambda(k)*b(temp))./b(temp);
        end
        lambdarl = [lambdaright; lambdaleft];
        lambdaexit = max(lambdarl(lambdarl < lambda(k) - EPS));
        if (isempty(lambdaexit))
            lambdaexit = -1;
        end
        if (~immobile)
            temp = ~(abs(y-hl)<EPS);
            temp(Elbow) = false;
            temp(obs) = false;
            lambdae = zeros(size(temp));
            lambdae(temp) = lambda(k).*dl(temp)./(y(temp)-hl(temp));
            lambdae(~temp) = -1;
            lambdaentry = max(lambdae(lambdae < lambda(k) - EPS));
            if (isempty(lambdaentry))
                lambdaentry = -1;
            end
        else
            lambdaentry = -1;
        end
        
        lambdamax = max(lambdaexit, lambdaentry);
        if (lambdamax < 0)
            break;
        end
        
        % update lambda
        lambda(k+1) = lambdamax;
        if (lambda(k+1) > lambda(k))
            break;
        end
        % update alpha
        alpha(Right, k+1) = 0;
        alpha(Left, k+1) = gamma*(y(Left)==1) + (1-gamma)*(y(Left)==-1);
        alpha(Elbow, k+1) = alpha(Elbow, k) - (lambda(k)-lambdamax)*b;
        fl = (K*(alpha(:, k+1).*y))/lambda(k+1);
        % update sets
        if (lambdaentry > lambdaexit)
            iin = find(abs(lambdae-lambdaentry)<EPS);
            obs = iin;
            Left = setdiff(Left, iin);
            Right = setdiff(Right, iin);
            Elbow = [Elbow; iin];
            Kstar = Kt(Elbow, Elbow);
        else
            tempr = abs(lambdaright - lambdaexit)<EPS;
            toright = Elbow(tempr);
            templ = abs(lambdaleft - lambdaexit)<EPS;
            toleft = Elbow(templ);
            temp = tempr | templ;
            Elbow = Elbow(~temp);
            Right = [Right; toright];
            Left = [Left; toleft];
            Kstar = Kt(Elbow, Elbow);
            obs = [toright; toleft];
        end
    else
        [lambdat, alphat, Elbowt] = svmPathInit(y(Left), Kt(Left, Left), gamma);
        if (isempty(lambdat) || lambdat<0 || lambdat>lambda(k))
            break;
        end
        lambda(k+1) = lambdat;
        alpha(:, k+1) = alpha(:, k);
        Elbow = Left(Elbowt);
        Left = setdiff(Left, Elbow);
        Kstar = Kt(Elbow, Elbow);
        fl = (K*(alpha(:, k+1).*y))/lambda(k+1);
        obs = Elbow;
    end
    
    k = k+1;
    elbow{k} = Elbow;
end

lambda = lambda(1:k);
alpha = alpha(:, 1:k);
elbow = elbow(1:k);
