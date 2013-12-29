% load('testSet.mat');
% [alphas, b] = SMO(X, y, C, toler);
% out = sign(calcK(X, X) * (alphas.*y) + b);
% disp(sum(out~=y)/size(y, 1));

load('testSetRBF.mat');

status1 = ones(40, 3);
status1i = 1;
status2 = ones(40, 3);
status2i = 1;

for C = 1:10:400
    for sigma = 0.01:0.05:2
        % train parameters
        [alphas, b] = SMO(X, y, C, toler, maxIter, kft, sigma);
        
        % select support vectors
        SV_p = alphas~=0;
        SV = X(SV_p, :);
        SV_alphas = alphas(SV_p);
        SV_y = y(SV_p);
        
        % test training set
        out1 = sign(calcK(X, SV, kft, sigma) * (SV_alphas.*SV_y) + b);
        status1(status1i, :) = [C, sigma, sum(out1~=y)/size(y, 1)];
        status1i = status1i + 1;
        disp(sum(out1~=y)/size(y, 1));
        
        % test test set
        out2 = sign(calcK(testX, SV, kft, sigma) * (SV_alphas.*SV_y) + b);
        status2(status2i, :) = [C, sigma, sum(out2~=testy)/size(testy, 1)];
        status2i = status2i + 1;
        disp(sum(out2~=testy)/size(testy, 1));
    end
end
