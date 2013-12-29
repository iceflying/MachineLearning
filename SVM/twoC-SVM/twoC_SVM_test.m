clear;
load('testSetRBF.mat');

% Cp = C;
% Cn = C;

status1 = ones(40*40*40, 4);
status1i = 1;
status2 = ones(40*40*40, 4);
status2i = 1;

ROC1 = zeros(40*40*40, 2);
ROC2 = zeros(40*40*40, 2);

for Cp = 1:10:400
    for Cn = 1:10:400
        
        C = zeros(size(y));
        C(y==1) = Cp;
        C(y==-1) = Cn;
        
        for sigma = 0.01:0.05:2
            [alphas, b] = SMO(X, y, C, toler, maxIter, kft, sigma);

            SV_p = alphas~=0;
            SV = X(SV_p, :);
            SV_alphas = alphas(SV_p);
            SV_y = y(SV_p);

            out1 = sign(calcK(X, SV, kft, sigma) * (SV_alphas.*SV_y) + b);
            status1(status1i, :) = [Cp, Cn, sigma, sum(out1~=y)/size(y, 1)];
            ROC1(status1i, 1) = sum(out1==1 & y==-1)/sum(y==-1);
            ROC1(status1i, 2) = sum(out1==1 & y==1)/sum(y==1);
            status1i = status1i + 1;
            disp(sum(out1~=y)/size(y, 1));

            out2 = sign(calcK(testX, SV, kft, sigma) * (SV_alphas.*SV_y) + b);
            status2(status2i, :) = [Cp, Cn, sigma, sum(out2~=testy)/size(testy, 1)];
            ROC2(status2i, 1) = sum(out2==1 & testy==-1)/sum(testy==-1);
            ROC2(status2i, 2) = sum(out2==1 & testy==1)/sum(testy==1);
            status2i = status2i + 1;
            disp(sum(out2~=testy)/size(testy, 1));
        end
    end
end

plot(ROC1(:, 1), ROC1(:, 2), '*');
hold('all');
plot(ROC2(:, 1), ROC2(:, 2), '*');
