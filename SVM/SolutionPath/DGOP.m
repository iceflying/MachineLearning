function [Estar, lambdastar] = DGOP(lambda, alpha, D, yD, V, yV, ...
    hkernel, kernelparam)
%DGOP Determine the global optimal parameter lambda
%   [Estar, lambdastar] determine the global optimal lambda which can get
%   the global minima of validation function in svm.
%   lambda      the output lambda of svmPath
%   alpha       the output alpha set corresponding to lambda
%   D           training data set,      |D| * p feature vector
%   yD          lables corresponding to training set,   |D| * 1 label
%   V           validation data set,    |V| * p feature vector
%   yV          lables corresponding to validation set, |V| * 1 label
%   Estar       the global minima value of validation function
%   lambdastar  the global optimal lambda which can get Estar

lmax = length(lambda);
nV = length(yV);

K = hkernel(D, V, kernelparam);     % |D| * |V|
g = diag(yD) * K;   % g(i) = y(i)*K(i, j)
hl = g' * (alpha * diag(lambda));   % |D| * l

l = 1;
E = sum(abs(yV-sign(hl(:, l))))/(2*nV);
Estar = E;
lambdastar = lambda(l);

while l < lmax
    Isl = find(hl(:, l).*hl(:, l+1)<0);
    lambdalstar = (lambda(l+1)*hl(Isl, l)-lambda(l)*hl(Isl, l+1))./...
                  (hl(Isl, l)-hl(Isl, l+1));
    [~, Ilambdal] = sort(lambdalstar, 'descend');
    E = sum(abs(yV-sign(hl(:, l))))/(2*nV);
    for i = 1:length(Ilambdal)
        if yV(Ilambdal(i)) == sign(hl(Ilambdal(i), l))
            E = E + 1/nV;
        else
            E = E - 1/nV;
        end
        if E < Estar
            Estar = E;
            lambdastar = lambdalstar(Ilambdal(i));
        end
    end
    l = l + 1;
end
