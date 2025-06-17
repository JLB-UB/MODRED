function [Theta, Y_pred, Phi] = narmax_mimo_activation(U, Y, na, nb, nk, degree, activation)
% narmax_mimo_activation - Identification NARMAX MIMO avec choix fonction activation
%
% Syntaxe :
%   [Theta, Y_pred, Phi] = narmax_mimo_activation(U, Y, na, nb, nk, degree, activation)
%
% Entrées :
%   U          : matrice entrée [N x m]
%   Y          : matrice sortie [N x p]
%   na         : ordre autoregressif (sorties retardées)
%   nb         : ordre entrée (entrées retardées)
%   nk         : délai d'entrée (scalaire ou vecteur [1 x m])
%   degree     : degré polynôme (utile si activation = 'poly')
%   activation : fonction activation : 'poly', 'sigmoid', 'tanh', 'relu', 'linear'
%
% Sorties :
%   Theta  : coefficients [nbases x p]
%   Y_pred : sortie prédite [N - max_delay x p]
%   Phi    : matrice de régression [N - max_delay x nbases]

[N, m] = size(U);
[Ny, p] = size(Y);

if N ~= Ny
    error('Nombre de lignes de U et Y doit être égal.');
end

if isscalar(nk)
    nk = nk * ones(1,m);
elseif length(nk) ~= m
    error('nk doit être un scalaire ou un vecteur de longueur m.');
end

max_delay = max([na, max(nb + nk - 1)]);

if N <= max_delay
    error('Taille des données trop petite pour les retards spécifiés.');
end

% Construction des variables retardées sorties
Phi_y = [];
for i=1:na
    Phi_y = [Phi_y, Y(max_delay - i + 1 : N - i, :)]; % (N-max_delay) x p
end
Phi_y = reshape(Phi_y, [], na*p);

% Construction des variables retardées entrées
Phi_u = [];
for j=1:nb
    for inp=1:m
        idx_start = max_delay - nk(inp) - j + 2;
        idx_end = N - nk(inp) - j + 1;
        Phi_u = [Phi_u, U(idx_start:idx_end, inp)];
    end
end

Phi_raw = [Phi_y, Phi_u]; % brute (N - max_delay) x (na*p + nb*m)

% Application fonction d'activation
switch lower(activation)
    case 'poly'
        Phi = ones(size(Phi_raw,1),1);
        for d=1:degree
            for c=1:size(Phi_raw,2)
                Phi = [Phi, Phi_raw(:,c).^d];
            end
        end
    case 'sigmoid'
        Phi = 1 ./ (1 + exp(-Phi_raw));
        Phi = [ones(size(Phi,1),1), Phi];
    case 'tanh'
        Phi = tanh(Phi_raw);
        Phi = [ones(size(Phi,1),1), Phi];
    case 'relu'
        Phi = max(0, Phi_raw);
        Phi = [ones(size(Phi,1),1), Phi];
    case 'linear'
        Phi = [ones(size(Phi_raw,1),1), Phi_raw];
    otherwise
        error('Activation inconnue : choisir parmi poly, sigmoid, tanh, relu, linear');
end

Y_train = Y(max_delay+1:end, :);

% Estimation des coefficients (moindres carrés)
Theta = (Phi' * Phi) \ (Phi' * Y_train);

% Prédiction
Y_pred = Phi * Theta;

end
