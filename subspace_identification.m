function [A, B, C, D] = subspace_identification(u, y, n, i)
    % SUBSPACE_IDENTIFICATION : Identification sous-espace par moindres carrés
    %
    % Entrées :
    %   u : (m x N) matrice des entrées
    %   y : (p x N) matrice des sorties
    %   n : ordre estimé du système
    %   i : horizon d'observation (i >= n)
    %
    % Sorties :
    %   A, B, C, D : matrices du modèle d'état
    
    [m, N] = size(u);  % nb entrées, durée
    [p, Ny] = size(y);
    if Ny ~= N
        error('u et y doivent avoir le même nombre de colonnes.');
    end
    j = N - 2*i + 1;  % longueur des blocs
    
    % Construction des matrices de Hankel
    Up = hankel_blocks(u, i, j);
    Yp = hankel_blocks(y, i, j);

    %Uf = hankel_blocks(u(:, i:end), i, j);
    Yf = hankel_blocks(y(:, i:end), i, j);
    
    % Projection orthogonale : orthogonaliser Yf par rapport à Up
    Z = [Up; Yp];  % matrice de données passées
    [~, ~, V] = svd(Z, 'econ');
    L = V(:, 1:(m + p) * i);  % base pour projection
    Proj = eye(j) - L * L';   % projection orthogonale
    Yf_ortho = Yf * Proj;     % données futures projetées
    
    % SVD sur Yf_ortho
    [U, S, ~] = svd(Yf_ortho, 'econ');
    
    % Extraire les n premières composantes
    U1 = U(:, 1:n);
    S1 = S(1:n, 1:n);
    Gamma = U1 * sqrt(S1);  % matrice d'observabilité estimée
    
    % Estimation des états (X) par pseudo-inverse
    X = pinv(Gamma) * Yf;
    
    % Régression pour A, B
    X1 = X(:, 1:end-1);
    X2 = X(:, 2:end);
    U1 = u(:, i : i + j - 2);
    
    XU = [X1; U1];
    AB = X2 * pinv(XU);
    A = AB(:, 1:n);
    B = AB(:, n+1:end);
    
    % Régression pour C, D
    U1 = u(:, i : i + j - 1);
    CD = y(:, i : i + j - 1) * pinv([X; U1]);
    C = CD(:, 1:n);
    D = CD(:, n+1:end);

end

% Fonction utilitaire pour créer des blocs Hankel
function H = hankel_blocks(data, i, j)
    % data : (d x N)
    % i : nb de blocs (lignes)
    % j : nb de colonnes (horizon)
    [d, ~] = size(data);
    H = zeros(i*d, j);
    for k = 1:i
        H((k-1)*d+1:k*d, :) = data(:, k:k+j-1);
    end
end
