function [A, B, C, D] = subspace_identification(u, y, n)
% SUBSPACE_IDENTIFICATION
% Identification sous-espace simplifiée type N4SID, avec vérifications dimensionnelles
%
% u : (N x m) entrée
% y : (N x l) sortie
% n : ordre du modèle

fprintf('subspace_identification: taille u = [%d %d], y = [%d %d]\n', size(u,1), size(u,2), size(y,1), size(y,2));
% Dimensions
[N, m] = size(u);
[Ny, l] = size(y);
assert(N == Ny, 'Entrée et sortie doivent avoir la même longueur');

% Limitation sécurisée du nombre de blocs Hankel
s = min(2*n, floor(N/2));
if s < 1
    error('Nombre de blocs Hankel "s" trop petit ou données trop courtes');
end

% Construction des matrices Hankel
U = hankel_matrix(u, s);   % taille (s*m x N-s+1)
Y = hankel_matrix(y, s);   % taille (s*l x N-s+1)

% Affichage debug
fprintf('Taille U: [%d x %d]\n', size(U,1), size(U,2));
fprintf('Taille Y: [%d x %d]\n', size(Y,1), size(Y,2));

% SVD sur données combinées
W = [U; Y];
[~, S, V] = svd(W, 'econ');
sing_vals = diag(S);
tol = max(size(W)) * eps(max(sing_vals));
r_eff = sum(sing_vals > tol);

if n > r_eff
    warning('Rang effectif = %d < n = %d. Ajustement automatique.', r_eff, n);
    n = r_eff;
end

Vn = V(:, 1:n);

% Calcul matrice d'observabilité estimée
Ob = Y * Vn / (Vn' * Vn);

fprintf('Taille Ob: [%d x %d]\n', size(Ob,1), size(Ob,2));
fprintf('Sorties l = %d\n', l);

if size(Ob,1) < l
    error('La matrice Ob est trop petite (%d lignes) pour extraire C avec l = %d', size(Ob,1), l);
end

C = Ob(1:l, :);
Ob1 = Ob(1:end-l, :);
Ob2 = Ob(l+1:end, :);

% Estimation A
A = pinv(Ob1) * Ob2;

% Estimation de l'état X
X = pinv(Ob) * Y;

% Construction des matrices pour la régression de B et D
Xk   = X(:, 1:end-1);
Xkp1 = X(:, 2:end);
Uk   = U(:, 2:end);

% Extraction des entrées instantanées (dernier bloc)
m = size(u, 2);
Uk_last = Uk(end-m+1:end, :);

% Régression pour A et B
AB = Xkp1 / [Xk; Uk_last];
A = AB(:, 1:n);
B = AB(:, n+1:end);

% Régression pour C et D (sortie instantanée)
Yk = Y(end-l+1:end, 2:end);
CD = Yk / [X(:, 2:end); Uk_last];
C = CD(:, 1:n);
D = CD(:, n+1:end);
end

function H = hankel_matrix(data, s)
% Crée une matrice de Hankel par blocs
[N, m] = size(data);
cols = N - s + 1;
H = zeros(s * m, cols);
for i = 1:s
    H((i-1)*m+1:i*m, :) = data(i:i+cols-1, :)';
end
end
