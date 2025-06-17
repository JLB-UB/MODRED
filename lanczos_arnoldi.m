function [V, D] = lanczos_arnoldi(A, k, v0, symmetric, verbose, sort_mode)
% LANCZOS_EIG_REORTH - Méthode de Lanczos/Arnoldi avec réorthogonalisation complète
%
% Entrées :
%   A         : matrice (n x n) ou handle @(x) => A*x
%   k         : nombre de valeurs propres dominantes désirées
%   v0        : vecteur initial (défaut = vecteur aléatoire)
%   symmetric : true si A est symétrique (-> Lanczos), sinon Arnoldi
%   verbose   : (optionnel) affiche les résidus des VP (défaut = false)
%   sort_mode : (optionnel) 'real' (par partie réelle décroissante, défaut) ou 'abs' (par amplitude)
%
% Sorties :
%   V : matrice des vecteurs propres (colonnes)
%   D : matrice diagonale des valeurs propres

    if nargin < 3 || isempty(v0), use_random = true; else, use_random = false; end
    if nargin < 4, symmetric = false; end
    if nargin < 5, verbose = false; end
    if nargin < 6, sort_mode = 'real'; end  % 'real' ou 'abs'

    if isa(A, 'function_handle')
        n = length(A(randn(10,1)));  % estimation dimension
    else
        n = size(A,1);
    end

    if use_random
        v0 = randn(n,1);
    end
    v0 = v0 / norm(v0);

    V = zeros(n, k+1);
    H = zeros(k+1, k);

    V(:,1) = v0;

    for j = 1:k
        if isa(A, 'function_handle')
            w = A(V(:,j));
        else
            w = A * V(:,j);
        end

        % Gram-Schmidt modifiée (2 passes)
        for i = 1:j
            hij = V(:,i)' * w;
            H(i,j) = hij;
            w = w - hij * V(:,i);
        end
        for i = 1:j
            hij = V(:,i)' * w;
            H(i,j) = H(i,j) + hij;
            w = w - hij * V(:,i);
        end

        H(j+1,j) = norm(w);
        if H(j+1,j) < 1e-12
            warning('La base de Krylov s'est effondrée à l'itération %d.', j);
            break;
        end
        V(:,j+1) = w / H(j+1,j);
    end

    % Réduction de taille
    H = H(1:k, 1:k);
    V = V(:,1:k);

    % Diagonalisation
    if symmetric
        H = (H + H') / 2;
        [Q, Dmat] = eig(H);
    else
        [Q, Dmat] = eig(H);
    end

    Dvals = diag(Dmat);

    % Tri selon le mode choisi
    switch lower(sort_mode)
        case 'real'
            [~, idx] = sort(real(Dvals), 'descend');
        case 'abs'
            [~, idx] = sort(abs(Dvals), 'descend');
        otherwise
            error('sort_mode doit être ''real'' ou ''abs''.');
    end

    Dvals = Dvals(idx);
    Q = Q(:,idx);

    V = V * Q;
    D = diag(Dvals);

    if verbose
        fprintf('Résidus relatifs :\n');
        for i = 1:k
            lambda = D(i,i);
            x = V(:,i);
            if isa(A, 'function_handle')
                r = A(x) - lambda*x;
            else
                r = A*x - lambda*x;
            end
            fprintf('  λ = %.6e,  ||Ax - λx|| / ||x|| = %.2e\n', lambda, norm(r)/norm(x));
        end
    end
end
