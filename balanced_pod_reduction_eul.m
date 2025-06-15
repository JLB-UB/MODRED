
function [Ar, Br, Cr, Dr] = balanced_pod_reduction_eul(A, B, C, D, r, dt, T)
% balanced_pod_reduction - Réduction équilibrée par snapshots (Balanced POD)
%
% Inputs:
%   A,B,C,D : matrices système (n×n, n×m, p×n, p×m)
%   r       : ordre réduit désiré
%   dt      : pas de temps de simulation
%   T       : durée totale de simulation pour les snapshots
%
% Outputs:
%   Ar, Br, Cr, Dr : système réduit d'ordre r

    n = size(A, 1);
    m = size(B, 2);
    p = size(C, 1);
    Dr = D;

    % Vérification de stabilité
    if any(real(eig(A)) > 1e-8)
        warning('⚠️ Le système semble instable (Re(λ)>0). La réduction équilibrée est instable ou non définie.');
    end

    % --- Nombre de snapshots ---
    N = floor(T / dt);

    % --- Initialisation des matrices de snapshots ---
    X = zeros(n, m * N);  % contrôlabilité
    Y = zeros(n, p * N);  % observabilité

    % --- Snapshots de contrôlabilité ---
    for i = 1:m
        x = zeros(n, 1);
        for k = 1:N
            u = zeros(m, 1);
            if k == 1
                u(i) = 1;  % impulsion sur l'entrée i
            end
            dx = A * x + B * u;
            x = x + dt * dx;

            if any(~isfinite(x))
                error('NaN ou Inf détecté dans les snapshots de contrôlabilité à l''étape %d.', k);
            end

            X(:, (i-1)*N + k) = x;
        end
    end

    % --- Snapshots d'observabilité (système adjoint) ---
    for i = 1:p
        x_adj = zeros(n, 1);
        for k = 1:N
            u_adj = zeros(p, 1);
            if k == 1
                u_adj(i) = 1;  % impulsion sur la sortie i (système adjoint)
            end
            dx_adj = A' * x_adj + C' * u_adj;
            x_adj = x_adj + dt * dx_adj;

            if any(~isfinite(x_adj))
                error('NaN ou Inf détecté dans les snapshots d''observabilité à l''étape %d.', k);
            end

            Y(:, (i-1)*N + k) = x_adj;
        end
    end

    % --- Produit croisé et SVD ---
    M = Y' * X;

    if any(~isfinite(M), 'all')
        error('NaN ou Inf détecté dans le produit Y''*X.');
    end

    [U, Sigma, V] = svd(M, 'econ');
    svals = diag(Sigma);

    % --- Vérification du rang numérique et ajustement de r si nécessaire ---
    tol = 1e-10;
    r_eff = sum(svals > tol);
    if r_eff < r
        warning('Seulement %d valeurs singulières significatives, r ajusté à %d.', r_eff, r_eff);
        r = r_eff;
    end

    % --- Bases équilibrées ---
    Sigma_r_sqrt_inv = diag(1 ./ sqrt(svals(1:r)));
    Phi = X * V(:, 1:r) * Sigma_r_sqrt_inv;
    Psi = Y * U(:, 1:r) * Sigma_r_sqrt_inv;

    % --- Système réduit ---
    Ar = Psi' * A * Phi;
    Br = Psi' * B;
    Cr = C * Phi;
end