function [Ar, Br, Cr, Dr] = balanced_pod_reduction_ode(A, B, C, D, r, dt, T)
% Réduction de modèle par Balanced POD avec intégration ode15s
%
% Entrées :
%   A, B, C, D : matrices du système original
%   r          : ordre réduit
%   dt, T      : pas de temps et temps total de simulation
%
% Sorties :
%   Ar, Br, Cr, Dr : système réduit

    n = size(A,1);  m = size(B,2);  p = size(C,1);
    N = floor(T/dt);                % nombre de pas de temps
    tspan = linspace(0, T, N);      % vecteur temps

    % ---- Initialisation des matrices de snapshots ----
    X = zeros(n, N);                % Snapshots pour contrôlabilité
    Y = zeros(n, N);                % Snapshots pour observabilité

    % ---- Simulation avec impulsion unité sur chaque entrée ----
    fprintf('Calcul des snapshots de contrôlabilité (ode15s)...\n');
    u_func = @(t) ones(m,1);        % Impulsion unité
    x0 = zeros(n,1);
    [~, x_snap] = ode15s(@(t,x) A*x + B*u_func(t), tspan, x0);
    X = x_snap';  % (n x N)

    % ---- Simulation adjoint avec impulsion unité en sortie ----
    fprintf('Calcul des snapshots d''observabilité (ode15s)...\n');
    u_adj_func = @(t) ones(p,1);
    x0_adj = zeros(n,1);
    [~, x_adj_snap] = ode15s(@(t,x) A'*x + C'*u_adj_func(t), tspan, x0_adj);
    Y = x_adj_snap';  % (n x N)

    % ---- Produit croisé et SVD ----
    M = Y' * X;                    % (N x N)
    [U, S, V] = svd(M, 'econ');    % Économie mémoire

    % ---- Bases équilibrées ----
    S_root_inv = diag(1 ./ sqrt(diag(S(1:r,1:r))));
    Phi = X * V(:,1:r) * S_root_inv;
    Psi = Y * U(:,1:r) * S_root_inv;

    % ---- Projection ----
    Ar = Psi' * A * Phi;
    Br = Psi' * B;
    Cr = C * Phi;
    Dr = D;

    fprintf('Réduction équilibrée terminée (ordre %d).\n', r);
end