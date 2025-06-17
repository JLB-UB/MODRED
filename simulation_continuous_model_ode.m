function [y, tcpu] = simulation_continuous_model_ode(A, B, C, D, U, dt)
% SIMULATION_CONTINUOUS_MODEL_ODE
% Simulation d'un système continu dx/dt = A x + B u(t), y = C x + D u(t)
% avec une entrée discrète interpolée et intégration par ode15s
%
% Entrées :
%   A, B, C, D : matrices d'état (réduites ou non)
%   U          : entrée discrète (Nd x p)
%   dt         : pas d'échantillonnage
%
% Sorties :
%   y          : sortie simulée (Nd x m)
%   tcpu       : temps CPU

    tcpu = cputime;

    % Conversion en matrices creuses si nécessaire
    if ~issparse(A), A = sparse(A); end
    if ~issparse(B), B = sparse(B); end
    if ~issparse(C), C = sparse(C); end
    if ~issparse(D), D = sparse(D); end

    % Dimensions
    [Nd, p] = size(U);      % Nd : nombre d'échantillons
    [n, ~] = size(B);       % n : taille de l'état
    m = size(C, 1);         % m : nombre de sorties

    % Grille temporelle
    T = (0:Nd-1) * dt;

    % Interpolateur de l'entrée u(t) (fonction qui retourne vecteur colonne)
    u_interp = @(t) interp1(T, U, t, 'previous', 'extrap')';  % (p x 1)

    % Définition du RHS : dx/dt = A x + B u(t)
    f = @(t, x) A*x + B*u_interp(t);

    % Conditions initiales
    x0 = zeros(n, 1);

    % Options d'intégration
    opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-9, 'MaxStep', dt);

    % Intégration du système
    [t_sim, x_sim] = ode15s(f, T, x0, opts);  % x_sim: Nt x n

    % Calcul de y(t) = C x + D u(t)
    Nt = length(t_sim);
    y = zeros(Nt, m);
    for i = 1:Nt
        u_now = u_interp(t_sim(i));
        y(i,:) = (C * x_sim(i,:)' + D * u_now)';
    end

    tcpu = cputime - tcpu;
end
