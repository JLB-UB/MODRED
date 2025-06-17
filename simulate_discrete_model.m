function [t_sim, y_sim, x_sim] = simulate_discrete_model(Ad, Bd, Cd, Dd, u_fun, dt, N, x0)
    % SIMULATE_DISCRETE_MODEL_FUN Simule un système discret avec une fonction anonyme u(t)
    %
    % Entrées :
    %   Ad, Bd, Cd, Dd : matrices du système discret
    %   u_fun : fonction anonyme @(t) renvoyant un vecteur colonne (m x 1)
    %   dt : pas d'échantillonnage
    %   N : nombre de pas de temps
    %   x0 : état initial (nx x 1)
    %
    % Sorties :
    %   t_sim : vecteur temps (1 x N)
    %   y_sim : sortie simulée (p x N)
    %   x_sim : états (nx x N)
    
    %assert(isscalar(N) && isnumeric(N) && N > 0 && isfinite(N), ...
    %    'N doit être un scalaire entier strictement positif.');
    
    % Initialisation
    nx = size(Ad, 1);
    p = size(Cd, 1);
    t_sim = (0:N-1) * dt;
    x_sim = zeros(nx, N);
    y_sim = zeros(p, N);
    
    x_k = x0;
    
    for k = 1:N
        t = t_sim(k);
        u_k = u_fun(t);  % entrée au temps t, doit retourner (m x 1)
        y_sim(:,k) = Cd * x_k + Dd * u_k;
        x_k = Ad * x_k + Bd * u_k;
        x_sim(:,k) = x_k;
    end
end
