function [A_r, B_r, C_r, D_r] = subspace_model_reduction(A, B, C, D, r, N, dt)
    % SUBSPACE_MODEL_REDUCTION : réduction par identification sous-espace (N4SID)
    %   Utilise des données de simulation pour identifier un modèle réduit
    %
    % Entrées :
    %   A, B, C, D : matrices du modèle complet (discret ou continu)
    %   r         : ordre réduit désiré
    %   N         : nombre de pas de temps
    %   dt        : pas d'échantillonnage
    %
    % Sorties :
    %   A_r, B_r, C_r, D_r : modèle réduit (continu)
    
    % === Dimensions ===
    nx = size(A, 1);
    m = size(B, 2);
    p = size(C, 1);
    
    % === Simulation du système complet ===
    T = (0:N-1) * dt;  % vecteur ligne
    
    % === Génération d'une entrée PRBS normalisée ===
    % rng(0); % pour reproductibilité
    % block_size = 100;
    % n_blocks = ceil(N / block_size);
    % % Génère des valeurs -1 ou +1 par bloc
    % u_blocks = 2 * (randi([0 1], m, n_blocks)) - 1;
    % % Répète chaque bloc block_size fois
    % u = repelem(u_blocks, 1, block_size);
    % % Tronque à N colonnes
    % u = u(:, 1:N);

    % u = sin(2*pi*0.5*T) + 0.5*sin(2*pi*2*T);
    % u = repmat(u, m, 1);
    % u = u / max(abs(u), [], 'all');

    u = ones(m,N);

    % Interpolation pour intégration
    u_interp = @(t) interp1(T, u', t, 'previous', 0)';
    
    % Simulation dynamique dx/dt = A x + B u(t)
    f = @(t, x) A*x + B*u_interp(t);
    x0 = zeros(nx, 1);
    opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-9, 'MaxStep', dt);
    
    [t_sim, x_sim] = ode15s(f, T, x0, opts);
    x_sim = x_sim';  % (nx x Nt)
    Nt = length(t_sim);
    
    % Correspondance avec les entrées
    u_used = u(:,1:Nt);
    y = C * x_sim + D * u_used;
    
    % ---- Correction interpolation ----
    % Forcer t_sim et T en vecteurs colonnes
    if isrow(t_sim), t_sim = t_sim'; end
    if isrow(T), T = T'; end
    
    % Transposer u_used et y pour interp1
    u_used_t = u_used';
    y_t = y';
    
    % Interpoler sur T (points réguliers d'origine)
    u_used_interp = interp1(t_sim, u_used_t, T, 'linear', 'extrap');
    y_interp = interp1(t_sim, y_t, T, 'linear', 'extrap');
    
    % Repasser en (m x N) et (p x N)
    u_used = u_used_interp';
    y = y_interp';
    
    fprintf('Après interpolation corrigée : u_used size = [%d, %d], y size = [%d, %d]\n', ...
        size(u_used,1), size(u_used,2), size(y,1), size(y,2));
    
    % === Identification du modèle réduit discret ===
    [A_r, B_r, C_r, D_r] = subspace_identification(u_used', y', r);

end
