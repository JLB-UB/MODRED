function [Ar, Br, Cr, Dr, V_r, y_stat] = pod_reduction(A, B, C, D, m, r)
    % POD avec correction statique des modes négligés
    % Génération des snapshots par simulation de la réponse à un échelon unitaire
    % avec ode15s (solveur implicite adapté aux systèmes raides)
    %
    % Entrées :
    % A, B, C, D : matrices d'état du système complet
    % m         : nombre total de modes POD calculés (m >= r)
    % r         : nombre de modes à conserver dans le modèle réduit
    %
    % Sorties :
    % Ar, Br, Cr, Dr : matrices du modèle réduit
    % V_r            : matrice de projection POD (colonnes = modes)
    % y_stat         : sortie statique approximée pour entrée constante

    n = size(A,1);
    t_final = 10;               % temps final de simulation (à ajuster)
    numSnapshots = 1000;        % nombre de snapshots voulus
    tspan = linspace(0, t_final, numSnapshots);
    % Définition de la dynamique avec entrée constante u=1
    odefun = @(t,x) A*x + B*1;
    % Conditions initiales
    x0 = zeros(n,1);
    % Résolution avec ode15s
    [~, Xsol] = ode15s(odefun, tspan, x0);

    % Transposition car ode15s renvoie solution (temps x variables)
    X = Xsol';

    % === SVD tronquée ===
    [U, S, ~] = svds(X, m, 'largest');
    svals = diag(S);

    % === Construction du modèle réduit ===
    A_m = U' * A * U;
    B_m = U' * B;
    C_m = C * U;

    Ar = A_m(1:r, 1:r);
    Br = B_m(1:r, :);
    Cr = C_m(:, 1:r);

    As = A_m(r+1:end, r+1:end);
    Bs = B_m(r+1:end, :);
    Cs = C_m(:, r+1:end);

    if ~isempty(As)
        Dr = D - Cs * (As \ Bs);
    else
        Dr = D;
    end

    % Sortie statique complète, par résolution itérative gmres
    u_stat = 1; % ou autre valeur selon contexte
    rhs = -B * u_stat;
    tol = 1e-8;
    n = size(A,1);
    maxit = min(1000, n);
    
    [x_stat, flag, relres] = gmres(A, rhs, [], tol, maxit);
    
    if flag ~= 0
        warning('gmres n''a pas convergé, flag = %d, résidu relatif = %g', flag, relres);
    end    
    y_stat = C * x_stat;

end