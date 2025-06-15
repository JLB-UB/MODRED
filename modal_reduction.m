function [Ar, Br, Cr, Dr, y_stat] = modal_reduction(A, B, C, D, m, r, u_stat)
    % modal_reduction : réduction modale avec correction statique et sortie stationnaire
    % Entrées :
    %   A,B,C,D : matrices du système complet
    %   m       : nombre total de modes calculés
    %   r       : nombre de modes conservés
    %   u_stat  : (optionnel) entrée constante pour calcul sortie stationnaire (par défaut = 1)
    %
    % Sorties :
    %   Ar,Br,Cr,Dr : matrices du modèle réduit
    %   y_stat      : sortie stationnaire associée à u_stat

    if nargin < 7
        u_stat = 1;
    end

    % Calcul des modes propres
    opts.issym = true;
    [Phi_m, Am] = eigs(A, m, 'LR', opts);

    % Projection des matrices d'entrée/sortie
    Bm = Phi_m' * B;
    Cm = C * Phi_m;

    % Construction du modèle réduit
    Ar = Am(1:r, 1:r);
    Br = Bm(1:r, :);
    Cr = Cm(:, 1:r);

    % Modes négligés pour correction statique
    As = Am(r+1:m, r+1:m);
    Bs = Bm(r+1:m, :);
    Cs = Cm(:, r+1:m);

    % Gestion inversion diagonale avec précaution
    lambda_s = diag(As);
    epsilon = 1e-12;
    if any(abs(lambda_s) < epsilon)
        warning('Valeurs propres quasi-nulles détectées dans modes négligés, régularisation appliquée.');
        lambda_s(abs(lambda_s) < epsilon) = sign(lambda_s(abs(lambda_s) < epsilon)) * epsilon;
    end
    iAs = diag(1 ./ lambda_s);

    % Correction statique
    Dr = D - Cs * iAs * Bs;

    % Calcul sortie stationnaire par résolution itérative
    rhs = -B * u_stat;
    tol = 1e-8; 
    n = size(A,1);
    maxit = min(1000, n);

    % Préconditionneur
    try
        M = ichol(A);
        [x_stat, flag, relres] = gmres(A, rhs, [], tol, maxit, M, M');
    catch
        % Si ichol échoue, gmres sans préconditionneur
        [x_stat, flag, relres] = gmres(A, rhs, [], tol, maxit);
    end

    if flag ~= 0
        warning('gmres n''a pas convergé, flag = %d, résidu relatif = %g', flag, relres);
    end

    y_stat = C * x_stat;
end
