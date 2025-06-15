function [Ar, Br, Cr, Dr, y_stat] = modal_reduction(A, B, C, D, m, r, u_stat)
    % modal_reduction : r�duction modale avec correction statique et sortie stationnaire
    % Entr�es :
    %   A,B,C,D : matrices du syst�me complet
    %   m       : nombre total de modes calcul�s
    %   r       : nombre de modes conserv�s
    %   u_stat  : (optionnel) entr�e constante pour calcul sortie stationnaire (par d�faut = 1)
    %
    % Sorties :
    %   Ar,Br,Cr,Dr : matrices du mod�le r�duit
    %   y_stat      : sortie stationnaire associ�e � u_stat

    if nargin < 7
        u_stat = 1;
    end

    % Calcul des modes propres
    opts.issym = true;
    [Phi_m, Am] = eigs(A, m, 'LR', opts);

    % Projection des matrices d'entr�e/sortie
    Bm = Phi_m' * B;
    Cm = C * Phi_m;

    % Construction du mod�le r�duit
    Ar = Am(1:r, 1:r);
    Br = Bm(1:r, :);
    Cr = Cm(:, 1:r);

    % Modes n�glig�s pour correction statique
    As = Am(r+1:m, r+1:m);
    Bs = Bm(r+1:m, :);
    Cs = Cm(:, r+1:m);

    % Gestion inversion diagonale avec pr�caution
    lambda_s = diag(As);
    epsilon = 1e-12;
    if any(abs(lambda_s) < epsilon)
        warning('Valeurs propres quasi-nulles d�tect�es dans modes n�glig�s, r�gularisation appliqu�e.');
        lambda_s(abs(lambda_s) < epsilon) = sign(lambda_s(abs(lambda_s) < epsilon)) * epsilon;
    end
    iAs = diag(1 ./ lambda_s);

    % Correction statique
    Dr = D - Cs * iAs * Bs;

    % Calcul sortie stationnaire par r�solution it�rative
    rhs = -B * u_stat;
    tol = 1e-8; 
    n = size(A,1);
    maxit = min(1000, n);

    % Pr�conditionneur
    try
        M = ichol(A);
        [x_stat, flag, relres] = gmres(A, rhs, [], tol, maxit, M, M');
    catch
        % Si ichol �choue, gmres sans pr�conditionneur
        [x_stat, flag, relres] = gmres(A, rhs, [], tol, maxit);
    end

    if flag ~= 0
        warning('gmres n''a pas converg�, flag = %d, r�sidu relatif = %g', flag, relres);
    end

    y_stat = C * x_stat;
end
