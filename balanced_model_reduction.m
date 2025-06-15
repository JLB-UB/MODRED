function [A_r, B_r, C_r, D_r, err_hinf, borne] = balanced_model_reduction(A, B, C, D, r)
    % balanced_truncation_large - Réduction par réalisation équilibrée pour gros systèmes
    %
    % Entrées :
    %   A,B,C,D : matrices du système original (A nxn)
    %   r       : ordre du modèle réduit souhaité (r < n)
    %
    % Sorties :
    %   sys_red : système réduit ss(A_r,B_r,C_r,D_r)
    %   err_hinf: norme H-infinity de l'erreur entre original et réduit
    %   borne   : borne théorique de l'erreur
    %   sigma   : valeurs singulières équilibrées
        
    % --- Résolution des équations de Lyapunov ---
    % Pour les gros systèmes, on peut utiliser lyapchol ou lyap pour matrices creuses.
    % Ici on utilise lyap, mais on peut adapter à lyapchol si nécessaire.
    Wc = lyap(A, B*B');
    Wo = lyap(A', C'*C);
    
    % Factorisation de Cholesky (peut échouer si Wc ou Wo ne sont pas définies positives exactes)
    Rc = chol(Wc, 'lower');
    Ro = chol(Wo, 'lower');
    
    % Calcul de la matrice M et SVD
    M = Ro' * Rc;
    [U,Sigma,V] = svd(M);
    
    % Transformation d'équilibrage
    Sigma_sqrt_inv = diag(1 ./ sqrt(diag(Sigma)));
    
    T = Rc * V * Sigma_sqrt_inv;
    T_inv = Sigma_sqrt_inv * U' * Ro';
    
    % Système équilibré
    A_bal = T_inv * A * T;
    B_bal = T_inv * B;
    C_bal = C * T;
    
    % Troncature
    A_r = A_bal(1:r, 1:r);
    B_r = B_bal(1:r, :);
    C_r = C_bal(:, 1:r);
    D_r = D;
    
    sys_red = ss(A_r, B_r, C_r, D_r);
    
    % Calcul de l'erreur
    sys = ss(A,B,C,D);
    sys_err = sys - sys_red;
    err_hinf = norm(sys_err, inf);
    
    sigma = diag(Sigma);
    borne = 2*sum(sigma(r+1:end));
end
