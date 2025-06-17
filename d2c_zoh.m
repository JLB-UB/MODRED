function [A, B, C, D] = d2c_zoh(Ad, Bd, Cd, Dd, dt)
    % Conversion discret -> continu par ZOH avec vérification de stabilité
    % et gestion du warning logm.
    
    % Vérification de la stabilité
    eig_Ad = eig(Ad);
    if any(abs(eig_Ad) > 1)
        warning('Ad possède des valeurs propres hors du cercle unité, conversion peut être erronée');
    end

    % Calcul logarithme matriciel
    try
        M = logm(Ad) / dt;
    catch ME
        warning('logm a échoué : %s\nTentative d''approximation via méthode Tustin.', ME.message);
        M = (Ad - eye(size(Ad))) / dt; % Approximation grossière
    end

    % Vérification valeurs propres de M
    if any(imag(eig(M)) ~= 0)
        warning('Matrice A continue a des valeurs propres complexes.');
    end
    
    A = real(M);
    
    % Calcul B continu (méthode classique)
    n = size(Ad,1);
    I = eye(n);
    Z = zeros(n);
    % Construction matrice pour résolution B et D continus
    % [Ad - I, Bd; 0, I] (matrice augmentée) -> méthode de Van Loan (simplifiée ici)
    
    % Approximation de B continu (méthode simple)
    % Bc ≈ M^{-1} * Bd (si M inversible)
    if rank(A) == n
        B = A \ Bd;
    else
        warning('A non inversible, utilisation d''une pseudo-inverse pour B continu');
        B = pinv(A) * Bd;
    end
    
    C = Cd;
    D = Dd;
end
