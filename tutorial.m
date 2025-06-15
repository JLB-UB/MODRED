function [A,B,C,D] = tutorial(L, Nx,k,rho_cp)
    % --- Paramètres physiques
    dx = L/(Nx-1);    % pas spatial
    alpha = k / rho_cp;  % diffusivité thermique [m2/s]
    
    % --- Etats : températures aux points internes (2..Nx-1)
    n = Nx - 2;
    
    % --- Matrice de Laplace discrétisée sur points internes
    e = ones(n,1);
    A_fd = spdiags([e -2*e e], -1:1, n, n) / dx^2;
    
    % --- Matrice A continue
    A = alpha * A_fd;
    
    % --- Vecteur B lié à la condition de flux imposé en x=0
    % Flux phi = -k * dT/dx(0) => approx (T(2)-T(1))/dx = -phi/k
    % T(1) est le bord non dans l'état, T(2) est premier état
    % Le flux impose une condition de Neumann, équivalent à un terme forcé dans
    % la première équation du système d'état
    
    B = zeros(n,1);
    B(1) = alpha * k / dx^2;
    
    % --- Sortie y = température au milieu x=L/2
    x_mid = L/2;
    idx_mid = round(x_mid/dx) - 1;  % -1 car états de 2..Nx-1
    
    C = zeros(1,n);
    C(idx_mid) = 1;
    
    D = zeros(1,1);

end