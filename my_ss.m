function sys = my_ss(A,B,C,D)
    % my_ss - crée une structure similaire à un système d'état MATLAB
    %
    % Usage:
    %   sys = my_ss(A,B,C,D)
    %
    % sys est une structure avec champs :
    %   A, B, C, D : matrices du système
    %   size       : taille [n,m,p]
    %   isCont     : true (système continu)
    %
    % Exemple:
    %   A = [0 1; -2 -3];
    %   B = [0; 1];
    %   C = [1 0];
    %   D = 0;
    %   sys = my_ss(A,B,C,D);
    
    sys.A = A;
    sys.B = B;
    sys.C = C;
    sys.D = D;
    
    sys.size = [size(A,1), size(B,2), size(C,1)]; % [n,m,p]
    sys.isCont = true;
    
    % Optionnel : ajouter une fonction de simulation simple intégrée
    sys.simulate = @(u,tspan,x0) simulate_ss(A,B,C,D,u,tspan,x0);

end