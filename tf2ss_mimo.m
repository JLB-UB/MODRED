function [A, B, C, D] = tf2ss_mimo(num, den)
% Convertit une fonction de transfert MIMO en un modèle d'état (A,B,C,D)
% Entrées :
%   num : cellule de cellules de numérateurs {ny x nu}
%   den : cellule de cellules de dénominateurs {ny x nu}
% Sorties :
%   A, B, C, D : matrices d'état
%
% exemple d'aplication
% Système MIMO 2x2
% num = {
%     [1],      [2 1];
%     [1 1],    [3]
% };
% den = {
%     [1 2],    [1 3];
%     [1 2],    [1 4]
% };
% 
% [A, B, C, D] = tf2ss_mimo(num, den);

[ny, nu] = size(num);
order_total = 0;
orders = zeros(ny, nu);

% 1. Déterminer l'ordre total
for i = 1:ny
    for j = 1:nu
        orders(i,j) = length(den{i,j}) - 1;
        order_total = order_total + orders(i,j);
    end
end

A = zeros(order_total);
B = zeros(order_total, nu);
C = zeros(ny, order_total);
D = zeros(ny, nu);

idx = 1; % position de début de bloc

for j = 1:nu
    for i = 1:ny
        n = orders(i,j);
        if n == 0
            D(i,j) = num{i,j}(1)/den{i,j}(1);
            continue;
        end
        
        % Normaliser les coefficients
        a = den{i,j} / den{i,j}(1);
        b = num{i,j} / den{i,j}(1);
        
        % Compléter b pour qu'il ait la même taille que a
        b = [zeros(1, n+1 - length(b)), b];
        
        % Matrices locales
        Ai = [zeros(n-1,1), eye(n-1); -fliplr(a(2:end))];
        Bi = [zeros(n-1,1); 1];
        Ci = b(2:end) - b(1)*a(2:end);
        Ci = fliplr(Ci);
        
        % Insérer dans les blocs globaux
        A(idx:idx+n-1, idx:idx+n-1) = Ai;
        B(idx:idx+n-1, j) = Bi;
        C(i, idx:idx+n-1) = Ci;
        D(i,j) = b(1);
        
        idx = idx + n;
    end
end
end