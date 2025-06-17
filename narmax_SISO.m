function [theta, y_pred, Phi] = narmax_SISO(u, y, na, nb, nk, degree, activation)
    % narmax_activation - Estimation NARMAX avec choix de fonction d'activation
    %
    % Syntaxe:
    %   [theta, y_pred, Phi] = narmax_activation(u, y, na, nb, nk, degree, activation)
    %
    % Entrées:
    %   u          : vecteur d'entrée [Nx1]
    %   y          : vecteur de sortie [Nx1]
    %   na         : ordre autoregressif (sorties retardées)
    %   nb         : ordre entrée (entrées retardées)
    %   nk         : délai d'entrée (décalage)
    %   degree     : degré de la non-linéarité (utilisé uniquement si activation = 'poly')
    %   activation : fonction d'activation à appliquer aux variables retardées, chaîne de caractères parmi :
    %                'poly' (polynôme), 'sigmoid', 'tanh', 'relu', 'linear' (pas de transformation)
    %
    % Sorties:
    %   theta  : coefficients estimés
    %   y_pred : sortie prédite sur les données
    %   Phi    : matrice de régression utilisée
    %
    % Exemple:
    %   [theta,y_pred] = narmax_activation(u, y, 2, 2, 1, 2, 'sigmoid');
    %
    % Script de test pour narmax_siso

    % % Paramètres du modèle
    % na = 2;    % sorties retardées
    % nb = 2;    % entrées retardées
    % nk = 1;    % délai entrée
    % degree = 2; % degré poly (pour 'poly')
    % activation = 'tanh'; % choix fonction d'activation
    % 
    % % Génération données synthétiques
    % N = 500;
    % u = randn(N,1); % entrée aléatoire
    % 
    % % Simuler un système non-linéaire (exemple simple)
    % y = zeros(N,1);
    % for t = max(na, nb+nk-1)+1 : N
    %     y(t) = 0.5*y(t-1) - 0.3*y(t-2) + 0.4*u(t-1) + 0.2*u(t-2) + 0.1*y(t-1)^2 ...
    %         + 0.05*sin(u(t-1)) + 0.1*randn(); % bruit gaussien
    % end
    % 
    % % Estimation NARMAX avec activation choisie
    % [theta, y_pred] = narmax_activation(u, y, na, nb, nk, degree, activation);
    % 
    % % Affichage résultats
    % figure;
    % plot(y, 'b', 'DisplayName', 'Sortie réelle'); hold on;
    % plot([nan(max(na,nb+nk-1),1); y_pred], 'r--', 'DisplayName', 'Sortie prédite');
    % xlabel('Temps');
    % ylabel('Sortie');
    % legend show;
    % title(['Estimation NARMAX avec activation: ', activation]);
    % grid on;
    
    N = length(y);
    max_delay = max(na, nb+nk-1);
    
    % Initialisation matrice retardées
    Phi_raw = [];
    
    % Sorties retardées
    for i=1:na
        Phi_raw = [Phi_raw, y(max_delay - i + 1 : N - i)];
    end
    
    % Entrées retardées
    for j=1:nb
        Phi_raw = [Phi_raw, u(max_delay - nk - j + 2 : N - nk - j + 1)];
    end
    
    % Application de la fonction d'activation
    switch lower(activation)
        case 'poly'
            % Construction polynôme jusqu'au degré spécifié (pas cross terms)
            Phi = ones(size(Phi_raw,1),1); % terme constant
            for d=1:degree
                for col=1:size(Phi_raw,2)
                    Phi = [Phi, Phi_raw(:,col).^d];
                end
            end
    
        case 'sigmoid'
            Phi = 1 ./ (1 + exp(-Phi_raw)); % sigmoïde élément par élément
            Phi = [ones(size(Phi,1),1), Phi]; % ajout terme constant
    
        case 'tanh'
            Phi = tanh(Phi_raw);
            Phi = [ones(size(Phi,1),1), Phi];
    
        case 'relu'
            Phi = max(0, Phi_raw);
            Phi = [ones(size(Phi,1),1), Phi];
    
        case 'linear'
            Phi = [ones(size(Phi_raw,1),1), Phi_raw];
    
        otherwise
            error('Fonction d''activation inconnue. Choisir parmi : poly, sigmoid, tanh, relu, linear.');
    end
    
    % Construction vecteur de sortie cible
    Y_train = y(max_delay+1:end);
    
    % Estimation par moindres carrés
    theta = (Phi' * Phi) \ (Phi' * Y_train);
    
    % Prédiction
    y_pred = Phi * theta;

end
