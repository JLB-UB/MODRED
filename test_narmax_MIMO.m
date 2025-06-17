% Exemple complet NARMAX MIMO avec validation croisée

% --- Données simulées ---
N = 600; % nombre de points temps
m = 2;  % nombre d'entrées
p = 2;  % nombre de sorties

rng(1); % seed pour reproductibilité
U = randn(N,m);
Y = zeros(N,p);

for t=4:N
    Y(t,1) = 0.5*Y(t-1,1) - 0.3*Y(t-2,2) + 0.4*U(t-1,1) + 0.2*U(t-2,2)^2 + 0.1*tanh(U(t-1,1));
    Y(t,2) = 0.3*Y(t-1,2) + 0.4*Y(t-2,1) + 0.3*U(t-1,2) + 0.1*U(t-2,1) + 0.1*sin(U(t-1,2));
    Y(t,:) = Y(t,:) + 0.05*randn(1,p); % bruit
end

% --- Paramètres ---
na = 2; nb = 2; nk = [1 1];

% Validation croisée pour le degré polynomial
degrees = 1:4;
activations = {'linear', 'poly', 'tanh', 'sigmoid', 'relu'};

N_train = 400;
N_val = N - N_train - max([na, max(nb + nk - 1)]);

% Découpage données
U_train = U(1:N_train, :);
Y_train = Y(1:N_train, :);

U_val = U(N_train+1:end, :);
Y_val = Y(N_train+1:end, :);

best_mse = Inf;
best_params = struct();

fprintf('Validation croisée NARMAX MIMO\n');

for act = activations
    actv = act{1};
    if strcmp(actv, 'poly')
        % Test sur plusieurs degrés
        for d = degrees
            try
                [~, Y_pred, ~] = narmax_mimo_activation(U_train, Y_train, na, nb, nk, d, actv);
                % Y_pred dimension = N_train - max_delay x p
                max_delay = max([na, max(nb + nk - 1)]);
                Y_true = Y_train(max_delay+1:end, :);
                mse = mean((Y_true - Y_pred).^2, 'all');

                % Validation sur données val
                [~, Y_val_pred, ~] = narmax_mimo_activation(U_val, Y_val, na, nb, nk, d, actv);
                Y_val_true = Y_val(max_delay+1:end, :);
                mse_val = mean((Y_val_true - Y_val_pred).^2, 'all');

                fprintf('Activation=%s, degree=%d, Train MSE=%.4f, Val MSE=%.4f\n', actv, d, mse, mse_val);

                if mse_val < best_mse
                    best_mse = mse_val;
                    best_params.activation = actv;
                    best_params.degree = d;
                end
            catch ME
                warning('Erreur avec activation %s et degré %d : %s', actv, d, ME.message);
            end
        end
    else
        try
            d = 1; % sans notion de degré, mais à passer quand même
            [~, Y_pred, ~] = narmax_mimo_activation(U_train, Y_train, na, nb, nk, d, actv);
            max_delay = max([na, max(nb + nk - 1)]);
            Y_true = Y_train(max_delay+1:end, :);
            mse = mean((Y_true - Y_pred).^2, 'all');

            [~, Y_val_pred, ~] = narmax_mimo_activation(U_val, Y_val, na, nb, nk, d, actv);
            Y_val_true = Y_val(max_delay+1:end, :);
            mse_val = mean((Y_val_true - Y_val_pred).^2, 'all');

            fprintf('Activation=%s, Train MSE=%.4f, Val MSE=%.4f\n', actv, mse, mse_val);

            if mse_val < best_mse
                best_mse = mse_val;
                best_params.activation = actv;
                best_params.degree = d;
            end
        catch ME
            warning('Erreur avec activation %s : %s', actv, ME.message);
        end
    end
end

fprintf('\nMeilleur modèle trouvé:\n Activation: %s\n Degré: %d\n Validation MSE: %.4f\n', ...
    best_params.activation, best_params.degree, best_mse);

% --- Estimation finale sur toutes données avec meilleurs paramètres ---
[Theta, Y_pred_full, ~] = narmax_mimo_activation(U, Y, na, nb, nk, best_params.degree, best_params.activation);

% Affichage sorties réelles vs prédites (pour la 1ère sortie)
max_delay = max([na, max(nb + nk - 1)]);
figure;
plot(Y(max_delay+1:end,1), 'b', 'DisplayName', 'Sortie réelle');
hold on;
plot(Y_pred_full(:,1), 'r--', 'DisplayName', 'Sortie prédite');
legend;
title(sprintf('NARMAX MIMO - Sortie 1 (%s, degree=%d)', best_params.activation, best_params.degree));
xlabel('Temps');
ylabel('Amplitude');
