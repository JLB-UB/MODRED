% test_narmax_validation.m
% Test complet NARMAX avec validation croisée automatique

clear; close all; clc;

% Données synthétiques
N = 600;
u = randn(N,1);
y = zeros(N,1);
for t = 4:N
    y(t) = 0.5*y(t-1) - 0.3*y(t-2) + 0.4*u(t-1) + 0.2*u(t-2) + 0.1*y(t-1)^2 ...
        + 0.05*sin(u(t-1)) + 0.1*randn();
end

% Séparation train/test
train_ratio = 0.7;
N_train = floor(train_ratio * N);

u_train = u(1:N_train);
y_train = y(1:N_train);
u_test = u(N_train+1:end);
y_test = y(N_train+1:end);

% Grilles paramètres
na_list = [1 2 3];
nb_list = [1 2 3];
nk_list = [1 2];
degree_list = [1 2 3];
activations = {'linear', 'poly', 'sigmoid', 'tanh', 'relu'};

best_rmse = Inf;
best_params = struct();

for na = na_list
    for nb = nb_list
        for nk = nk_list
            for degree = degree_list
                for a = 1:length(activations)
                    activation = activations{a};
                    try
                        [theta, ~] = narmax_activation(u_train, y_train, na, nb, nk, degree, activation);

                        % Construction matrice test
                        max_delay = max(na, nb + nk -1);
                        if length(y_test) <= max_delay
                            error('Trop peu de données test pour retards spécifiés');
                        end

                        Phi_raw_test = [];
                        for i=1:na
                            Phi_raw_test = [Phi_raw_test, y_test(max_delay - i + 1:end - i)];
                        end
                        for j=1:nb
                            Phi_raw_test = [Phi_raw_test, u_test(max_delay - nk - j + 2:end - nk - j +1)];
                        end

                        switch lower(activation)
                            case 'poly'
                                Phi_test = ones(size(Phi_raw_test,1),1);
                                for d=1:degree
                                    for col=1:size(Phi_raw_test,2)
                                        Phi_test = [Phi_test, Phi_raw_test(:,col).^d];
                                    end
                                end
                            case 'sigmoid'
                                Phi_test = 1 ./ (1 + exp(-Phi_raw_test));
                                Phi_test = [ones(size(Phi_test,1),1), Phi_test];
                            case 'tanh'
                                Phi_test = tanh(Phi_raw_test);
                                Phi_test = [ones(size(Phi_test,1),1), Phi_test];
                            case 'relu'
                                Phi_test = max(0, Phi_raw_test);
                                Phi_test = [ones(size(Phi_test,1),1), Phi_test];
                            case 'linear'
                                Phi_test = [ones(size(Phi_raw_test,1),1), Phi_raw_test];
                        end

                        y_pred_test = Phi_test * theta;
                        y_true_test = y_test(max_delay+1:end);

                        rmse_test = compute_rmse(y_true_test, y_pred_test);

                        if rmse_test < best_rmse
                            best_rmse = rmse_test;
                            best_params.na = na;
                            best_params.nb = nb;
                            best_params.nk = nk;
                            best_params.degree = degree;
                            best_params.activation = activation;
                            best_params.theta = theta;
                            best_params.y_pred_test = y_pred_test;
                            best_params.y_true_test = y_true_test;
                        end
                    catch
                        % Ignorer erreurs (ex: données insuffisantes)
                    end
                end
            end
        end
    end
end

fprintf('Meilleur RMSE : %.4f avec na=%d, nb=%d, nk=%d, degree=%d, activation=%s\n', ...
    best_rmse, best_params.na, best_params.nb, best_params.nk, best_params.degree, best_params.activation);

figure;
plot(best_params.y_true_test, 'b', 'DisplayName', 'Sortie test réelle'); hold on;
plot(best_params.y_pred_test, 'r--', 'DisplayName', 'Sortie test prédite');
xlabel('Temps');
ylabel('Sortie');
legend show;
title('Meilleur modèle NARMAX par validation croisée');
grid on;
