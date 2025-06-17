function rmse = compute_rmse(y_true, y_pred)
% compute_rmse - Calcule la racine de l'erreur quadratique moyenne
rmse = sqrt(mean((y_true - y_pred).^2));
end
