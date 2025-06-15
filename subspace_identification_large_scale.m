function [Ar, Br, Cr, Dr] = subspace_identification_large_scale(data, order)
    % Perform subspace identification and model reduction for large-scale systems.
    %
    % Parameters:
    % data: A structure containing input and output data
    %   data.u: Input data matrix (NxM)
    %   data.y: Output data matrix (NxP)
    % order: Desired order of the reduced model
    %
    % Returns:
    % Ar, Br, Cr, Dr: State-space matrices of the reduced-order model
    %
    % Explication
    % 
    %     Extraction des Données: Les données d'entrée (u) et de sortie (y) 
    %     sont extraites de la structure data.
    % 
    %     Matrices de Hankel: Les matrices de Hankel sont construites à 
    %     partir des données d'entrée et de sortie de manière efficace en 
    %     utilisant des opérations en bloc.
    % 
    %     Décomposition en Valeurs Singulières (SVD): La SVD est appliquée 
    %     à la matrice de Hankel initiale en utilisant une méthode efficace 
    %     pour les grandes matrices, telle que svds pour obtenir une 
    %     troncature des valeurs singulières.
    % 
    %     Troncature de la SVD: Seules les premières order valeurs 
    %     singulières et les vecteurs correspondants sont conservés pour 
    %     former la séquence d'états.
    % 
    %     Estimation des Matrices d'État: Les matrices d'état du modèle 
    %     réduit (Ar, Br, Cr, Dr) sont estimées à partir de la séquence 
    %     d'états et des données d'entrée et de sortie.

    % Extract input and output data
    u = data.u;
    y = data.y;

    % Determine the number of samples and system dimensions
    [N, m] = size(u); % N: number of samples, m: number of inputs
    [~, p] = size(y); % p: number of outputs

    % Compute the block Hankel matrices efficiently using iterative methods
    L = 2 * order; % Number of block rows in Hankel matrices
    H0 = hankel_matrix_large_scale(u, y, L);
    H1 = hankel_matrix_large_scale([u(2:end, :); zeros(1, m)], y(2:end, :), L);

    % Perform Singular Value Decomposition (SVD) using an efficient method for large matrices
    [U, S, V] = svds(H0, order);

    % Compute the state sequence
    X = S * V';

    % Estimate the state-space matrices
    X1 = X(:, 1:end-1);
    X2 = X(:, 2:end);
    Y = y(L+1:end, :)';
    U = u(L+1:end, :)';

    Ar = (X2 * X1') / (X1 * X1');
    Br = (X2 * U') / (U * U');
    Cr = Y / X1;
    Dr = zeros(p, m); % Assuming zero feedthrough for simplicity

end

function H = hankel_matrix_large_scale(u, y, L)
    % Construct the Hankel matrix for subspace identification efficiently for large-scale systems.
    %
    % Parameters:
    % u: Input data matrix
    % y: Output data matrix
    % L: Number of block rows in the Hankel matrix
    %
    % Returns:
    % H: Hankel matrix

    [N, m] = size(u);
    [~, p] = size(y);

    % Initialize the Hankel matrix
    H = zeros(L * (m + p), N - L + 1);

    % Construct the Hankel matrix using efficient block operations
    for i = 1:L
        H((i-1)*(m+p)+1:i*(m+p), :) = [u(i:N-L+i, :)'; y(i:N-L+i, :)'];
    end
end

% Example usage
% data.u = randn(10000, 2); % Example input data for a large-scale system
% data.y = randn(10000, 1); % Example output data for a large-scale system
% order = 10;
% 
% [Ar, Br, Cr, Dr] = subspace_identification_large_scale(data, order);
% disp('Reduced A matrix:');
% disp(Ar);
% disp('Reduced B matrix:');
% disp(Br);
% disp('Reduced C matrix:');
% disp(Cr);
% disp('Reduced D matrix:');
% disp(Dr);