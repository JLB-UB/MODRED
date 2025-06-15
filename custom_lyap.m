function X = custom_lyap(A, Q)
% Solve the continuous-time Lyapunov equation: A*X + X*A' + Q = 0
%
% Parameters:
% A: Square matrix
% Q: Symmetric matrix
%
% Returns:
% X: Solution to the Lyapunov equation

% Get the size of the matrix A
n = size(A, 1);

% Formulate the Kronecker product and vectorize Q
I = eye(n);
K = kron(I, A) + kron(A', I);
q = -Q(:);

% Solve the linear system
x = K \ q;

% Reshape the solution vector into a matrix
X = reshape(x, n, n);
end