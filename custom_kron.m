function K = custom_kron(A, B)
    %KRON Kronecker tensor product.
    %   K = KRON(A,B) returns the Kronecker tensor product of matrices A and B.
    %   The result is a large matrix formed by multiplying each element of A by
    %   the entire matrix B.
    %
    %   Example:
    %       A = [1 2; 3 4];
    %       B = [0 5; 6 7];
    %       K = kron(A, B);
    %       % Result:
    %       % K =
    %       %      0     5     0    10
    %       %      6     7    12    14
    %       %      0    15     0    20
    %       %     18    21    24    28
    
    % Get the dimensions of the input matrices
    [m, n] = size(A);
    [p, q] = size(B);
    
    % Initialize the result matrix
    K = zeros(m * p, n * q);
    
    % Compute the Kronecker product
    for i = 1:m
        for j = 1:n
            K((i-1)*p + (1:p), (j-1)*q + (1:q)) = A(i,j) * B;
        end
    end
end