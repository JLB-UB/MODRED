function [Ar,Br,Cr,Dr,q_cont_v] = modal_reduction_litz(A,B,C,D,r,nc)
%
% The Litz algorithm performs the state model reduction by modal troncature
% and aggregation
% The Lanczos or Arnoldi algorithms are used to perform the calculation of
% nc eigenmodes (nc is taken large compare to the reduction order r but
% smaller than the dimension of the original state model)
% The agregation is performed following an optimal path by minimizing the
% gap between the response of the complete and reduced model
% description of the procedure is given in:
%
% INPUT variables :
%
% The input variables are related to the complete state-space model:
%
%       dT
%      ---- = A T + B u
%       dt
%
%       Y   = C T + D u
%
%
%   A       state matrix
%	B       input matrix
%	C       output matrix
%	D       static matrix
%	r       reduction order
%   nc      number of computed eigenmodes
%		
% OUTPUT variables :
%
% The output variables are relmated to the matrices of the reduced model
%
%       dX
%      ---- = Ar X + Br u
%       dt
%
%       Y   = C T + S u
% 
%   Ar       	diagonal state matrix (eigenvalues)
%	Br          input matrix
%	Cr          output matrix
%	Dr          correction matrix
%   q_cont_v    contribution of the modes
%
% see also: EIGS
%
% Author: Pr. Jean-Luc Battaglia
% Affiliation: I2M Lab. University of Bordeaux
% email: jean-luc.battaglia@u-bordeaux.fr
% update: July 2021
%
[m,~] = size(C);
[~,p] = size(B);
% 
% computation of the modal basis using the Arnoldi algorithm
%
[P_eig,Ad] = eigs(A,nc,0);
Bd = P_eig\B;
Cd = C*P_eig;
%
% calculation of the modal dominance
%
q_cont = zeros(nc,1);
for k = 1:nc
    for i = 1:m	
        for j = 1:p
            q_cont(k)=q_cont(k)+abs(Cd(i,k)*Bd(k,j)/Ad(k,k));
        end
    end
end
% ordering the modes by decresing dominance
[q_cont_v,Iq] = sort(q_cont,'descend');
% creation of the sub state-space model based on the nc calculated
% eigenmodes
Am = Ad(Iq,Iq);
Bm = Bd(Iq,:);
Cm = Cd(:,Iq);
%
% splitting the model in dominant and non-dominant parts
Ar1 = Am(1:r,1:r);
Ar2 = Am(r+1:end,r+1:end);
Br1 = Bm(1:r,:);
Br2 = Bm(r+1:end,:);
Cr1 = Cm(:,1:r);
Cr2 = Cm(:,r+1:end);
%
% the contribution of the non-dominant modes will be aggregated to the
% dominant ones
Qp = eye(p);
F0 = Br2*Qp*Br1';
F1 = Br1*Qp*Br1';
S = zeros(nc-r,r);
T = zeros(r,r);
for i = 1:nc-r
    for j = 1:r
        S(i,j) = -F0(i,j)/(Am(r+i,r+i)+Am(j,j));
    end
end
for i = 1:r
    for j = 1:r
        T(i,j) = -F1(i,j)/(Am(i,i)+Am(j,j));
    end
end
E11 = Br2-(S*(T\Br1));
E22 = (Br1'/T)*Br1;
E33 = E22\Br1';
E44 = E11*E33;
E55 = S+E44;
E66 = E55/T;
E77 = E66*Ar1;
E88 = eye(nc-r,nc-r).*(1./spdiags(Ar2));
E = E88*E77;
%
% reduced model
Ar = Ar1;
Br = Br1;
Cr = Cr1+Cr2*E;
Dr = D;