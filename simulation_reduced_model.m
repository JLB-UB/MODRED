function [y,tcpu]=simulation_reduced_model(A,B,C,D,U,dt)
%
% Simulation du modèle réduit
%
%    dX              du
%   ----- = A X + B ----
%    dt              dt
%
%     T   = C X + D  u
%
% Paramètres d'entrée
%
%	A : 	matrice diagonale des valeurs propres domimantes
%	B : 	matrice d'entrée pour les modes dominants
%	C : 	matrice de sortie pour les modes dominants
%	D : 	matrice statique
%	U : 	vecteur d'entrée
%	dt :	période d'échantillonnage
%
% Paramètre de sortie
%
%	y : 	vecteur de sortie calculé
% 	tcpu :temps de calcul
%
tcpu=cputime;
%
if issparse(A)~=1, A=sparse(A); end
if issparse(B)~=1, B=sparse(B); end
if issparse(C)~=1, C=sparse(C); end
if issparse(D)~=1, D=sparse(D); end
%
[Nd,p]=size(U);
[n,p]=size(B);
[m,n]=size(C);
%
X=B*U(1,:).';
y=sparse(zeros(Nd,m));
%
epz=exp(spdiags(A)*dt);
phi=sparse(diag(epz,0));
gama=C*phi;
S0=gama*B+D;
%
t=dt;
for i = 2:Nd
  t=t+dt;
  fprintf('temps = %8.6f\n',t);   
  y0=S0*U(i,:)'+gama*(X-B*U(i-1,:)');
  y(i,1:m)=y0';
  X=phi*(X+B*(U(i,:)'-U(i-1,:)'));
end
tcpu=cputime-tcpu;