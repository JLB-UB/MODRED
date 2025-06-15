clearvars
%% model de base
L = 0.1;          % épaisseur [m]
Nx = 100;         % nombre de points en espace
k = 10;           % conductivité thermique [W/m.K]
rho_cp = 1.0e5;   % capacité thermique volumique [J/m3.K]
%
r = 20;            % ordre de réduction
m = 4*r;
%
[A,B,C,D] = tutorial(L, Nx,k,rho_cp);
sys = sparss(A,B,C,D);
% --- Exemple de simulation avec ode45
u_fun = @(t) 100;  % flux constant 100 W/m2
% Fonction dynamique dx/dt = A x + B u(t)
odefun = @(t,x) A*x + B*u_fun(t);
% Condition initiale nulle
[n,~] = size(A);
x0 = zeros(n,1);
% Simulation temporelle
tspan = [0 10];
[t_sol_0, x_sol] = ode45(odefun, tspan, x0);    
% Calcul sortie
y_sol_0 = (C * x_sol')';
% --- Affichage
figure(1)
plot(t_sol_0, y_sol_0, 'LineWidth', 1.5)
hold on
title('Réponse indicielle')
xlabel('Temps (s)')
ylabel('Température en x = L/2 (°C)')

%% réduction modale
[Ar,Br,Cr,Dr] = modal_reduction(A,B,C,D,m,r);
% --- Simulation du système réduit avec ode45
u_fun = @(t) 100; % flux constant    
odefun_r = @(t,z) Ar*z + Br*u_fun(t);
z0 = zeros(r,1);
tspan = [0 10];
[t_sol_1, z_sol] = ode45(odefun_r, tspan, z0);
% --- Sortie réduite
y_r_1 = (Cr * z_sol')' + Dr * u_fun(t_sol_1)' ;
% --- Affichage
figure(1), plot(t_sol_1, y_r_1, '+','LineWidth',1.5)
y_r_1 = interp1(t_sol_1,y_r_1,t_sol_0);
figure(2), plot(t_sol_0, y_sol_0-y_r_1, 'o', 'LineWidth',1.5)
title('erreur Norme-1')
hold on
%% réduction POD
[Ar, Br, Cr, Dr] = pod_reduction(A, B, C, D, m, r);
% --- Simulation du système réduit avec ode45
u_fun = @(t) 100; % flux constant    
odefun_r = @(t,z) Ar*z + Br*u_fun(t);
z0 = zeros(r,1);
tspan = [0 10];
[t_sol_2, z_sol] = ode45(odefun_r, tspan, z0);
% --- Sortie réduite
y_r_2 = (Cr * z_sol')' + Dr * u_fun(t_sol_2)' ;
% --- Affichage
figure(1), plot(t_sol_2, y_r_2, 's', 'LineWidth',1.5)
y_r_2 = interp1(t_sol_2,y_r_2,t_sol_0);
figure(2), plot(t_sol_0, y_sol_0-y_r_2, 'o', 'LineWidth',1.5)
hold on
%% réduction équilibrée
dt = 0.01;
T = 10;
r = 4;
[Ar,Br,Cr,Dr] = balanced_pod_reduction_ode(A,B,C,D,r,dt,T);
% --- Simulation du système réduit avec ode45
u_fun = @(t) 100; % flux constant    
odefun_r = @(t,z) Ar*z + Br*u_fun(t);
z0 = zeros(r,1);
tspan = [0 10];
[t_sol_3, z_sol] = ode45(odefun_r, tspan, z0);
% --- Sortie réduite
y_r_3 = (Cr * z_sol')' + Dr * u_fun(t_sol_3)' ;    
% --- Affichage
figure(1), plot(t_sol_3, y_r_3, 'o', 'LineWidth',1.5)
legend('Model Complet (100)', 'réduction MODALE (20)', 'Réduction POD (20)','Réduction Equilibrée (4)')
grid on
y_r_3 = interp1(t_sol_3,y_r_3,t_sol_0);
figure(2), plot(t_sol_0, y_sol_0-y_r_3, 'o', 'LineWidth',1.5)
legend('réduction MODALE (20)', 'réduction POD (20)','réduction équilibrée (4)')
grid on