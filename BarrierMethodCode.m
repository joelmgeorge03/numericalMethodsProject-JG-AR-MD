clc;
clear;
close all;

rho = 2700;
L = 10;
EE = 68.9e9;
%mu_vec = [0.0001;0.001;0.01;0.1;1;10;100]; 
mu_vec = [0.01;0.05;0.1;0.5;1];
sigma_y = 276e6;
P = -50000;

k_max = 5000;
k_vec = [];

x_k_vec_mu = zeros(4,length(mu_vec));

f = @(b,h,t1,t2) rho.*L.*((b.*h) - ((h- 2.*t1).*(b-2.*t2)));

syms b h t1 t2 mu

I = (b.*h.^3)./12 - ((h - 2.*t1).^3.*(b - 2.*t2))./12;

h_s = [
h - 1; 
0.1 - h; 
-0.5 - (L.^3.*P)./(3.*EE.*I); 
b - 1; 
0.1 - b; 
-1.5*(h.*L.*P)./(2.*I) - sigma_y;
t1 - 0.25; 
0.01 - t1; 
t2 - 0.25; 
0.01 - t2
];

phi_s = (h*b - ((h - 2*t1)*(b-2*t2)))*rho*L - mu * sum(log(-h_s));

x = [b; h; t1; t2];
grad_s = gradient(phi_s, x);
hess_s = hessian(phi_s, x);

gradfunc = matlabFunction(grad_s, 'Vars', {b, h, t1, t2, mu});
hessfunc = matlabFunction(hess_s, 'Vars', {b, h, t1, t2, mu});
phifunc  = matlabFunction(phi_s,  'Vars', {b, h, t1, t2, mu});
h_vec    = matlabFunction(h_s,    'Vars', {b, h, t1, t2});     


for m = 1:length(mu_vec)
    mu_val = mu_vec(m);
    
    x_k_NM = [0.51; 0.51; 0.051; 0.051];
    x_k_vec_NM(:,1) = x_k_NM;
    epsilon_a = 1000;
    tol_for_problem = 10^-6;
    k = 0;

    
    while epsilon_a > tol_for_problem
        H_k1_phi = hessfunc(x_k_NM(1),x_k_NM(2),x_k_NM(3),x_k_NM(4),mu_val);
    
        subtracted_k1_NM = H_k1_phi\gradfunc(x_k_NM(1),x_k_NM(2),x_k_NM(3),x_k_NM(4),mu_val);
    
        alpha_NM = 1;
    
        while any(h_vec(x_k_NM(1) - alpha_NM*subtracted_k1_NM(1), x_k_NM(2) - alpha_NM*subtracted_k1_NM(2), x_k_NM(3) - alpha_NM*subtracted_k1_NM(3), x_k_NM(4) - alpha_NM*subtracted_k1_NM(4)) > 0) && alpha_NM > 10^-14
            alpha_NM = alpha_NM * 0.1;
        end
        x_k1_NM = x_k_NM - alpha_NM * subtracted_k1_NM;
    
        epsilon_a = norm((x_k1_NM - x_k_NM))/norm(x_k1_NM);
        x_k_NM = x_k1_NM;
        x_k_vec_NM(:,k+2) = x_k_NM;
        k = k + 1;
    end
    x_k_vec_mu(:,m) = x_k_NM;
    k_vec(m) = k;
end

b_NM = x_k_NM(1);
h_NM = x_k_NM(2);
t1_NM = x_k_NM(3);
t2_NM = x_k_NM(4);

%% Contraint Variables

Izz = (1/12).*x_k_vec_mu(1,:).*x_k_vec_mu(2,:).^3 - (1/12).*(x_k_vec_mu(1,:) - 2.*x_k_vec_mu(4,:)).*(x_k_vec_mu(2,:) - 2.*x_k_vec_mu(3,:)).^3;

f(b_NM,h_NM,t1_NM,t2_NM);

sigma_max = -(P*L*(x_k_vec_mu(2,:)/2))./Izz;

FS = abs(sigma_y./sigma_max);

delta_L = (P.*L.^3)./(3.*EE.*((1/12).*x_k_vec_mu(1,:).*x_k_vec_mu(2,:).^3 - (1/12).*(x_k_vec_mu(1,:) - 2.*x_k_vec_mu(4,:)).*(x_k_vec_mu(2,:) - 2*x_k_vec_mu(3,:)).^3));

fprintf('b = %4.2f, h = %4.2f, t_1 = %4.2f, t_2 = %4.2f using Newtons Method \n', x_k_NM(1), x_k_NM(2), x_k_NM(3), x_k_NM(4));


%% Deflection

minima_val2 = 2;

deflection2 = @(x) -P*(3*L*x^2 - x^3)/(6*EE*Izz(minima_val2));
b_vals2 = x_k_vec_mu(1,minima_val2);
h_vals2 = x_k_vec_mu(2,minima_val2);
t1_vals2 = x_k_vec_mu(3,minima_val2);
t2_vals2 = x_k_vec_mu(4,minima_val2);

x_plot = linspace(0, L, 201);
defl_vals2 = P.*(3*L.*x_plot.^2 - x_plot.^3)./(6*EE*Izz(minima_val2));

minima_val3 = 3;

deflection3 = @(x) -P*(3*L*x^2 - x^3)/(6*EE*Izz(minima_val3));
b_vals3 = x_k_vec_mu(1,minima_val3);
h_vals3 = x_k_vec_mu(2,minima_val3);
t1_vals3 = x_k_vec_mu(3,minima_val3);
t2_vals3 = x_k_vec_mu(4,minima_val3);
defl_vals3 = P.*(3*L.*x_plot.^2 - x_plot.^3)./(6*EE*Izz(minima_val3));


p5 = plot(x_plot, defl_vals2, 'r-', 'LineWidth', 1.5);
hold on

p6 = plot(x_plot, defl_vals3, 'b--', 'LineWidth', 1.5);
grid on;
xlabel('x (m)');
ylabel('Deflection (m)');
title('Deflection along Beam Length');

legend([p5 p6],{'Initial Barrier Solution: \mu = 0.1','Final Barrier Solution: \mu = 0.05'},FontSize=12);
%% Objective Function

% evaluate f for each design (columns of x_k_vec_mu) and plot versus mu
f_vals = zeros(1, size(x_k_vec_mu,2));
for i = 1:size(x_k_vec_mu,2)
    f_vals(i) = f(x_k_vec_mu(1,i), x_k_vec_mu(2,i), x_k_vec_mu(3,i), x_k_vec_mu(4,i))
end
semilogx(mu_vec, f_vals, '-o','LineWidth',1.2);
hold on
grid on;
xlabel('\mu');
ylabel('Mass (kg)');
title('Mass (Objective Function)')
p3 = xline(0.1,LineStyle='--',LineWidth=2,Color='r');
xlim([min(mu_vec), max(mu_vec)])
%p4 = xline(0.05,LineStyle='--',LineWidth=2,Color='b');
legend([p3],{'Initial Barrier Solution: \mu = 0.1','Final Barrier Solution: \mu = 0.05'},FontSize=12);
%% Plots
mu_plot = mu_vec;

sgtitle('The Effect of \mu on the Design Variables, Factor of Safety, and \delta_L')
subplot(2,3,1);
semilogx(mu_plot, x_k_vec_mu(1,:), '-o','LineWidth',1.2);
title('b')
hold on
yline(0.1,LineStyle='--',LineWidth=2)
yline(1,LineStyle='--',LineWidth=2)
p1 = xline(0.1,LineStyle='--',LineWidth=2,Color='r');
p2 = xline(0.05,LineStyle='--',LineWidth=2,Color='b');
grid on;
xlabel('\mu');
ylabel('b (m)');
xlim([min(mu_vec), max(mu_vec)])

subplot(2,3,2);
semilogx(mu_plot, x_k_vec_mu(2,:), '-o','LineWidth',1.2);
title('h')
hold on
yline(0.1,LineStyle='--',LineWidth=2)
yline(1,LineStyle='--',LineWidth=2)
xline(0.1,LineStyle='--',LineWidth=2,Color='r')
xline(0.05,LineStyle='--',LineWidth=2,Color='b');
grid on;
xlabel('\mu');
ylabel('h (m)');
xlim([min(mu_vec), max(mu_vec)])

subplot(2,3,4);
semilogx(mu_plot, x_k_vec_mu(3,:), '-o','LineWidth',1.2);
title('t_1')
hold on
yline(0.01,LineStyle='--',LineWidth=2)
yline(.25,LineStyle='--',LineWidth=2)
xline(0.1,LineStyle='--',LineWidth=2,Color='r')
xline(0.05,LineStyle='--',LineWidth=2,Color='b');
grid on;
xlabel('\mu');
ylabel('t_1 (m)');
xlim([min(mu_vec), max(mu_vec)])

subplot(2,3,5);
semilogx(mu_plot, x_k_vec_mu(4,:), '-o','LineWidth',1.2);
title('t_2')
hold on
yline(0.01,LineStyle='--',LineWidth=2)
yline(.25,LineStyle='--',LineWidth=2)
xline(0.1,LineStyle='--',LineWidth=2,Color='r')
xline(0.05,LineStyle='--',LineWidth=2,Color='b');
grid on;
xlabel('\mu');
ylabel('t_2 (m)');
xlim([min(mu_vec), max(mu_vec)])

subplot(2,3,3);
semilogx(mu_plot, FS, '-o','LineWidth',1.2);
title('FS')
hold on
yline(1.5,LineStyle='--',LineWidth=2)
xline(0.1,LineStyle='--',LineWidth=2,Color='r')
xline(0.05,LineStyle='--',LineWidth=2,Color='b');
grid on;
grid on;
xlabel('\mu');
ylabel('Factor of Safety');
xlim([min(mu_vec), max(mu_vec)])

subplot(2,3,6);
semilogx(mu_plot, delta_L, '-o','LineWidth',1.2);
title('\delta_L')
hold on
yline(-0.5,LineStyle='--',LineWidth=2)
xline(0.1,LineStyle='--',LineWidth=2,Color='r')
xline(0.05,LineStyle='--',LineWidth=2,Color='b');
grid on;
xlabel('\mu');
ylabel('\delta_L (m)');
xlim([min(mu_vec), max(mu_vec)])
legend([p1 p2],{'Initial Barrier Solution: \mu = 0.1','Final Barrier Solution: \mu = 0.05'},FontSize=12);

