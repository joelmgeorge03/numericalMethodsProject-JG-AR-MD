% Name: Joel George
% Date: 4/23/26
% This script is our group's attempt at formulating and implementing an SQP
% method to solve the nonlinear optimization problem
% of our project, which seeks to minimize the mass of a cantilvered beam
% with a tip load, subject to inequality constraints on the dimensions of
% the beam cross section and stress/deflection constraints. 


%% Setting up Script: 
clear; clc; close all; 

%% Model Geometry, Properties, and Loads

% Geometry Inputs:
L_beam = 10; % cantilevered beam length, meters

% Material Properties (Al-6061 T6) from: https://asm.matweb.com/search/specificmaterial.asp?bassnum=ma6061t6
rho_beam = 2700; % density, kg/m^3
E_beam = 68.9e9; % Young's Modulus, Pa
sigma_yield = 276e6; % Tensile Yield Strength, Pa
nu_beam = 0.33; % Poisson's Ratio

% Loads: 
P_applied = -50000; % applied Tip Load, Newtons

%% Defining Governing Equations:

% Objective Function
f = @(x) rho_beam*L_beam*( (x(1))*(x(2)) - (x(1) - 2*x(4))*(x(2) - 2*x(3)) );

% Gradient of Objective Function
gradf = @(x) [2*L_beam*rho_beam*x(3);
              2*L_beam*rho_beam*x(4);
              2*L_beam*rho_beam*(x(1) - 2*x(4));
              2*L_beam*rho_beam*(x(2) - 2*x(3));]; 

% Constraint Vector - derived by hand
h1 = @(x) 0.1 - x(1);
h2 = @(x) x(1) - 1;
h3 = @(x) 0.1 - x(2);
h4 = @(x) x(2) - 1;
h5 = @(x) 0.01 - x(3);
h6 = @(x) x(3) - 0.25;
h7 = @(x) 0.01 - x(4);
h8 = @(x) x(4) - 0.25;
h9 = @(x) (P_applied*L_beam^3)/( 3*E_beam*( (1/12)*x(1)*x(2)^3 - (1/12)*(x(1) - 2*x(4))*(x(2) - 2*x(3))^3 ) ) - 0.5;
h10 = @(x) -(P_applied*L_beam*x(2)/2)/(( (1/12)*x(1)*x(2)^3 - (1/12)*(x(1) - 2*x(4))*(x(2) - 2*x(3))^3 ) ) - sigma_yield/1.5;

h = @(x) [h1(x); 
          h2(x);
          h3(x);
          h4(x);
          h5(x);
          h6(x);
          h7(x);
          h8(x);
          h9(x);
          h10(x)];


%% Constraint Jacobian - derived in Mathematica

% 8 Linear Constraints (min and max beam cross-section dimensions)
J_h_1 = @(x) [-1 0 0 0];
J_h_2 = @(x) [1 0 0 0];
J_h_3 = @(x) [0 -1 0 0];
J_h_4 = @(x) [0 1 0 0];
J_h_5 = @(x) [0 0 -1 0];
J_h_6 = @(x) [0 0 1 0];
J_h_7 = @(x) [0 0 0 -1];
J_h_8 = @(x) [0 0 0 1];

% 2 Nonlinear Constraints (Max Tip Deflection & Max Bending Stress)
denom = @(x) ( x(1)*x(3)*( 3*x(2)^2 - 6*x(2)*x(3) + 4*x(3)^2 ) + ( x(2) - 2*x(3) )^3*x(4) )^2;

J_h_9 = @(x) (1/(E_beam*denom(x)))*[2*L_beam^3*P_applied*x(3)*(3*x(2)^2 - 6*x(2)*x(3) + 4*x(3)^2), ...
     6*L_beam^3*P_applied*( 2*x(1)*x(3)*( x(2) - x(3)) + (x(2)-2*x(3))^2*x(4) ), ...
     6*L_beam^3*P_applied*(x(2) - 2*x(3))^2*(x(1) - 2*x(4)), ...
     2*L_beam^3*P_applied*(x(2) - 2*x(3))^3];

J_h_10 = @(x) (1/(denom(x)))*[3*L_beam*P_applied*x(2)*x(3)*(3*x(2)^2 - 6*x(2)*x(3) + 4*x(3)^2), ...
     3*L_beam*P_applied*(x(1)*(3*x(2)^2*x(3) - 4*x(3)^3) + 2*(x(2) - 2*x(3))^2*(x(2) + x(3))*x(4)), ...
     9*L_beam*P_applied*x(2)*(x(2) - 2*x(3))^2*(x(1) - 2*x(4)), ...
     3*L_beam*P_applied*x(2)*(x(2) - 2*x(3))^3];

% Combining all components into single vector
J_h = @(x) [J_h_1(x); 
            J_h_2(x); 
            J_h_3(x); 
            J_h_4(x); 
            J_h_5(x); 
            J_h_6(x); 
            J_h_7(x); 
            J_h_8(x); 
            J_h_9(x); 
            J_h_10(x)];

% Gradient of Lagrangian (for BFGS update)
gradL = @(x,lambda) gradf(x) + J_h(x)'*lambda;

%% Initializing SQP solver:
%x0 = [0.5; 0.5; 0.1; 0.1]; % initial guess for sizing

x0 = [0.51; 0.51; 0.051; 0.051]; % other initial guess
lambda0 = zeros(10,1); % initial guess of Lagrange Mulitipliers
HessMat0 = eye(length(x0)); % initial guess for Hessian Matrix
solTol = 1e-6; % solution tolerance
maxIter = 1000; % maximum iteration count

%% SQP Algorithm - Active Set Method with BFGS Hessian Update

x_current = x0; % set initial guess for design variables
lambda_current = lambda0; % set initial guess for Lagrange Multipliers
HessMat = HessMat0; % set initial guess for the Hessian of the Lagrangian (this gets updated using BFGS)

numDVs = length(x_current);
numCons = length(lambda_current);

history.dv_vals = [];
history.objval = [];
history.activeCons = cell(1,maxIter);
history.lambda = cell(1,maxIter);
history.iter = [];
qp_options = optimoptions('quadprog', 'Algorithm','active-set','Display','iter','MaxIterations',1000);

for k = 1:maxIter % begin loop

    % Evaluate objective gradient, constraints, and constraint Jacobian at current point
    gradf_current = gradf(x_current);
    h_current = h(x_current);
    J_h_current = J_h(x_current);
    
    % Setup for quadprog
    x0_quadprog = zeros(length(x_current), 1);
    H_quadprog = (HessMat + HessMat')/2;
    f_quadprog = gradf_current;
    A_quadprog = J_h_current;
    b_quadprog = -h_current;
    
    % Solve the QP subproblem using quadprog
    [dvStep, ~, exitflag, output, lambda_out] = quadprog(H_quadprog, f_quadprog, A_quadprog, b_quadprog, [], [], [], [], x0_quadprog, qp_options);
    lambda_new = lambda_out.ineqlin;

    % Advice from Dr. Wang - use a small step size along the calculated step direction
    alpha = 0.44; % tune this to provide a suitable solution - should really be a line search   
    % alpha = 1.0;
    x_new = x_current + alpha*dvStep;
    % maxLS_iter = 75;
    % LS_iter = 0;
    % while any(h(x_new) > 0) && LS_iter < maxLS_iter % this is outside the feasible region
    %     alpha = alpha*0.5;
    %     x_new = x_current  + alpha*dvStep; % Update the guess again
    %     LS_iter = LS_iter + 1;
    %     if LS_iter   == maxLS_iter
    %         disp('Line search did not converge within max iterations.')
    %     else
    %         % nothing
    %     end
    % end
     
    % Update the Hessian using BFGS (Based on Algorithm 6.5 from Heath)
    y_current = gradL(x_new, lambda_new) - gradL(x_current, lambda_new); % Gradient difference
    s_current = x_new - x_current; % Step difference
    HessMat = HessMat + (y_current * y_current') / (y_current' * s_current) - (HessMat * (s_current * s_current') * HessMat) / (s_current' * HessMat * s_current);
    
    % Update design variable guess and Lagrange multiplier guess
    x_current = x_new; % update the design variables
    lambda_current = lambda_new; % update Lagrange Multipliers
    objval = f(x_current); % compute the mass of the beam
    
    % Identifying the active constraints
    h_active_idx = find(lambda_current > 1e-5);

    % Computing Tip Deflections and Factor of Safety
    tipDef = (P_applied*L_beam^3)/( 3*E_beam*( (1/12)*x_current(1)*x_current(2)^3 - (1/12)*(x_current(1) - 2*x_current(4))*(x_current(2) - 2*x_current(3))^3 ) );
    FoS = sigma_yield/( (-P_applied*L_beam*x_current(2)/2)/(( (1/12)*x_current(1)*x_current(2)^3 - (1/12)*(x_current(1) - 2*x_current(4))*(x_current(2) - 2*x_current(3))^3 ) ) );
    
    % Store values and active constraints
    history.dv_vals(:, k) = x_current; % Store the current design variable guess
    history.lambda{k} = lambda_current; % Store the current Lagrange multipliers
    history.objval(:,k) = objval; % Store the current beam mass
    history.activeCons{k} = h_active_idx; % store the current active constraints
    history.tipDef(:, k) = tipDef; % Store the current tip deflection
    history.FoS(:,k) = FoS;
    history.iter(k) = k; % Store the current iteration number

    % Check for convergence
    if norm(dvStep) < solTol
        disp(['Converged in ', num2str(k), ' iterations.'])
        break; 
    end

end

figure(Theme="light")

% Plotting the history of design variable values and objective function
subplot(2,1,1);
plot(history.iter, history.objval, '-o', 'LineWidth',2);
grid minor
xlabel('Iteration');
ylabel('Objective Value (Beam Mass)');
title('Objective Value Convergence');

subplot(2,1,2);
plot(history.iter, history.dv_vals(1,:), '-o', 'LineWidth',2)
hold on; grid minor
plot(history.iter, history.dv_vals(2,:), '-o', 'LineWidth',2)
plot(history.iter, history.dv_vals(3,:), '-o', 'LineWidth',2)
plot(history.iter, history.dv_vals(4,:), '-o', 'LineWidth',2);

yline(0.01, 'r--')
yline(0.25, 'g--')
yline(0.1, 'b--')
yline(1, 'k--')

xlabel('Iteration');
ylabel('Design Variables');
ylim([-0.5 1.1])
legend('DV1', 'DV2', 'DV3', 'DV4', 'Wall Thickness Lower Bound', 'Wall Thickness Upper Bound', 'Beam Width/Height Lower Bound', 'Beam Width/Height Upper Bound', 'Location','best');
title('Design Variables Convergence');