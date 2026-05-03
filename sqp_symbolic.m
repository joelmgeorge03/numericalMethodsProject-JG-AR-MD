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

% Material Properties (Al-6061 T6)
rho_beam = 2700; % density, kg/m^3
E_beam = 68.9e9; % Young's Modulus, Pa
sigma_yield = 276e6; % Tensile Yield Strength, Pa
nu_beam = 0.33; % Poisson's Ratio

% Loads: 
P_applied = -50000; % applied Tip Load, Newtons

%% Defining Governing Equations:
x_sym = sym('x', [4 1]);

% Objective Function
f_sym = rho_beam*L_beam*( (x_sym(1))*(x_sym(2)) - (x_sym(1) - 2*x_sym(4))*(x_sym(2) - 2*x_sym(3)) );
f = matlabFunction(f_sym, 'Vars', {x_sym});

% Gradient of Objective
gradf_sym =  gradient(f_sym, x_sym);
gradf = matlabFunction(gradf_sym, 'Vars', {x_sym});

% Constraint Vector
Izz_sym = (1/12)*(x_sym(1)*x_sym(2)^3) - (1/12)*( (x_sym(1) - 2*x_sym(4) )*( x_sym(2) - 2*x_sym(3) )^3);

h_sym = [0.1 - x_sym(1); 
         x_sym(1) - 1;
         0.1 - x_sym(2); 
         x_sym(2) - 1;
         0.01 - x_sym(3);
         x_sym(3) - 0.25;
         0.01 - x_sym(4);
         x_sym(4) - 0.25;
         -(P_applied*L_beam^3)/(3*E_beam*Izz_sym) - 0.5;
         -1.5*(P_applied*L_beam*x_sym(2)/2)/(Izz_sym) - sigma_yield];
h = matlabFunction(h_sym, 'Vars', {x_sym});

% Constraint Jacobian
J_h_sym = jacobian(h_sym, x_sym);
J_h = matlabFunction(J_h_sym, 'Vars', {x_sym});   

% Formulate Lagrange Multipliers and Lagrangian
lambda_sym = sym('lambda', [10 1]);
L_sym = f_sym + lambda_sym.' * h_sym;

% Gradient of Lagrangian
gradL_sym = gradient(L_sym, x_sym);
gradL = matlabFunction(gradL_sym, 'Vars', {x_sym, lambda_sym});


%% Initializing SQP solver:
x0 = [0.5; 0.5; 0.1; 0.1]; % initial guess for sizing
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
    x0_quadprog = zeros(length(x_current), 1); % start with 0 
    H_quadprog = (HessMat + HessMat')/2; % ensure the Hessian is symmetric
    f_quadprog = gradf_current; % pass in linear objective term
    A_quadprog = J_h_current; % pass in linearized inequality constraints
    b_quadprog = -h_current; % pass in linearized inequality constraints
    
    % Solve the QP subproblem using quadprog
    [dvStep, ~, exitflag, output, lambda_out] = quadprog(H_quadprog, f_quadprog, A_quadprog, b_quadprog, [], [], [], [], x0_quadprog, qp_options);
    lambda_new = lambda_out.ineqlin;

    % Advice from Dr. Wang - use a small step size along the calculated step direction
    alpha = 0.44; % tune this to provide a suitable solution - should really be a line search   
    % alpha = 1.0; % base case - produced weird results
    x_new = x_current + alpha*dvStep;

    % Similar Feasibility Check as HW8's Barrier Method
    % alpha = 1.0;
    % x_new = x_current + alpha*dvStep;
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
    y_current = gradL(x_new, lambda_new) - gradL(x_current, lambda_current); % Gradient difference
    s_current = x_new - x_current; % Step difference

    % BFGS Curvature Check
    if y_current' * s_current > 1e-6
        HessMat = HessMat + (y_current * y_current') / (y_current' * s_current) - (HessMat * (s_current * s_current') * HessMat) / (s_current' * HessMat * s_current);
    else 
        % keep the current Hessian since the curvature is negative
    end 

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

% Plotting the history of design variable values and objective function
figure(Theme="light")
sgtitle('Beam Optimization Convergence History','Interpreter','latex', 'FontWeight', 'bold')
subplot(2,3,1);
plot(history.iter, history.objval, '-o', 'LineWidth',2);
grid minor
xlabel('Iteration','Interpreter','latex', 'FontWeight', 'bold');
ylabel('Objective Value (Beam Mass)','Interpreter','latex', 'FontWeight', 'bold');
title('Convergence History for Objective','Interpreter','latex', 'FontWeight', 'bold');

subplot(2,3,3);
plot(history.iter, history.dv_vals(1,:), '-o', 'LineWidth',2)
hold on; grid minor
plot(history.iter, history.dv_vals(2,:), '-o', 'LineWidth',2)
plot(history.iter, history.dv_vals(3,:), '-o', 'LineWidth',2)
plot(history.iter, history.dv_vals(4,:), '-o', 'LineWidth',2);

yline(0.01, 'r--', 'LineWidth',1.5)
yline(0.25, 'g--', 'LineWidth',1.5)
yline(0.1, 'b--', 'LineWidth',1.5)
yline(1, 'k--', 'LineWidth',1.5)

xlabel('Iteration','Interpreter','latex', 'FontWeight', 'bold');
ylabel('Design Variables','Interpreter','latex', 'FontWeight', 'bold');
ylim([-0.5 1.1])
title('Convergence History of Design Variables','Interpreter','latex', 'FontWeight', 'bold');

subplot(2,3,4)
plot(history.iter, history.FoS, '-o', 'LineWidth',2)
grid minor
xlabel('Iteration','Interpreter','latex', 'FontWeight', 'bold');
ylabel('Factor of Safety','Interpreter','latex', 'FontWeight', 'bold');
yline(1.5, 'c--', 'LineWidth',1.5)
title('Convergence History of Factor of Safety','Interpreter','latex', 'FontWeight', 'bold');

subplot(2,3,6)
plot(history.iter, history.tipDef, '-o', 'LineWidth',2)
grid minor
xlabel('Iteration','Interpreter','latex', 'FontWeight', 'bold');
ylabel('Tip Deflection','Interpreter','latex', 'FontWeight', 'bold');
yline(-0.5, 'm--', 'LineWidth',1.5)
title('Convergence History of Tip Deflection','Interpreter','latex', 'FontWeight', 'bold');

subplot(2,3,3)
legend(["DV1: $b$", "DV2: $h$", "DV3: $t_{1}$", "DV4: $t_{2}$", "$t_1$, $t_{2}$ Lower Bound", "$t_1$, $t_{2}$ Upper Bound", "$b$ & $h$ Lower Bound", "$b$ & $h$ Upper Bound"], "Interpreter", "latex", "Location", "none", "Position", [0.4278 0.5577 0.1488, 0.1659])
hLegend = findobj(gcf,"Type","legend");
hLegend.Interpreter = "latex";
hLegend.FontSize = 16;
hLegend.FontWeight = 'bold';