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
h9 = @(x) (P_applied*L_beam^3)/( 3*E_beam*( (1/12)*x(1)*x(2)^3 - (1/12)*(x(1) - 2*x(4))*(x(2) - 2*x(3))^3 ) ) + 0.5;
h10 = @(x) -(P_applied*L_beam*x(2)/2)/(( (1/12)*x(1)*x(2)^3 - (1/12)*(x(1) - 2*x(4))*(x(2) - 2*x(3))^3 ) ) - 1.5*sigma_yield;

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
x0 = [0.5; 0.5; 0.1; 0.1]; % initial guess for sizing
lambda0 = zeros(10,1); % initial guess of Lagrange Mulitipliers
HessMat0 = eye(length(x0)); % initial guess for Hessian Matrix
solTol = 1e-6; % solution tolerance
maxIter = 10; % maximum iteration count

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

for k = 1:maxIter % begin loop

    % Evaluate objective gradient, constraints, and constraint Jacobian at current point
    gradf_current = gradf(x_current);
    h_current = h(x_current);
    J_h_current = J_h(x_current);
    
    % The active set handling is currently not correct. No considerations
    % are being made for the other inactive ineqaulity constraints. 

    % This code will implement a from-scratch active set method


    % Find the active constraints
    actTol = 1e-2; % tolerance for active constraints
    h_active_idx = find( h_current >= -actTol ); % find the indices of the active constraints that are close to 0 if not exact
    h_active = h_current(h_active_idx); % create the vector of active constraints
    gradh_active = J_h_current(h_active_idx, :); % create the Jacobian of the active constraints
    
    % Create 0 matrix for Linear System
    q = length(h_active_idx); % length of active constraint vector
    zeroMat = zeros(q, q);

    % Form the linear system of KKT conditions
    coeffMat = [ HessMat         gradh_active'
                 gradh_active         zeroMat];
    rhs = [-gradf_current; -h_active];
   
    % Solve the linear system
    stepVec = coeffMat \ rhs; % this produces the step direction

    % Extract step direction values
    dvStep = stepVec(1:4); % step direction for design variables
    lambda_step = stepVec(5:end); % new Lagrange multiplier guess
    lambda_new = zeros(10,1);
    lambda_new(h_active_idx) = lambda_step;

    % Advice from Dr. Wang - use a small step size along the calculated step direction
    alpha = 0.01; % tune this to provide a suitable solution - should really be a line search
    x_new = x_current + alpha*dvStep;
     
    % Update the Hessian using BFGS (Based on Algorithm 6.5 from Heath)
    y_current = gradL(x_new, lambda_new) - gradL(x_current, lambda_current); % Gradient difference
    s_current = x_new - x_current; % Step difference
    HessMat = HessMat + (y_current * y_current') / (y_current' * s_current) - (HessMat * (s_current * s_current') * HessMat) / (s_current' * HessMat * s_current);
    
    % Check for convergence
    if norm(dvStep) < solTol
        break; 
    end
    
    % Update design variable guess and Lagrange multiplier guess
    x_current = x_new; % update the design variables
    lambda_current = lambda_new; % update Lagrange Multipliers
    objval = f(x_current); % compute the mass of the beam

    % Store values and active constraints
    history.dv_vals(:, k) = x_current; % Store the current design variable guess
    history.lambda{k} = lambda_current; % Store the current Lagrange multipliers
    history.objval(:,k) = objval; % Store the current beam mass
    history.activeCons{k} = h_active_idx; % store the current active constraints
    history.iter(k) = k; % Store the current iteration number

end
