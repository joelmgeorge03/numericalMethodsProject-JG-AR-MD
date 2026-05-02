%Numerical Methods Project
%fmincon method

%%
% %General Set up
% %Definitions
% 
% %General Variables
% rho = 2700; %density
% L = 10; %total beam length
% EE = 68.9e9; %Young's Modulus of material
% sigma_y = 276e6; %yield strength of the material
% P = -50000; %Applied tip load
% 
% %X=[width; height; thickness top and bottom; thickness side]
% %b= width of the cross section
% %h= height of the cross section
% %t1= thickness of the top and bottom
% %t2= thickness of te side walls
% 
% %Equations
% 
% A= @(b,h,t1,t2) ((h*b)-(h-2*t1)*(b-2*t2)); %Cross Sectional Area
% 
% f = @(b,h,t1,t2) rho.*L.*A;
% 
% delta_L= @(b,h,t1,t2) (P*L^3)/( 3*EE*((1/12)*b*h^3)-(1/12)*(b-2*t2)*(h-2*t1)^3);
% 
% sigma_max= @(b,h,t1,t2) -(P*L*(h/2))/(((1/12)*b*h^3)-(1/12)*(b-2*t2)*(h-2*t1)^3);

%%
%Call fmincon
clear
close all
clc
rho = 2700; %density
L = 10; %total beam length
EE = 68.9e9; %Young's Modulus of material
sigma_y = 276e6; %yield strength of the material
P = -50000; %Applied tip load

nonlcon = @constraints;
A = [];
b = [];
Aeq = [];
beq = [];
lb = [0.1, 0.1, 0.01, 0.01];
ub = [1, 1, 0.25, 0.25];
x0 = [.2, .2, 0.05, 0.05];
options = optimoptions('fmincon','Algorithm','interior-point','SpecifyObjectiveGradient',false, ...
    'SpecifyConstraintGradient',false,'MaxIterations',200,'OptimalityTolerance',1e-6,'Display','iter-detailed');
[x,fval]=fmincon(@(x) beam(x, rho, L), x0, A, b, Aeq, beq, lb, ub, @(x) constraints(x,P,L,EE,sigma_y), options);

% Plot cross-section using optimized x and display mass (fval)
b = x(1);
h = x(2);
t1 = x(3);
t2 = x(4);

figure;
hold on;
axis equal;
xlabel('Width (m)');
ylabel('Height (m)');
title(sprintf('Optimized Cross-Section (mass = %.6g kg)', fval));

% Outer rectangle (centered at origin lower-left at (0,0))
outer_x = [0, b, b, 0, 0];
outer_y = [0, 0, h, h, 0];
plot(outer_x, outer_y, 'k-', 'LineWidth', 2);

% Inner hollow rectangle coordinates (offset by thicknesses)
inner_x = [t2, b-t2, b-t2, t2, t2];
inner_y = [t1, t1, h-t1, h-t1, t1];

% If inner dimensions valid, plot hole
if (b-2*t2 > 0) && (h-2*t1 > 0)
    plot(inner_x, inner_y, 'k-', 'LineWidth', 2);
    % Fill cross-section area between outer and inner
    % Create polygon for outer minus inner by using patch with hole via patch + 'faces' not trivial;
    % Instead fill outer then overlay inner with background color to simulate hollow
    patch(outer_x, outer_y, [0.8 0.9 1], 'EdgeColor','none');
    patch(inner_x, inner_y, 'w', 'EdgeColor','none');
else
    % Solid section (no hole)
    patch(outer_x, outer_y, [0.8 0.9 1], 'EdgeColor','none');
end

% Draw thickness markers
plot([t2 t2], [0 h], 'r--');
plot([b-t2 b-t2], [0 h], 'r--');
plot([0 b], [t1 t1], 'r--');
plot([0 b], [h-t1 h-t1], 'r--');

%legend('Outer','Inner','Location','bestoutside');
xlim([-0.05, b+0.05]);
ylim([-0.05, h+0.05]);
hold off;

%%
%Function Definitions

function F= beam(x, rho, L)

b=x(1);
h=x(2);
t1=x(3);
t2=x(4);

A=(b*h)-(b-2*t2)*(h-2*t1); %Cross Sectional Area

F = rho*L*A;

end

%%
%Constraints
function [c,ceq] = constraints(x,P,L,EE,sigma_y)

b=x(1);
h=x(2);
t1=x(3);
t2=x(4);

delta_L= (P*L^3) / ( 3*EE* ( ( (1/12)*b*h^3 ) -( (1/12)*(b-2*t2)*((h-2*t1)^3) ) ) );

sigma_max= -(P*L*(h/2)) / ( ((1/12)*b*h^3)-((1/12)*(b-2*t2)*(h-2*t1)^3) );

delta_max=0.5;

c=zeros(6,1);

c(1)=sigma_max-(sigma_y/1.5);

c(2)=-delta_L-delta_max;

c(3)=2*t2-b;

c(4)=2*t1-h;

c(5)=-t1;

c(6)=-t2;

ceq=[];

end