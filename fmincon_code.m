%Numerical Methods Project
%fmincon method

fmincon



%%
%Part 2
%Function at the bottom
clc
nonlcon = @unitdisk;
A = [];
b = [];
Aeq = [];
beq = [];
lb = [];
ub = [];
x0 = [0,0];
options = optimoptions('fmincon','Algorithm','interior-point','SpecifyObjectiveGradient',false, ...
    'SpecifyConstraintGradient',false,'MaxIterations',200,'OptimalityTolerance',1e-6,'Display','iter-detailed');
[x,fval] = fmincon(F,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)



%%


function [c,ceq] = unitdisk(x)
c = x(1)^2 + x(2)^2 - 1;
ceq = [];
end