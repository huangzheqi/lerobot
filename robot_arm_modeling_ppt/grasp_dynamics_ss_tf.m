%% =======================================================================
%  grasp_dynamics_ss_tf.m
%  机械臂抓取任务：动力学建模 -> 状态空间模型 -> 传递函数
%  -----------------------------------------------------------------------
%  模型：平面二连杆机械臂（关节力矩驱动），末端抓取质量为 mp 的负载
%        （视为与连杆 2 固连的点质量）。
%  流程：
%    1) 拉格朗日方法得到动力学方程
%         M(q)*q'' + C(q,q')*q' + G(q) + Fv*q' = tau
%    2) 在悬垂抓取平衡位形 q0 = [-pi/2; 0]（此时 G(q0)=0, tau0=0）
%       做小偏差线性化：  M0*dq'' + Fv*dq' + Kg*dq = dtau
%    3) 取状态 x = [dq; dq']，得状态空间矩阵 (A,B,C,D)
%    4) 由 (A,B,C,D) 求传递函数矩阵 G(s)，并做极点 / 阶跃 / 频域分析
%  依赖：Control System Toolbox（ss, tf, ss2tf, pole, dcgain, step, bode）
%% =======================================================================
clear; clc; close all;

%% 1. 物理参数（含抓取负载）
m1  = 2.0;   m2  = 1.0;   mp = 0.5;     % 连杆 1/2 质量, 抓取负载质量 [kg]
l1  = 0.5;   l2  = 0.4;                 % 连杆长度 [m]
lc1 = 0.25;  lc2 = 0.2;                 % 连杆质心到关节的距离 [m]
I1  = 0.05;  I2  = 0.02;                % 连杆绕质心的转动惯量 [kg*m^2]
fv  = [0.5; 0.3];                       % 关节粘性摩擦系数 [N*m*s/rad]
g   = 9.81;                             % 重力加速度 [m/s^2]

%% 2. 平衡点：悬垂抓取位形 q0 = [-pi/2; 0], dq0 = 0
%  完整动力学方程 M(q)q''+C(q,q')q'+G(q)+Fv q' = tau 中：
%    M11 = m1*lc1^2+I1+I2+m2*(l1^2+lc2^2+2*l1*lc2*cos q2)
%          + mp*(l1^2+l2^2+2*l1*l2*cos q2)
%    M12 = I2+m2*(lc2^2+l1*lc2*cos q2)+mp*(l2^2+l1*l2*cos q2),  M21 = M12
%    M22 = I2+m2*lc2^2+mp*l2^2
%    G1  = h1*cos q1 + h2*cos(q1+q2),  G2 = h2*cos(q1+q2)
%    h1  = (m1*lc1+(m2+mp)*l1)*g,      h2 = (m2*lc2+mp*l2)*g
q0 = [-pi/2; 0];
c2 = cos(q0(2)); s1 = sin(q0(1)); s12 = sin(q0(1)+q0(2));

% --- 平衡点处的惯性矩阵 M0 = M(q0)（含负载 mp 的贡献）---
M11 = m1*lc1^2 + I1 + I2 + m2*(l1^2+lc2^2+2*l1*lc2*c2) ...
      + mp*(l1^2+l2^2+2*l1*l2*c2);
M12 = I2 + m2*(lc2^2+l1*lc2*c2) + mp*(l2^2+l1*l2*c2);
M22 = I2 + m2*lc2^2 + mp*l2^2;
M0  = [M11, M12; M12, M22];

% --- 重力矩与平衡输入：tau0 = G(q0)（悬垂位形下为 0）---
h1   = (m1*lc1 + (m2+mp)*l1)*g;
h2   = (m2*lc2 + mp*l2)*g;
tau0 = [h1*cos(q0(1)) + h2*cos(q0(1)+q0(2)); h2*cos(q0(1)+q0(2))];

% --- 重力刚度矩阵 Kg = dG/dq|q0（线性化的核心）---
Kg = [-h1*s1 - h2*s12, -h2*s12;
      -h2*s12,         -h2*s12];

Fv = diag(fv);                          % 阻尼矩阵

fprintf('M0 = M(q0) =\n'); disp(M0);
fprintf('tau0 = G(q0) =\n'); disp(tau0.');
fprintf('Kg = dG/dq|q0 =\n'); disp(Kg);

%% 3. 状态空间模型
%  线性化方程  M0*dq'' + Fv*dq' + Kg*dq = dtau
%  取 x = [dq; dq'] (4x1), u = dtau (2x1), y = dq (2x1)：
%     x' = A x + B u,   y = C x + D u
n = 2;
A = [zeros(n), eye(n); -M0\Kg, -M0\Fv];
B = [zeros(n); inv(M0)];
C = [eye(n), zeros(n)];
D = zeros(n);
sys = ss(A, B, C, D);

fprintf('A =\n'); disp(A);
fprintf('B =\n'); disp(B);

%% 4. 传递函数  G(s) = C*(s*I - A)^(-1)*B + D
G   = tf(sys);                          % 2x2 传递函数矩阵
G11 = minreal(G(1,1));                  % 通道：dtau1 -> dq1
disp('G11(s) ='); G11

[num, den] = ss2tf(A, B, C, D, 1);      % 等价做法（第 1 个输入）
fprintf('den(s)  : '); disp(den);
fprintf('num11(s): '); disp(num(1,:));

%% 5. 分析与验证
p = pole(sys);                          % 极点（= eig(A)）
fprintf('poles =\n'); disp(p);
assert(all(real(p) < 0), '存在右半平面极点！');
fprintf('全部极点位于左半平面 -> 平衡位形附近渐近稳定\n');

k0 = dcgain(G11);                       % 直流增益
fprintf('dcgain(G11) = %.4f, inv(Kg)(1,1) = %.4f（应一致）\n', ...
        k0, [1 0]*(Kg\[1;0]));

figure('Name','step');  step(G11, 5); grid on;
title('阶跃响应：\delta\tau_1 \rightarrow \deltaq_1');
figure('Name','bode');  bode(G11);   grid on;
