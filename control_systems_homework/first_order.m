%% 一阶系统的时域响应分析 —— 使用传递函数建立模型
%  作业：参考视频，修改参数，设计自己的一阶系统并观察响应曲线
%  参考视频原参数： G(s) = 5/(s+5)
%  我的设计参数  ： G(s) = 4/(s+2)  ( 直流增益 K = 2, 时间常数 tau = 0.5 s )
clc; clear; close all;

%% 加载 Control Package（若使用 MATLAB，请注释掉下面一行）
pkg load control

%% 定义一阶系统  G(s) = K/(tau*s + 1) = 4/(s+2)
K   = 2;        % 直流增益（决定阶跃稳态值）
tau = 0.5;      % 时间常数（决定响应快慢），极点 s = -1/tau = -2
G_a = tf([K/tau], [1, 1/tau]);   % => 4/(s+2)

%% 仿真：三种典型响应
figure;

% ① 单位冲激响应
subplot(3,1,1)
impulse(G_a);
title('单位冲激响应');

% ② 单位阶跃响应
subplot(3,1,2)
step(G_a);
title('单位阶跃响应');

% ③ 对初始状态的响应（状态空间，零输入）
subplot(3,1,3)
A = -2;  B = 4;  C = 1;  D = 0;   % dx/dt = -2x + 4u, y = x
sys = ss(A, B, C, D);
x0 = 5;                            % 初始状态
initial(sys, x0);
title('初始状态响应 (x0 = 5)');
