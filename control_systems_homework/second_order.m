%% 二阶系统的时域响应分析 —— 使用传递函数建立模型
%  作业：参考视频，修改参数，设计自己的二阶系统并观察响应曲线
%  参考视频原参数： zeta = 0.5, w_n = 10
%  我的设计参数  ： zeta = 0.3, w_n = 5   =>  G(s) = 25/(s^2 + 3s + 25)
clc; clear; close all;

%% 加载 Control Package（若使用 MATLAB，请注释掉下面一行）
pkg load control

%% 定义二阶系统  G(s) = w_n^2 / (s^2 + 2*zeta*w_n*s + w_n^2)
zeta = 0.3;     % 阻尼比（决定超调与振荡）—— 欠阻尼
w_n  = 5;       % 自然频率（决定响应快慢与振荡频率）
G_s  = tf([w_n^2], [1, 2*w_n*zeta, w_n^2]);   % => 25/(s^2+3s+25)

%% 仿真：三种典型响应
figure;

% ① 单位冲激响应
subplot(3,1,1)
impulse(G_s);
title('单位冲激响应');

% ② 单位阶跃响应
subplot(3,1,2)
step(G_s);
title('单位阶跃响应');

% ③ 对初始状态的响应（状态空间，零输入）
subplot(3,1,3)
A = [0 1; -w_n^2 -2*zeta*w_n];   % 可控标准型
B = [0; w_n^2];
C = [1 0];
D = 0;
sys = ss(A, B, C, D);
z0 = [1; 0];                      % 初始位置 1，初始速度 0
initial(sys, z0);
title('初始状态响应 (z0 = [1;0])');

%% 拓展：阻尼比 zeta 对阶跃响应的影响（固定 w_n = 5）
figure; hold on; grid on;
for zeta = [0.1, 0.3, 0.707, 1.0, 2.0]
    G = tf([w_n^2], [1, 2*w_n*zeta, w_n^2]);
    step(G);
end
legend('zeta=0.1','zeta=0.3','zeta=0.707','zeta=1.0','zeta=2.0');
title('阻尼比 zeta 对二阶系统阶跃响应的影响');
