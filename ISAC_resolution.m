
clear; clc; close all;

%% ========== 数据集构造参数 ==========
scale_factor_W = 1e4;
scale_factor_CRB = 100;
num_samples = 57;

% 输出数据保留小数位数
save_digits = 3;

% 求解失败的次数
solve_fail = 0;

% 发射天线数 N_t 和接收天线数 N_r
Nt = 12;
Nr = 10;
% 用户个数K（K < N_t < N_r）
K = 7;

%% ========== 读取旧数据 ==========
filename = sprintf('ISAC_Dataset_Wk_%d_RL.mat', K);
if exist(filename, 'file')
    fprintf('检测到已存在文件: %s\n', filename);
    fprintf('正在读取旧数据...\n');
    old_data = load(filename);
    
    % 取出旧数据矩阵
    old_features = old_data.final_features;
    old_labels = old_data.final_labels;
    
    fprintf('旧文件包含样本数: %d\n', size(old_features, 1));
else
    fprintf('未找到文件 %s，将创建新文件。\n', filename);
    old_features = [];
    old_labels = [];
end

%% ========== 数据预分配 ==========

% 输入特征: [theta(1), PT_dBm(1), Gamma(1), H_real, H_imag]
feat_dim = 1 + 1 + 1 + Nt*K + Nt*K;

% 注意到用户的波束成型矩阵W是Hermitian复矩阵
% 虽然W的维度是Nt×Nt，但是只需要存对角线和上班部分的矩阵元素，并且对角线的虚部为0
% W的上半一共有Nt(Nt-1)/2个数，也就是有Nt(Nt-1)/2个实部和虚部
m = Nt * (Nt - 1) / 2;

% 每个用户的实数段长度，对角线实数加上半的实数：diag_real + upper_real
real_len_per_user = Nt + m;
% 每个用户的虚数段长度，对角线为0，不需要存储：upper_imag
imag_len_per_user = m;

% 每个用户的总输出长度，Nt(Nt-1)/2 * 2 + Nt = Nt^2
compact_len_per_user = real_len_per_user + imag_len_per_user;
% 全体用户的 W 标签总长度：Nt^2 * K
W_dim = compact_len_per_user * K;

% 加上CRB作为辅助任务预测
label_dim = W_dim + 1;

data_features = zeros(num_samples, feat_dim);
data_labels = zeros(num_samples, label_dim);
valid_indices = false(num_samples, 1);  % 记录有效样本的逻辑索引

%% ========== 系统模型参数 ==========

% DFRC 帧长L
L = 10;

sigma2C_dBm = 0;     % 通信噪声方差
sigma2R_dBm = 0;     % 雷达噪声方差

% 将dBm转换到瓦
sigma2C = 10^((sigma2C_dBm - 30)/10);  % 通信噪声方差（W）
sigma2R = 10^((sigma2R_dBm - 30)/10);  % 雷达噪声方差（W）

Gamma_dB = 10;
Gamma_lin = 10^(Gamma_dB/10);      % 转成线性
Gamma_vec = Gamma_lin * ones(K,1); % Γ_k，所有用户相同

% 阵列参数：半波长间距的均匀线阵 ULA
lambda = 1;     % 归一化波长，可以取 1（只影响相位比例）
d = lambda/2;   % 阵元间隔：半波长
k0 = 2*pi/lambda; % 波数 k0 = 2π/λ

% 生成发射和接收阵列的方向矢量 a(θ) 和 b(θ)
n_t = (-((Nt-1)/2):((Nt-1)/2)).';   % 发射阵列的索引列向量，中心对称
n_r = (-((Nr-1)/2):((Nr-1)/2)).';   % 接收阵列的索引列向量，中心对称

% 反射系数α，设为1（无衰落，仅相当于单位反射）
alpha = 0.5;

%% ========== 数据集生成过程 ==========
for loop_idx = 1:num_samples
    % A. 随机化参数功率
    PT_dBm_min = 12;
    PT_dBm_max = 13;
    PT_dBm = PT_dBm_min + (PT_dBm_max - PT_dBm_min) * rand();  % 总发射功率
    PT = 10^((PT_dBm - 30)/10);  % 将dBm转换到瓦

    % B. 随机化信道噪声
    % H ∈ C^{K×N_t}，每一行是一个用户到基站发射阵列的信道，假设h_k独立，用Rayleigh信道模拟生成
    H = (randn(K, Nt) + 1j * randn(K, Nt)) / sqrt(2);
    % 向信道矩阵 H 添加 AWGN 噪声
    Z_C = sqrt(sigma2C/2) * (randn(K, Nt) + 1j * randn(K, Nt));  % AWGN噪声
    % 加噪声后的信道
    H = H + Z_C;

    % C. 随机化角度
    theta_deg_min = 20;
    theta_deg_max = 30;
    theta_deg = theta_deg_min + (theta_deg_max - theta_deg_min) * rand();
    theta = deg2rad(theta_deg);  % 转成弧度，后面用 sin(θ), cos(θ)
    % 发射阵列方向矢量 a(θ) ∈ C^{N_t×1}
    a_theta = exp(1j * k0 * d * n_t * sin(theta));
    % 接收阵列方向矢量 b(θ) ∈ C^{N_r×1}
    b_theta = exp(1j * k0 * d * n_r * sin(theta));
    % 目标响应矩阵 G = α * b(θ) * a^H(θ) ∈ C^{N_r×N_t}
    G = alpha * (b_theta * a_theta');
    A = b_theta * a_theta';         % A(θ) ≜ b(θ)a^H(θ)
    % A 点对 θ 的导数： Ȧ(θ) = ḃ(θ)a^H(θ) + b(θ)ȧ^H(θ)
    a_dot = 1j * k0 * d * n_t * cos(theta) .* a_theta;
    b_dot = 1j * k0 * d * n_r * cos(theta) .* b_theta;
    Ad = b_dot * a_theta' + b_theta * a_dot';  % Ȧ(θ) 

    % 构造 Q_k = h_k h_k^H
    Q = zeros(Nt, Nt, K);
    for k = 1:K
        hk = H(k,:).';
        Q(:,:,k) = hk * hk';
    end

    % D. 使用CVX求解SDP问题
    % 变量： W_k (Nt×Nt，Hermitian PSD), t（标量）
    % 目标： maximize t  等价于 minimize -t
    % 约束：
    %   1) Schur 补 LMI（对应 CRB 最大化）
    %   2) 每个用户的 SINR 约束
    %   3) 总功率约束 sum_k tr(W_k) ≤ P_T
    %   4) W_k ⪰ 0

    cvx_begin quiet
        cvx_precision best         % 提高精度
    
        % W 是一个[Nt×Nt×K]的Hermitian PSD三维数组
        variable W(Nt, Nt, K) hermitian semidefinite
        variable t                 % 辅助变量t
        
        % === (1) 构造 R_X = sum_k W_k ===
        expression RX(Nt, Nt)
        RX = zeros(Nt, Nt);
        for k = 1:K
            RX = RX + W(:,:,k);
        end
        
        % === (2) Schur 补 LMI，对应式矩阵半正定约束 ===
        % 为了保证是 Hermitian，我们手动构造：
        a11 = real(trace(Ad' * Ad * RX));      % tr(Ȧ^H Ȧ R_X)
        a12 = trace(Ad' * A  * RX);           % tr(Ȧ^H A R_X)
        a22 = real(trace(A'  * A  * RX));     % tr(A^H A R_X)
        
        LMI = [a11 - t,   a12;
               conj(a12), a22];
        
        LMI == hermitian_semidefinite(2);     % 表示 LMI ⪰ 0
        
        % === (3) SINR 约束：对每个用户 k 施加式 (31) ===
        for k = 1:K
            Gammak = Gamma_vec(k);           % 第 k 个用户的 Γ_k
            Qk = Q(:,:,k);                   % 对应的 Q_k
            
            % 信号项：tr(Q_k W_k)
            signal_k = trace(Qk * W(:,:,k));
            
            % 干扰项：Γ_k * sum_{i≠k} tr(Q_k W_i)
            interference_k = 0;
            for i = 1:K
                if i ~= k
                    interference_k = interference_k + trace(Qk * W(:,:,i));
                end
            end
            interference_k = Gammak * interference_k;
            
            % 噪声项：Γ_k σ_C^2
            noise_k = Gammak * sigma2C;
            
            % SINR 约束：tr(Q_k W_k) - Γ_k * sum_{i≠k} tr(Q_k W_i) ≥ Γ_k σ_C^2
            real(signal_k - interference_k) >= noise_k;
        end
        
        % === (4) 总功率约束 sum_k tr(W_k) ≤ P_T ===
        total_power = 0;
        for k = 1:K
            total_power = total_power + trace(W(:,:,k));
        end
        real(total_power) <= PT;
        
        % === (5) 目标函数：minimize -t  等价于 maximize t ===
        minimize(-t)
    cvx_end
    
    % E. 数据检查和收集
    if strcmp(cvx_status, 'Solved')
        valid_indices(loop_idx) = true; % 标记该行有效

        % 提取最优解，因为CVX很多矩阵是以稀疏的状态存储的，这里是补全矩阵的数据
        W_opt = full(W);    % Nt x Nt x K
        RX_opt = full(RX);  % Nt x Nt

        % 按照原始CRB公式计算CRB(θ)
        num_CRB = sigma2R * trace(A' * A * RX_opt);
        term1 = trace(Ad' * Ad * RX_opt);
        term2 = trace(A'  * A  * RX_opt);
        term3 = trace(Ad' * A  * RX_opt);
        
        den_CRB = 2 * abs(alpha)^2 * L * ( term1 * term2 - abs(term3)^2 );
        CRB_rad = real(num_CRB / den_CRB);
        
        CRB_deg  = CRB_rad * (180/pi)^2;  % 换成度^2
        CRB_deg2   = sqrt(CRB_deg);  % 角度标准差（单位：度）

        % -------- 构造输入特征：把信道增益H按用户分块 --------
        % H_feat 的顺序是：
        %   user_1: [Re(h1_1..h1_Nt), Im(h1_1..h1_Nt)]
        %   user_2: [Re(...), Im(...)]

        H_feat = zeros(1, 2 * Nt * K);
        for k = 1:K
            base = (k-1) * 2 * Nt;
            H_feat(base + (1:Nt)) = real(H(k, :));
            H_feat(base + (Nt+1 : 2*Nt)) = imag(H(k, :));
        end
       
        % [theta, PT_dBm, Gamma_lin, H_real_flat, H_imag_flat]
        feature_vector = [theta, PT_dBm, Gamma_lin, H_feat];
        
        % -------- 构造标签：W的Hermitian压缩编码（按用户分块）--------
        W_scaled = W_opt * scale_factor_W;
        CRB_scaled = CRB_deg2 * scale_factor_CRB;

        % 每用户一个块：[diag_real, upper_real, upper_imag]，长度 Nt^2
        W_compact_all = zeros(1, compact_len_per_user * K);

        for k = 1:K
            % 取出每个用户k的矩阵
            Wk = W_scaled(:,:,k);

            % 数值稳定，强制 Hermitian（CVX 数值误差可能导致不完全共轭）
            Wk = (Wk + Wk') / 2;

            % 压缩编码
            [real_part, imag_part] = encode_hermitian_compact(Wk);
            user_block = [real_part, imag_part];   % 先实后虚
            
            % (1:compact_len_per_user)生成从1到compact_len_per_user的连续序列索引
            idx = (k-1) * compact_len_per_user + (1:compact_len_per_user);
            W_compact_all(idx) = user_block;
        end

        label_vector = [W_compact_all, CRB_scaled];

        % -------- 在写入数据集之前统一保留小数位 --------
        feature_vector = round(feature_vector, save_digits);
        label_vector   = round(label_vector, save_digits);

        % 存入数据集
        data_features(loop_idx, :) = feature_vector;
        data_labels(loop_idx, :) = label_vector;

        if mod(loop_idx, 50) == 0
            fprintf('Loop %d: OK. CRB=%.4f\n', loop_idx, CRB_deg2);
        end

%         fprintf('\n===== 优化完成 =====\n');
%         fprintf('CVX 状态：%s\n', cvx_status);
%         fprintf('最优 t = %.4e\n', t);
%         fprintf('Root-CRB (deg) = %.3e\n', CRB_deg2);
%         fprintf('功率约束 = %.3e W\n', PT);

    else
        solve_fail = solve_fail + 1;
        valid_indices(loop_idx) = false;
        if mod(loop_idx, 10) == 0
            fprintf('Loop %d/%d: Failed (%s)\n', loop_idx, num_samples, cvx_status);
        end
    end
end

%% ========== 保存为数据集为 .mat 文件 ==========
% 提取有效数据
current_valid_features = data_features(valid_indices, :);
current_valid_labels = data_labels(valid_indices, :);

fprintf('\n数据集生成完毕。\n有效样本数: %d\n失败样本数: %d\n', ...
    size(current_valid_features, 1), solve_fail);

final_features = [old_features; current_valid_features];
final_labels   = [old_labels; current_valid_labels];
fprintf('合并后总样本数: %d\n', size(final_features, 1));

% 保存所有必要信息
save(filename, ...
    'final_features', 'final_labels', 'save_digits', ...
    'scale_factor_W', 'scale_factor_CRB', ...
    'Nt', 'Nr', 'K', 'PT_dBm_min', 'PT_dBm_max', ...
    'theta_deg_min', 'theta_deg_max', ...
    'real_len_per_user', 'imag_len_per_user', 'compact_len_per_user'); 

fprintf('数据已保存为 %s\n', filename);

TEMP = load(filename) %加载数据集


%% ============================================================
% 局部函数：Hermitian 压缩编码
function [real_part, imag_part] = encode_hermitian_compact(Wk)
    % 输入：
    %   Wk: Nt x Nt 复数 Hermitian 矩阵（建议外部已做 (Wk+Wk')/2）
    %
    % 输出（用于每个用户的压缩块）：
    %   real_part = [ real(diag(Wk)), real(upper_triangle_no_diag(Wk)) ]
    %   imag_part = [ imag(upper_triangle_no_diag(Wk)) ]
    %
    % 关键点：
    %   - Hermitian 矩阵对角线虚部应为 0，所以不存对角虚部
    %   - 下三角由上三角共轭决定，所以不存下三角

    Nt = size(Wk, 1);

    % 1) 对角线实部（1 x Nt）
    diag_r = real(diag(Wk)).';

    % 2) 上三角（不含对角）取出顺序说明：
    % mask = triu(true(Nt), 1) 选中 i<j 的元素
    % Wk(mask) 会按 MATLAB 的列优先顺序线性取出（列1 -> 列2 -> ...）
    mask = triu(true(Nt), 1);
    upper = Wk(mask);              % m x 1

    upper_re = real(upper).';      % 1 x m
    upper_im = imag(upper).';      % 1 x m

    % 3) 拼接输出
    real_part = [diag_r, upper_re];
    imag_part = upper_im;
end
