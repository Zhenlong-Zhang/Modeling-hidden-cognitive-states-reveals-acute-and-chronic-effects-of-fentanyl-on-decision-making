% Author: Zhenlong Zhang (2025)
%
% Purpose:
%   Fit a Mixture-of-Agents HMM (MoA-HMM) to two-step task data.
% Data layout (CSV columns):
%   1) choice        : 1 or 2
%   2) transition    : 0=rare, 1=common
%   3) reward        : 0 or 1
%   4) new_sess      : 0=continue, 1=new session
%
% Model:
%   Agents = {MFc, MFr, MBc, MB, Bias} with 4 learning rates [MFc,MFr,MBc,MB].
%   Per-state beta weights (nAgents x nStates).
%   π (initial state), A (transition), and betas parameterized via softmax.
%   Emissions: logistic choice model from agent features x(t) and betas.
%
% Optimization:
%   fmincon (SQP), MaxIterations=200; objective = NLL + L2 (theta=0.01).
%   Learning rates constrained to [0.05, 0.95]; others unconstrained (softmax).
%
% Output (allresults.mat):
%   allResults.(group).(ratID).nStatesK with fields:
%     .nLL      : -(objective at optimum)  % equals -(NLL + regularization)
%     .logLik   : log-likelihood (without regularization)
%     .freeLR   : [4x1] learning rates
%     .freeBeta : [nAgents x nStates] state-wise betas
%     .A        : [nStates x nStates] transition matrix
%     .pi       : [1 x nStates] initial state distribution
%     .gamma    : [T x nStates] posterior state probabilities

clear; clc; close all;

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('local', 5);
elseif poolobj.NumWorkers ~= 5
    delete(poolobj);
    parpool('local', 5);
end

mainDir = 'data-cleaned';
subDirs = dir(mainDir);
subDirs = subDirs([subDirs.isdir]);
subDirs = subDirs(~ismember({subDirs.name}, {'.','..'}));


taskList = [];
taskIndex = 1;
for g = 1:length(subDirs)
    groupName = subDirs(g).name;
    groupPath = fullfile(mainDir, groupName);
    csvFiles = dir(fullfile(groupPath, '*.csv'));
    for r = 1:numel(csvFiles)
        ratPath = fullfile(groupPath, csvFiles(r).name);
        [~, ratID, ~] = fileparts(csvFiles(r).name);
        for nStates = 1:7
            taskList(taskIndex).group = groupName;
            taskList(taskIndex).ratID = ratID;
            taskList(taskIndex).ratPath = ratPath;
            taskList(taskIndex).nStates = nStates;
            taskIndex = taskIndex + 1;
        end
    end
end
nTasks = numel(taskList);
fprintf('Total tasks to fit: %d\n', nTasks);



resultList = cell(nTasks, 1);

parfor i = 1:nTasks
    T = taskList(i);
    dataTable = readtable(T.ratPath);
    choices = dataTable{:,1};
    trans_common = (dataTable{:,2} == 1);
    rewards = dataTable{:,3};
    new_sess = (dataTable{:,4} == 1);
    forced = false(height(dataTable),1);
    p_congruent = 0.8;
    modelResult = runMoAHMM(choices, rewards, new_sess, forced, trans_common, p_congruent, T.nStates);
    resultList{i} = struct(...
        'group', T.group, ...
        'ratID', T.ratID, ...
        'nStates', T.nStates, ...
        'modelResult', modelResult);
end


allResults = struct();
for i = 1:nTasks
    res = resultList{i};
    grp = res.group;
    rid = res.ratID;
    ns = res.nStates;
    if ~isfield(allResults, grp)
        allResults.(grp) = struct();
    end
    if ~isfield(allResults.(grp), rid)
        allResults.(grp).(rid) = struct();
    end
    allResults.(grp).(rid).(['nStates' num2str(ns)]) = res.modelResult;
end

save('allresults.mat','allResults');
fprintf('All results saved to allresults.mat\n');


function modelResult = runMoAHMM(choices, rewards, new_sess, forced, trans_common, p_congruent, nStates)
    
    nRestarts = 50;

    agents = {'betaMFc','betaMFr','betaMBc','betaMB','betaBias'};
    nAgents_total = numel(agents);

    pack_hmm = @(pi_vec, A_mat, beta_mat, lr_vec)[pi_vec(:); A_mat(:); beta_mat(:); lr_vec(:)];
    unpack_hmm = @(x) deal( ...
        x(1:nStates), ...
        reshape(x(nStates+1:nStates+nStates^2), [nStates, nStates]), ...
        reshape(x(nStates+nStates^2+1:nStates+nStates^2+nAgents_total*nStates), [nAgents_total, nStates]), ...
        x(nStates+nStates^2+nAgents_total*nStates+1:end) );


    obj_fun_hmm = @(p) hmm_nll(p, agents, choices, rewards, new_sess, forced, trans_common, 0.5, p_congruent, nStates, unpack_hmm, true, 0.01);

    options = optimoptions('fmincon', ...
        'Display','off', 'Algorithm','sqp', ...
        'MaxIterations',200, 'TolFun',1e-5, 'TolX',1e-5);

    best_fval = inf;
    best_p = [];
    best_unpack = struct();


    for r = 1:nRestarts
       
        freeLR_init = 0.05 + (0.95-0.05)*rand(4,1);            
        freePi_init = rand(nStates,1);
        freeA_init  = rand(nStates,nStates);
        freeBeta_init = rand(nAgents_total, nStates);

        x0_hmm = pack_hmm(freePi_init, freeA_init, freeBeta_init, freeLR_init);

        
        lb = -inf(size(x0_hmm));
        ub =  inf(size(x0_hmm));
        lb(end-3:end) = 0.05;
        ub(end-3:end) = 0.95;

        
        [p_opt, fval_hmm] = fmincon(obj_fun_hmm, x0_hmm, [], [], [], [], lb, ub, [], options);

        
        if fval_hmm < best_fval
            best_fval = fval_hmm;
            best_p = p_opt;
            [freePi_fit, freeA_fit, freeBeta_fit, freeLR_fit] = unpack_hmm(p_opt);
            best_unpack.freePi_fit  = freePi_fit;
            best_unpack.freeA_fit   = freeA_fit;
            best_unpack.freeBeta_fit= freeBeta_fit;
            best_unpack.freeLR_fit  = freeLR_fit;
        end
    end

    
    [freePi_fit, freeA_fit, freeBeta_fit, freeLR_fit] = deal( ...
        best_unpack.freePi_fit, best_unpack.freeA_fit, best_unpack.freeBeta_fit, best_unpack.freeLR_fit);

   
    pi_fit = softmax_vec(freePi_fit)';                   
    A_fit  = zeros(nStates, nStates);
    for i = 1:nStates
        A_fit(i,:) = softmax_vec(freeA_fit(i,:)')';
    end

    x = updateAgents(agents, freeLR_fit, ...
        'p_cong', p_congruent, ...
        'choices', choices, 'nonchoices', [], 'outcomes', [], ...
        'rewards', rewards, 'new_sess', new_sess, 'forced', forced, ...
        'initQ', 0.5, 'trans_common', trans_common);

    B = B_from_x_emission(x, freeBeta_fit, choices);
    [alpha_fb, beta_fb, c_vec] = forward_backward(B, pi_fit, A_fit, new_sess);
    logLik = sum(log(c_vec+eps));
    gamma = (alpha_fb .* beta_fb) ./ (sum(alpha_fb .* beta_fb,2)+eps);

  
    modelResult.nLL      = -best_fval;   
    modelResult.logLik   = logLik;       
    modelResult.freeLR   = freeLR_fit;
    modelResult.freeBeta = freeBeta_fit;
    modelResult.A        = A_fit;
    modelResult.pi       = pi_fit;
    modelResult.gamma    = gamma;
end


function nLL = hmm_nll(free_x, agents, choices, rewards, new_sess, forced, trans_common, initQ, p_cong, nStates, unpack_hmm, l2_penalty, theta)
    [freePi, freeA, freeBeta, freeLR] = unpack_hmm(free_x);
    pi_vec = softmax_vec(freePi)';
    A_mat = zeros(nStates, nStates);
    for i = 1:nStates
        A_mat(i,:) = softmax_vec(freeA(i,:)')';
    end
    x = updateAgents(agents, freeLR, 'p_cong', p_cong, 'choices', choices, 'nonchoices', [], 'outcomes', [], 'rewards', rewards, 'new_sess', new_sess, 'forced', forced, 'initQ', initQ, 'trans_common', trans_common);
    B = B_from_x_emission(x, freeBeta, choices);
    [~, ~, c_vec] = forward_backward(B, pi_vec, A_mat, new_sess);
    logLik = sum(log(c_vec+eps));
    nLL = -logLik;
    if l2_penalty
        reg = theta * sum(free_x.^2);
        nLL = nLL + reg;
    end
end

function B = B_from_x_emission(x, beta_mat, actions)
    nAgents_total = size(beta_mat,1);
    [~, T] = size(x);
    nStates = size(beta_mat,2);
    B = nan(T, nStates);
    for t = 1:T
        for s = 1:nStates
            net_val = 0;
            for k = 1:nAgents_total
                net_val = net_val + beta_mat(k,s) * x(k,t);
            end
            p_left = 1/(1+exp(-net_val));
            if actions(t)==1
                B(t,s) = p_left;
            else
                B(t,s) = 1 - p_left;
            end
        end
    end
end

function [alpha, beta, c] = forward_backward(B, pi_vec, A_mat, new_sess)
    T = size(B,1);
    nStates = size(B,2);
    alpha = zeros(T, nStates);
    beta = zeros(T, nStates);
    c = zeros(T,1);
    alpha(1,:) = pi_vec .* B(1,:);
    c(1) = sum(alpha(1,:));
    if c(1)==0, c(1)=eps; end
    alpha(1,:) = alpha(1,:) / c(1);
    for t = 2:T
        alpha(t,:) = (alpha(t-1,:) * A_mat) .* B(t,:);
        c(t) = sum(alpha(t,:));
        if c(t)==0, c(t)=eps; end
        alpha(t,:) = alpha(t,:) / c(t);
    end
    beta(T,:) = ones(1, nStates) / c(T);
    for t = T-1:-1:1
        beta(t,:) = (beta(t+1,:) .* B(t+1,:)) * A_mat';
        s = sum(beta(t,:));
        if s==0, s=eps; end
        beta(t,:) = beta(t,:) / s;
    end
end

function y = softmax_vec(x)
    expx = exp(x - max(x));
    y = expx / sum(expx);
end

function [x, dx] = updateAgents(agents, freeLR, varargin)
    ip = inputParser;
    ip.addParameter('p_cong', 0.8);
    ip.addParameter('choices', []);
    ip.addParameter('nonchoices', []);
    ip.addParameter('outcomes', []);
    ip.addParameter('rewards', []);
    ip.addParameter('new_sess', []);
    ip.addParameter('forced', []);
    ip.addParameter('initQ', 0.5);
    ip.addParameter('trans_common', []);
    ip.addParameter('freeLR', freeLR);
    ip.parse(varargin{:});
    params = ip.Results;
    parse_vars(params);
    nt = sum(~params.forced);
    x = nan(numel(agents), nt);
    dx = [];
    Q_val = cell(1, numel(agents));
    for agent_i = 1:numel(agents)
        if strcmp(agents{agent_i}, 'betaBias')
            Q_val{agent_i} = [0, 0];
        else
            Q_val{agent_i} = params.initQ * ones(1,2);
        end
    end
    prev_choice = nan;
    prev_reward = nan;
    prev_trans_common = nan;
    t2 = 1;
    for t1 = 1:length(params.choices)
        if params.new_sess(t1)
            for agent_i = 1:numel(agents)
                if ~strcmp(agents{agent_i}, 'betaBias')
                    Q_val{agent_i} = params.initQ * ones(1,2);
                end
            end
            prev_choice = nan;
            prev_reward = nan;
            prev_trans_common = nan;
        end
        if ~params.forced(t1)
            for agent_i = 1:numel(agents)
                switch agents{agent_i}
                    case 'betaBias'
                        if isnan(prev_choice)
                            out_val = 0;
                        elseif prev_choice == 1
                            out_val = 1;
                        else
                            out_val = -1;
                        end
                    case 'betaMFc'
                        if ~isnan(prev_choice)
                            lr = params.freeLR(1); % MFc
                            Q_val{agent_i} = Q_val{agent_i} * (1 - lr);
                            Q_val{agent_i}(prev_choice) = Q_val{agent_i}(prev_choice) + lr;
                        end
                        out_val = Q_val{agent_i}(1) - Q_val{agent_i}(2);
                    case 'betaMFr'
                        if ~isnan(prev_choice) && ~isnan(prev_reward)
                            lr = params.freeLR(2); % MFr
                            Q_val{agent_i} = Q_val{agent_i} * (1 - lr);
                            Q_val{agent_i}(prev_choice) = Q_val{agent_i}(prev_choice) + lr * prev_reward;
                        end
                        out_val = Q_val{agent_i}(1) - Q_val{agent_i}(2);
                    case 'betaMBc'
                        if ~isnan(prev_choice) && ~isnan(prev_trans_common)
                            lr = params.freeLR(3); % MBc
                            Q_val{agent_i} = Q_val{agent_i} * (1 - lr);
                            if prev_trans_common
                                Q_val{agent_i}(prev_choice) = Q_val{agent_i}(prev_choice) + lr;
                            else
                                Q_val{agent_i}(3 - prev_choice) = Q_val{agent_i}(3 - prev_choice) + lr;
                            end
                        end
                        out_val = Q_val{agent_i}(1) - Q_val{agent_i}(2);
                    case 'betaMB'
                        if ~isnan(prev_choice) && ~isnan(prev_trans_common) && ~isnan(prev_reward)
                            lr = params.freeLR(4); % MB
                            Q_val{agent_i} = Q_val{agent_i} * (1 - lr);
                            if prev_trans_common
                                Q_val{agent_i}(prev_choice) = Q_val{agent_i}(prev_choice) + lr * prev_reward;
                            else
                                Q_val{agent_i}(3 - prev_choice) = Q_val{agent_i}(3 - prev_choice) + lr * prev_reward;
                            end
                        end
                        out_val = Q_val{agent_i}(1) - Q_val{agent_i}(2);
                    otherwise
                        error('Unknown agent %s', agents{agent_i});
                end
                x(agent_i, t2) = out_val;
            end
            prev_choice = params.choices(t1);
            prev_reward = params.rewards(t1);
            prev_trans_common = params.trans_common(t1);
            t2 = t2 + 1;
        end
    end
end

function parse_vars(ip)
    fn = fieldnames(ip);
    for fi = 1:numel(fn)
        assignin('caller', fn{fi}, ip.(fn{fi}));
    end
end
