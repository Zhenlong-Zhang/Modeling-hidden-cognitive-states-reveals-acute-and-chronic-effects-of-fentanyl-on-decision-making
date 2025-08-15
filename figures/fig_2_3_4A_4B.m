% Written by Eric Garr (2025)

load 4LR

% need to re-order struct fields because of discrepency with ZZ.

new_order_coh4 = {'rat9_data', 'rat10_data', 'rat11_data', 'rat12_data'};
allResults.coh4_fent_train_fent = orderfields(allResults.coh4_fent_train_fent,new_order_coh4);
allResults.coh4_fent_train_suc = orderfields(allResults.coh4_fent_train_suc,new_order_coh4);

new_order_coh5 = {'rat6_data', 'rat8_data', 'rat10_data'};
allResults.coh5_suc_train_fent = orderfields(allResults.coh5_suc_train_fent,new_order_coh5);
allResults.coh5_suc_train_suc = orderfields(allResults.coh5_suc_train_suc,new_order_coh5);

new_order_coh8 = {'rat1_data', 'rat3_data', 'rat10_data'};
allResults.coh8_fent_train_fent = orderfields(allResults.coh8_fent_train_fent,new_order_coh8);
allResults.coh8_fent_train_suc = orderfields(allResults.coh8_fent_train_suc,new_order_coh8);

% continue

max_cohorts = 9;
max_states = 7;

training_rewards = {'fent','suc','fent','fent','suc','suc','suc','fent','suc'};

numBins = 9;

for i = 1:max_cohorts % loop over cohorts
 
    cohort_struct_fent{i} = ['coh',num2str(i),'_',training_rewards{i},'_train_fent'];
    cohort_struct_suc{i} = ['coh',num2str(i),'_',training_rewards{i},'_train_suc'];
    
    rat_fieldnames{i} = fieldnames(allResults.(cohort_struct_fent{i}));
    
    for j = 1:length(fieldnames(allResults.(cohort_struct_fent{i}))) % loop over rats within a cohort

        gammas_fent{i}{j} = get_gammas(max_states,allResults.(cohort_struct_fent{i}).(rat_fieldnames{i}{j}));
        gammas_fent_nStates3{i}{j} = gammas_fent{i}{j}{3};
        
        gammas_suc{i}{j} = get_gammas(max_states,allResults.(cohort_struct_suc{i}).(rat_fieldnames{i}{j}));
        gammas_suc_nStates3{i}{j} = gammas_suc{i}{j}{3};
        
        % divide fentanyl sessions into bins
        
        addpath(['data-cleaned\',cohort_struct_fent{i}])
        fent_data{i}{j} = readmatrix(rat_fieldnames{i}{j});
        fent_sess_start{i}{j} = find(fent_data{i}{j}(:,4)); % session boundaries
        
        for m = 1:length(fent_sess_start{i}{j}) % loop over fentanyl sessions
            
            if m ~= length(fent_sess_start{i}{j})
                
                gammas_per_sess_fent{i}{j}{m} = gammas_fent_nStates3{i}{j}(fent_sess_start{i}{j}(m):fent_sess_start{i}{j}(m+1)-1,:);
                
            else
                
                gammas_per_sess_fent{i}{j}{m} = gammas_fent_nStates3{i}{j}(fent_sess_start{i}{j}(m):length(fent_data{i}{j}(:,4)),:);
                
            end
            
           binned_gammas_fent{i}{j}{m} = get_binned_gammas(gammas_per_sess_fent{i}{j}{m},numBins);
            
        end
        
        % divide sucrose sessions into bins
        
        rmpath(['data-cleaned\',cohort_struct_fent{i}])
        addpath(['data-cleaned\',cohort_struct_suc{i}])
        suc_data{i}{j} = readmatrix(rat_fieldnames{i}{j});
        suc_sess_start{i}{j} = find(suc_data{i}{j}(:,4)); % session boundaries
        
        for m = 1:length(suc_sess_start{i}{j}) % loop over sucrose sessions
            
            if m ~= length(suc_sess_start{i}{j})
                
                gammas_per_sess_suc{i}{j}{m} = gammas_suc_nStates3{i}{j}(suc_sess_start{i}{j}(m):suc_sess_start{i}{j}(m+1)-1,:);
                
            else
                
                gammas_per_sess_suc{i}{j}{m} = gammas_suc_nStates3{i}{j}(suc_sess_start{i}{j}(m):length(suc_data{i}{j}(:,4)),:);
                
            end
            
            binned_gammas_suc{i}{j}{m} = get_binned_gammas(gammas_per_sess_suc{i}{j}{m},numBins);
            
        end
        
        rmpath(['data-cleaned\',cohort_struct_suc{i}])
        
        % average binned gammas over sessions

        mean_binned_gammas_fent{i}{j} = cat(3,binned_gammas_fent{i}{j}{:});
        mean_binned_gammas_suc{i}{j} = cat(3,binned_gammas_suc{i}{j}{:});

        % betas
        
        betas_fent{i}{j} = get_betas(max_states,allResults.(cohort_struct_fent{i}).(rat_fieldnames{i}{j}));
        betas_suc{i}{j} = get_betas(max_states,allResults.(cohort_struct_suc{i}).(rat_fieldnames{i}{j}));

        betas_fent_nState3{i}{j} = betas_fent{i}{j}{3};
        betas_suc_nState3{i}{j} = betas_suc{i}{j}{3};

    end
end

%% organize betas into state x agent matrices per rat

weights_fent = cell(1, 35); 
weights_suc = cell(1, 35); 
index = 1;

for i = 1:numel(betas_fent_nState3)
    inner_cells_fent = betas_fent_nState3{i}; 
    inner_cells_suc = betas_suc_nState3{i};
    for j = 1:numel(inner_cells_fent)
        weights_fent{index} = inner_cells_fent{j}; 
        weights_suc{index} = inner_cells_suc{j};
        rats_weights_combined{index} = (weights_fent{index} + weights_suc{index})/2;
        index = index + 1;
    end
end

%% organize gammas into bin x state matrices per rat

gammas_fent = cell(1, 35); 
gammas_suc = cell(1, 35);  
index = 1;

for i = 1:numel(mean_binned_gammas_fent)
    inner_cells_fent = mean_binned_gammas_fent{i};  
    inner_cells_suc = mean_binned_gammas_suc{i};
    for j = 1:numel(inner_cells_fent)
        gammas_fent{index} = inner_cells_fent{j};  
        gammas_suc{index} = inner_cells_suc{j};
        gammas_comb{index} = (gammas_fent{index}+gammas_suc{index})/2;
        mean_gammas_comb{index} = mean(gammas_comb{index},3);
        index = index + 1;
    end
end

%% get transition probabilities

for i = 1:max_cohorts % loop over cohorts
    for j = 1:length(fieldnames(allResults.(cohort_struct_fent{i}))) % loop over rats within a cohort

        trans_prob_fent{i}{j} = get_transition_probs(max_states,allResults.(cohort_struct_fent{i}).(rat_fieldnames{i}{j}));
        trans_prob_fent_nStates3_coh{i}{j} = trans_prob_fent{i}{j}{3};

        trans_prob_suc{i}{j} = get_transition_probs(max_states,allResults.(cohort_struct_suc{i}).(rat_fieldnames{i}{j}));
        trans_prob_suc_nStates3_coh{i}{j} = trans_prob_suc{i}{j}{3};

    end

    trans_prob_fent_coh_cat{i} = cat(3,trans_prob_fent_nStates3_coh{i}{:});
    trans_prob_suc_coh_cat{i} = cat(3,trans_prob_suc_nStates3_coh{i}{:});
end

% concatentate over cohorts

trans_prob_fent_cat = cat(3,trans_prob_fent_coh_cat{:});
trans_prob_suc_cat = cat(3,trans_prob_suc_coh_cat{:});

%% set up full tensor

% subjects x sessions x agents x time bins x states

subject = 1:35;

for i = 1:length(subject) % subjects

    for j = 1:size(gammas_fent{i},3) % fent sessions

        for k = 1:5 % agents

            X_full(i,j,k,:,:) = weights_fent{i}(:,k)' .* gammas_fent{i}(:,j);

        end

        for j = 1:size(gammas_suc{i},3) % suc sessions

            for k = 1:5 % agents

            X_full(i,j+3,k,:,:) = weights_suc{i}(:,k)' .* gammas_suc{i}(:,j);

        end

        end
    end
end

%% set up partial tensor (average over states)

X_partial = squeeze(mean(X_full,5));

%% TCA

tca_path = 'Z:\Eric\OF 2step\modelling\4LR\github code\tensor_toolbox-v3.6'; 
addpath(genpath(tca_path));

T = tensor(X_partial);
total_variance = norm(T)^2;

R = 1:9;
iters = 500;
h = waitbar(0, 'Progress...');

for i = 1:length(R)
    for j = 1:iters

        rng(j); 
        waitbar(j / iters, h, sprintf('Progress: %d%%', round(j / iters * 100)));

        [P{i}{j},Uinit{i}{j},output{i}{j}] = cp_als(T,R(i));
    
        % reconstruction error
        T_hat{i}{j} = full(P{i}{j});
        recon_error(i,j) = norm(T - T_hat{i}{j})^2;
        norm_recon_error(i,j) = recon_error(i,j) / total_variance;

    end

end

% get similarity score

for i = 1:length(R)

    min_error_ind(i) = find(norm_recon_error(i,:) == min(norm_recon_error(i,:)));

    for j = 1:iters

        % similarity score

        similarity_matrix(i,j)= score(P{i}{j}, P{i}{min_error_ind(i)});

    end

end

mean_sim = mean(similarity_matrix,2);

best_iter = find(similarity_matrix(3,:) == max(similarity_matrix(3,:))); 

%% fig 2

figure('Position', [500 100 445 650]);
subplot(3,2,1)
plot(mean(norm_recon_error,2),'k','linewidth',2)
xticks(1:9)
xticklabels({'1','2','3','4','5','6','7','8','9'})
xlabel('components')
ylabel('reconstruction error')
ylim([.1 .6])
ax = gca;
ax.FontSize = 10; 

subplot(3,2,2)
plot(mean_sim,'k','linewidth',2)
xticks(1:9)
xticklabels({'1','2','3','4','5','6','7','8','9'})
xlabel('components')
ylabel('similarity score')
ax = gca;
ax.FontSize = 10; 

subplot(3,2,3)
plot([1 2 3],[P{3}{best_iter}.u{1}(:,1) P{3}{best_iter}.u{1}(:,2) P{3}{best_iter}.u{1}(:,3)],'color',[.5 .5 .5])
hold on;
scatter(ones(35,1),P{3}{best_iter}.u{1}(:,1),25,'MarkerEdgeColor','r')
scatter(ones(35,1)*2,P{3}{best_iter}.u{1}(:,2),25,'MarkerEdgeColor','b')
scatter(ones(35,1)*3,P{3}{best_iter}.u{1}(:,3),25,'MarkerEdgeColor','k')
xlim([0 4])
xticks(1:3)
xticklabels({'C1','C2','C3'})
ylabel('subject-mode loading')
ylim([-.5 1])
ax = gca;
ax.FontSize = 10; 

% show session-mode loadings

fent_sess = P{3}{best_iter}.u{2}(1:3,:);
suc_sess = P{3}{best_iter}.u{2}(4:end,:);

subplot(3,2,4)
plot(1:3,fent_sess(1:3,1),'Color', 'r','linewidth',2)
hold on;
plot(1:3,fent_sess(1:3,2),'Color', 'b','linewidth',2)
plot(1:3,fent_sess(1:3,3),'Color', 'k','linewidth',2)
plot(5:7,suc_sess(1:3,1),'Color', 'r','linewidth',2)
plot(5:7,suc_sess(1:3,2),'Color', 'b','linewidth',2)
plot(5:7,suc_sess(1:3,3),'Color', 'k','linewidth',2)
xlim([0 8])
xticks([1 2 3 5 6 7])
xticklabels({'1','2','3','1','2','3'});
xlabel('sessions')
ylabel('session-mode loading','fontsize',12)
ylim([-.5 1])
ax = gca;
ax.FontSize = 10; 

% show agent-mode loadings

subplot(3,2,5)
b = bar(P{3}{best_iter}.u{3},...
    'LineWidth', 1);
b(1).EdgeColor = 'red';
b(1).FaceColor = 'red';
b(2).EdgeColor = 'blue';
b(2).FaceColor = 'blue';
b(3).EdgeColor = 'black';
b(3).FaceColor = 'black';
xticklabels({'MB','MF','persev','side bias','TP'})
xtickangle(45)
ylabel('agent-mode loading','fontsize',12)
ylim([-.5 1])
ax = gca;
ax.FontSize = 10; 

% show bin-mode loadings

subplot(3,2,6)
plot(P{3}{best_iter}.u{4}(:,1),'r','linewidth',2)
hold on;
plot(P{3}{best_iter}.u{4}(:,2),'b','linewidth',2)
plot(P{3}{best_iter}.u{4}(:,3),'k','linewidth',2)
xlim([0 10])
xticks([1 5 9])
xticklabels({'0','.5','1'})
xlabel('time bins (normalized)')
ylabel('bin-mode loading','fontsize',12)
ylim([-.5 1])
ax = gca;
ax.FontSize = 10; 

%% assign group and sex

coh1_sex = [1 1 1 0 0 0];
coh2_sex = [1 0 0 0 0];
coh3_opto1_fent_train_sex = [1 1 1 1];
coh3_opto2_fent_train_sex = [0 0 0 0];
coh3_opto1_suc_train_sex = [0 0 0];
coh3_opto2_suc_train_sex = [1 1 1];
coh4_sex = [1 1 1];
coh5_fent_train_sex = [1 1 0];
coh5_suc_train_sex = [1 1 1 0];

% sex. 1 = male, 0 = female
sex_cat = horzcat(coh1_sex,coh2_sex,coh3_opto1_fent_train_sex,coh3_opto2_fent_train_sex,coh3_opto1_suc_train_sex,coh3_opto2_suc_train_sex,coh4_sex,coh5_fent_train_sex,coh5_suc_train_sex)';
% group. 1 = fent train, 0 = suc train
group_cat = [ones(1,6),zeros(1,5),ones(1,8),zeros(1,6),zeros(1,3),ones(1,3),zeros(1,4)]';

%% align state labels based on TCA results using Hungarian algorithm

aligned_state_labels_fent = NaN(length(subject), 3); 
aligned_state_labels_suc = NaN(length(subject), 3); 

for i = 1:length(subject)

    weighted_weights_fent{i} = weights_fent{i} .* mean(mean(gammas_fent{i},3))';
    weighted_weights_suc{i} = weights_suc{i} .* mean(mean(gammas_suc{i},3))';
    
    canonical_agent_state_matrix_fent{i} = P{3}{best_iter}.lambda' .* (P{3}{best_iter}.u{3} .* (P{3}{best_iter}.u{1}(i,:) .* mean(fent_sess)));
    canonical_agent_state_matrix_suc{i} = P{3}{best_iter}.lambda' .* (P{3}{best_iter}.u{3} .* (P{3}{best_iter}.u{1}(i,:) .* mean(suc_sess)));
    agent_state_matrix_fent = weighted_weights_fent{i}';
    agent_state_matrix_suc = weighted_weights_suc{i}';

    % Build cost matrix (3 × 3): cost(i,j) = dissimilarity between subject's state i and canonical state j
    for j = 1:3  % subject states
        for k = 1:3  % canonical states
            v1_fent = agent_state_matrix_fent(:,j);
            v1_suc = agent_state_matrix_suc(:,j);
            v2_fent = canonical_agent_state_matrix_fent{i}(:,k);
            v2_suc = canonical_agent_state_matrix_suc{i}(:,k);
            % Dissimilarity measure — use squared Euclidean distance
            cost_matrix_fent{i}(j, k) = sum((v1_fent - v2_fent).^2); %-corr(v1_fent,v2_fent); 
            cost_matrix_suc{i}(j, k) = sum((v1_suc - v2_suc).^2); %-corr(v1_suc,v2_suc); 
        end
    end

    % Solve the assignment problem (Hungarian algorithm)
    % We want to permute subject's rows to best match canonical rows
    assignment_fent{i} = munkres(cost_matrix_fent{i}); 
    assignment_suc{i} = munkres(cost_matrix_suc{i}); 

    for m = 1:3

        aligned_state_labels_fent(i,m) = find(assignment_fent{i}(m,:) == 1);
        aligned_state_labels_suc(i,m) = find(assignment_suc{i}(m,:) == 1);

    end
end

%% figs 3A and B

i = 6;
heatmap_lim = [-.48 .99];

figure('Position', [500 100 633 225]);
ax = subplot(1,4,1);
imagesc(P{3}{best_iter}.lambda')
ax.XTick = 1:3;
ax.XTickLabel = ({'C1','C2','C3'});
title('lambdas')
colorbar('northoutside')
ax = gca;
ax.FontSize = 10; 

ax = subplot(1,4,2);
imagesc(P{3}{best_iter}.u{3}, heatmap_lim)
colormap('hot')
ax.XTick = 1:3;
ax.XTickLabel = ({'C1','C2','C3'});
ax.YTick = 1:5;
ax.YTickLabel = ({'MB','MF','persev','bias','TP'});
title('agent loadings')
colorbar('northoutside')
ax = gca;
ax.FontSize = 10; 

ax = subplot(1,4,3);
imagesc(mean(fent_sess), heatmap_lim)
ax.XTick = 1:3;
ax.XTickLabel = ({'C1','C2','C3'});
title('session loadings')
colorbar('northoutside')
ax = gca;
ax.FontSize = 10; 

ax = subplot(1,4,4);
imagesc(P{3}{best_iter}.u{1}(i,:), heatmap_lim)
ax.XTick = 1:3;
ax.XTickLabel = ({'C1','C2','C3'});
title({'subject loadings'})
colorbar('northoutside')
ax = gca;
ax.FontSize = 10; 

figure('Position', [500 100 425 275]);
ax = subplot(1,3,1);
imagesc(P{3}{best_iter}.lambda' .* (P{3}{best_iter}.u{3} .* (P{3}{best_iter}.u{1}(i,:) .* mean(fent_sess))))
ax.XTick = 1:3;
ax.XTickLabel = ({'C1','C2','C3'});
ax.YTick = 1:5;
ax.YTickLabel = ({'MB','MF','persev','bias','TP'});
title({'agent x',' components'})
colorbar('northoutside')
ax = gca;
ax.FontSize = 10; 
colormap('hot')

agent_state_matrix_fent_example = weighted_weights_fent{i}';

ax = subplot(1,3,2);
imagesc(agent_state_matrix_fent_example)
ax.XTick = 1:3;
ax.XTickLabel = ({'s1','s2','s3'});
title({'agent x states','(unsorted)'})
colorbar('northoutside')
ax = gca;
ax.FontSize = 10; 

ax = subplot(1,3,3);
imagesc(agent_state_matrix_fent_example(:,aligned_state_labels_fent(i,:)))
ax.XTick = 1:3;
ax.XTickLabel = ({'s1','s2','s3'});
title({'agent x states','(sorted)'})
colorbar('northoutside')
ax = gca;
ax.FontSize = 10; 

%% align betas and gammas to new state ID's

for i = 1:length(subject)

    % betas
    rats_weights_fent_sorted{i} = weights_fent{i}(aligned_state_labels_fent(i,:), :);
    rats_weights_suc_sorted{i} = weights_suc{i}(aligned_state_labels_suc(i,:), :);

    MFc_fent_sorted(i,:) = rats_weights_fent_sorted{i}(:,1)';
    MFr_fent_sorted(i,:) = rats_weights_fent_sorted{i}(:,2)';
    MBc_fent_sorted(i,:) = rats_weights_fent_sorted{i}(:,3)';
    MBr_fent_sorted(i,:) = rats_weights_fent_sorted{i}(:,4)';
    bias_fent_sorted(i,:) = rats_weights_fent_sorted{i}(:,5)';

    MFc_suc_sorted(i,:) = rats_weights_suc_sorted{i}(:,1)';
    MFr_suc_sorted(i,:) = rats_weights_suc_sorted{i}(:,2)';
    MBc_suc_sorted(i,:) = rats_weights_suc_sorted{i}(:,3)';
    MBr_suc_sorted(i,:) = rats_weights_suc_sorted{i}(:,4)';
    bias_suc_sorted(i,:) = rats_weights_suc_sorted{i}(:,5)';

    % gammas
    gammas_fent_sorted{i} = gammas_fent{i}(:,aligned_state_labels_fent(i,:),:);
    gammas_suc_sorted{i} = gammas_suc{i}(:,aligned_state_labels_suc(i,:),:);
    gammas_comb_sorted{i} = (mean(gammas_fent_sorted{i},3)+mean(gammas_suc_sorted{i},3))/2;

    mean_gamas_fent_sorted{i} = mean(gammas_fent_sorted{i},3);
    mean_gamas_suc_sorted{i} = mean(gammas_suc_sorted{i},3);

    % weighted betas
    weighted_MFc_fent_sorted(i,:) = MFc_fent_sorted(i,:) .* mean(mean(gammas_fent_sorted{i},3));
    weighted_MFr_fent_sorted(i,:) = MFr_fent_sorted(i,:) .* mean(mean(gammas_fent_sorted{i},3));
    weighted_MBc_fent_sorted(i,:) = MBc_fent_sorted(i,:) .* mean(mean(gammas_fent_sorted{i},3));
    weighted_MBr_fent_sorted(i,:) = MBr_fent_sorted(i,:) .* mean(mean(gammas_fent_sorted{i},3));
    weighted_bias_fent_sorted(i,:) = bias_fent_sorted(i,:) .* mean(mean(gammas_fent_sorted{i},3));

    weighted_MFc_suc_sorted(i,:) = MFc_suc_sorted(i,:) .* mean(mean(gammas_suc_sorted{i},3));
    weighted_MFr_suc_sorted(i,:) = MFr_suc_sorted(i,:) .* mean(mean(gammas_suc_sorted{i},3));
    weighted_MBc_suc_sorted(i,:) = MBc_suc_sorted(i,:) .* mean(mean(gammas_suc_sorted{i},3));
    weighted_MBr_suc_sorted(i,:) = MBr_suc_sorted(i,:) .* mean(mean(gammas_suc_sorted{i},3));
    weighted_bias_suc_sorted(i,:) = bias_suc_sorted(i,:) .* mean(mean(gammas_suc_sorted{i},3));

end

cat_gammas_fent_sorted = cat(3,mean_gamas_fent_sorted{:});
cat_gammas_suc_sorted = cat(3,mean_gamas_suc_sorted{:});

for i = 1:35
trans_prob_sorted_fent(:,:,i) = trans_prob_fent_cat(aligned_state_labels_fent(i,:),aligned_state_labels_fent(i,:),i);
trans_prob_sorted_suc(:,:,i) = trans_prob_suc_cat(aligned_state_labels_suc(i,:),aligned_state_labels_suc(i,:),i);
end

trans_prob_comb = (trans_prob_sorted_fent+trans_prob_sorted_suc)/2;

%% fig s 3C and D

figure('Position', [500 200 420 400]);

ax = subplot(2,2,1);
cat_gammas_comb_sorted = cat(3,gammas_comb_sorted{:});
gammas_s1 = cat_gammas_comb_sorted(:,1,:);
gammas_s2 = cat_gammas_comb_sorted(:,2,:);
gammas_s3 = cat_gammas_comb_sorted(:,3,:);

shadedErrorBar(1:numBins,mean(gammas_s1,3),std(gammas_s1,[],3)/sqrt(length(subject)),...
    'lineProps',{'-','Color', 'r', 'LineWidth', 1.5})
hold on;
shadedErrorBar(1:numBins,mean(gammas_s2,3),std(gammas_s2,[],3)/sqrt(length(subject)),...
    'lineProps',{'-','Color', 'b', 'LineWidth', 1.5})
shadedErrorBar(1:numBins,mean(gammas_s3,3),std(gammas_s3,[],3)/sqrt(length(subject)),...
    'lineProps',{'-','Color', 'k', 'LineWidth', 1.5})
xticks([0 5 9])
xticklabels({'0','0.5','1'})
xlabel('session time (normalized)','fontsize',12)
ylabel('state probability','fontsize',12)
xlim([0 numBins+1])
ylim([.28 .42])
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

ax = subplot(2,2,2);
means = [mean(squeeze(cat(3,mean(gammas_s1)))), mean(squeeze(cat(3,mean(gammas_s2)))), mean(squeeze(cat(3,mean(gammas_s3))))];

b = bar(means,...
    'LineWidth', 1);
hold on;
plot([1 2 3],[squeeze(cat(3,mean(gammas_s1))) squeeze(cat(3,mean(gammas_s2))) squeeze(cat(3,mean(gammas_s3)))],'color',[.7 .7 .7])
b(1).EdgeColor = 'k';
b(1).FaceColor = 'k';
xlabel('states')
ylabel('state probability','fontsize',12)
ylim([0 .6])
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

ax = subplot(2,2,3);
heatmap_lim = [.23 .4];

imagesc(mean(trans_prob_comb,3), heatmap_lim)
ax.XTick = 1:3;
xlabel('model weights')
ax.YTick = 1:3;
xlabel('state (t)','fontsize',12)
ylabel('state (t-1)','fontsize',12)
colormap gray
colorbar
title('transition matrix','fontsize',12)
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

ax = subplot(2,2,4);
means = [mean((MBr_fent_sorted+MBr_suc_sorted)/2); mean((MFr_fent_sorted+MFr_suc_sorted)/2); mean((MFc_fent_sorted+MFc_suc_sorted)/2); mean((bias_fent_sorted+bias_suc_sorted)/2); mean((MBc_fent_sorted+MBc_suc_sorted)/2)];
sems = [std((MBr_fent_sorted+MBr_suc_sorted)/2)/sqrt(length(subject)); std((MFr_fent_sorted+MFr_suc_sorted)/2)/sqrt(length(subject)); std((MFc_fent_sorted+MFc_suc_sorted)/2)/sqrt(length(subject));...
    std((bias_fent_sorted+bias_suc_sorted)/2)/sqrt(length(subject)); std((MBc_fent_sorted+MBc_suc_sorted)/2)/sqrt(length(subject))];
xs = [.78 1 1.22; 1.78 2 2.22; 2.78 3 3.22; 3.78 4 4.22; 4.78 5 5.22];

b = bar(means,...
    'LineWidth', 1);
hold on;
e = errorbar(xs,means,sems,'.', 'MarkerSize',0.01);
e(1).Color = 'r';
e(2).Color = 'b';
e(3).Color = 'k';
b(1).EdgeColor = 'r';
b(1).FaceColor = 'r';
b(2).EdgeColor = 'b';
b(2).FaceColor = 'b';
b(3).EdgeColor = 'k';
b(3).FaceColor = 'k';
xticks([1 2 3 4 5])
xticklabels({'MB','MF','persev','side bias','TP'})
xtickangle(45)
ylabel('agent weight','fontsize',12)
ylim([-2 6])
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

%% fig 4A

figure('Position', [500 200 284 250]);
ax = subplot(1,2,1);
heatmap_lim = [.19 .48];

imagesc(mean(trans_prob_sorted_fent,3), heatmap_lim)
ax.XTick = 1:3;
xlabel('model weights')
ax.YTick = 1:3;
xlabel('state (t)','fontsize',12)
ylabel('state (t-1)','fontsize',12)
colormap gray
colorbar('northoutside')
title('fentanyl','fontsize',12)
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

ax = subplot(1,2,2);

imagesc(mean(trans_prob_sorted_suc,3), heatmap_lim)
ax.XTick = 1:3;
xlabel('model weights')
ax.YTick = 1:3;
xlabel('state (t)','fontsize',12)
ylabel('state (t-1)','fontsize',12)
colormap gray
colorbar('northoutside')
title('sucrose','fontsize',12)
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

%% fig 4B

weighted_MBr_fent_train_fent = weighted_MBr_fent_sorted(group_cat == 1,:);
weighted_MBr_fent_train_suc = weighted_MBr_suc_sorted(group_cat == 1,:);
weighted_MBr_suc_train_fent = weighted_MBr_fent_sorted(group_cat == 0,:);
weighted_MBr_suc_train_suc = weighted_MBr_suc_sorted(group_cat == 0,:);

weighted_MBr_fent_train = (weighted_MBr_fent_train_fent+weighted_MBr_fent_train_suc)/2;
weighted_MBr_suc_train = (weighted_MBr_suc_train_fent+weighted_MBr_suc_train_suc)/2;

y_lim = [-6 5];

figure('Position', [500 200 425 200]);
subplot(1,3,1)
plot(weighted_MBr_fent_train','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(weighted_MBr_fent_train(:,1)) mean(weighted_MBr_fent_train(:,1))],'color','r','linewidth',3)
plot([1.8 2.2],[mean(weighted_MBr_fent_train(:,2)) mean(weighted_MBr_fent_train(:,2))],'color','b','linewidth',3)
plot([2.8 3.2],[mean(weighted_MBr_fent_train(:,3)) mean(weighted_MBr_fent_train(:,3))],'color','k','linewidth',3)
xlabel('states','fontsize',12)
xlim([0 4])
xticks(1:3)
ylabel('MB weight')
title('fentanyl trained')
ax = gca;
ax.FontSize = 10; 
ylim(y_lim)

subplot(1,3,2)
plot(weighted_MBr_suc_train','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(weighted_MBr_suc_train(:,1)) mean(weighted_MBr_suc_train(:,1))],'color','r','linewidth',3)
plot([1.8 2.2],[mean(weighted_MBr_suc_train(:,2)) mean(weighted_MBr_suc_train(:,2))],'color','b','linewidth',3)
plot([2.8 3.2],[mean(weighted_MBr_suc_train(:,3)) mean(weighted_MBr_suc_train(:,3))],'color','k','linewidth',3)
xlabel('states','fontsize',12)
xlim([0 4])
xticks(1:3)
ylabel('MB weight')
title('sucrose trained')
ax = gca;
ax.FontSize = 10; 
ylim(y_lim)

subplot(1,3,3)
scatter(ones(length(weighted_MBr_fent_train),1),weighted_MBr_fent_train(:,1),75,'markeredgecolor',[.7 .7 .7])
hold on;
scatter(ones(length(weighted_MBr_suc_train),1)*2,weighted_MBr_suc_train(:,1),75,'markeredgecolor',[.7 .7 .7])
plot([.8 1.2],[mean(weighted_MBr_fent_train(:,1)),mean(weighted_MBr_fent_train(:,1))],'color',[.2745 .5098 .7059],'linewidth',3)
plot([1.8 2.2],[mean(weighted_MBr_suc_train(:,1)),mean(weighted_MBr_suc_train(:,1))],'color',[.9412 .502 .502],'linewidth',3)
xlim([0 3])
ylim([-3 4])
xticks(1:2)
xticklabels({'fentanyl','sucrose'})
xlabel('training group')
ylabel('MB weight')
title('state 1')
ax = gca;

ax.FontSize = 10; 

