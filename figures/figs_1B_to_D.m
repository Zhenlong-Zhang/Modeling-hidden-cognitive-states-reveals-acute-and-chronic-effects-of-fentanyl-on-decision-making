% Written by Eric Garr (2025)

load 4LR

% need to re-order struct fields because of discrepency with ZZ

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

cohorts = 9;
max_states = 7;

training_rewards = {'fent','suc','fent','fent','suc','suc','suc','fent','suc'};

numBins = 9;

for i = 1:cohorts % loop over cohorts
 
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

        % get betas
        
        % fentanyl sessions
        
        betas_fent{i}{j} = get_betas(max_states,allResults.(cohort_struct_fent{i}).(rat_fieldnames{i}{j}));
        betas_fent_nStates3{i}{j} = betas_fent{i}{j}{3}';
        MFc_fent{i}{j} = betas_fent_nStates3{i}{j}(1,:);
        MFr_fent{i}{j} = betas_fent_nStates3{i}{j}(2,:);
        MBc_fent{i}{j} = betas_fent_nStates3{i}{j}(3,:);
        MBr_fent{i}{j} = betas_fent_nStates3{i}{j}(4,:);
        bias_fent{i}{j} = betas_fent_nStates3{i}{j}(5,:);

        % sucrose sessions

        betas_suc{i}{j} = get_betas(max_states,allResults.(cohort_struct_suc{i}).(rat_fieldnames{i}{j}));
        betas_suc_nStates3{i}{j} = betas_suc{i}{j}{3}';
        MFc_suc{i}{j} = betas_suc_nStates3{i}{j}(1,:);
        MFr_suc{i}{j} = betas_suc_nStates3{i}{j}(2,:);
        MBc_suc{i}{j} = betas_suc_nStates3{i}{j}(3,:);
        MBr_suc{i}{j} = betas_suc_nStates3{i}{j}(4,:);
        bias_suc{i}{j} = betas_suc_nStates3{i}{j}(5,:);

        % sort states a la Venditto et al. (2024)

        % fentanyl sessions

        for k = 1:size(mean_binned_gammas_fent{i}{j},3) % loop over sessions
            
            init_prob_fent{i}{j}(k,:) = gammas_per_sess_fent{i}{j}{k}(1,:);

            % ID state with highest initial probability per sessions and call it state 1. Then sort remaining states by MB weight.
            [sort_init_prob_fent{i}{j}(k,:),sort_init_prob_fent_ind{i}{j}(k,:)] = sort(init_prob_fent{i}{j}(k,:));
            s1_fent_ind{i}{j}(k,:) = sort_init_prob_fent_ind{i}{j}(k,end);
            s3_s2_fent_ind{i}{j}{k} = sortrows([sort_init_prob_fent_ind{i}{j}(k,1:2)', MBr_fent{i}{j}(sort_init_prob_fent_ind{i}{j}(k,1:2))'],2);
            sorted_states_fent{i}{j}(k,:) = horzcat(s1_fent_ind{i}{j}(k,:),flip(s3_s2_fent_ind{i}{j}{k}(:,1))');

            sorted_binned_gammas_fent{i}{j}(:,:,k) = mean_binned_gammas_fent{i}{j}(:,sorted_states_fent{i}{j}(k,:),k);

            MFc_fent_sorted{i}{j}(:,:,k) = MFc_fent{i}{j}(sorted_states_fent{i}{j}(k,:));
            MFr_fent_sorted{i}{j}(:,:,k) = MFr_fent{i}{j}(sorted_states_fent{i}{j}(k,:));
            MBc_fent_sorted{i}{j}(:,:,k) = MBc_fent{i}{j}(sorted_states_fent{i}{j}(k,:));
            MBr_fent_sorted{i}{j}(:,:,k) = MBr_fent{i}{j}(sorted_states_fent{i}{j}(k,:));
            bias_fent_sorted{i}{j}(:,:,k) = bias_fent{i}{j}(sorted_states_fent{i}{j}(k,:));

            % corrected agent weights (mulitplied by session-wide state probabilities)
            corrected_MFc_fent{i}{j}(k,:) = mean(MFc_fent_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_fent{i}{j}(:,:,k));
            corrected_MFr_fent{i}{j}(k,:) = mean(MFr_fent_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_fent{i}{j}(:,:,k));
            corrected_MBc_fent{i}{j}(k,:) = mean(MBc_fent_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_fent{i}{j}(:,:,k));
            corrected_MBr_fent{i}{j}(k,:) = mean(MBr_fent_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_fent{i}{j}(:,:,k));
            corrected_bias_fent{i}{j}(k,:) = mean(bias_fent_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_fent{i}{j}(:,:,k));

        end

        mean_sorted_binned_gammas_fent{i}{j} = mean(sorted_binned_gammas_fent{i}{j},3);

        mean_MFc_fent_sorted{i}{j} = mean(MFc_fent_sorted{i}{j},3);
        mean_MFr_fent_sorted{i}{j} = mean(MFr_fent_sorted{i}{j},3);
        mean_MBc_fent_sorted{i}{j} = mean(MBc_fent_sorted{i}{j},3);
        mean_MBr_fent_sorted{i}{j} = mean(MBr_fent_sorted{i}{j},3);
        mean_bias_fent_sorted{i}{j} = mean(bias_fent_sorted{i}{j},3);

        mean_corrected_MFc_fent{i}{j} = mean(corrected_MFc_fent{i}{j});
        mean_corrected_MFr_fent{i}{j} = mean(corrected_MFr_fent{i}{j});
        mean_corrected_MBc_fent{i}{j} = mean(corrected_MBc_fent{i}{j});
        mean_corrected_MBr_fent{i}{j} = mean(corrected_MBr_fent{i}{j});
        mean_corrected_bias_fent{i}{j} = mean(corrected_bias_fent{i}{j});

        % sucrose sessions

        for k = 1:size(mean_binned_gammas_suc{i}{j},3) % loop over sessions
            
            % Identify state with highest initial probability per sessions and call it state 1. Then sort remaining states by MB weight.
            init_prob_suc{i}{j}(k,:) = gammas_per_sess_suc{i}{j}{k}(1,:);
            [sort_init_prob_suc{i}{j}(k,:),sort_init_prob_suc_ind{i}{j}(k,:)] = sort(init_prob_suc{i}{j}(k,:));
            s1_suc_ind{i}{j}(k,:) = sort_init_prob_suc_ind{i}{j}(k,end);
            s3_s2_suc_ind{i}{j}{k} = sortrows([sort_init_prob_suc_ind{i}{j}(k,1:2)', MBr_suc{i}{j}(sort_init_prob_suc_ind{i}{j}(k,1:2))'],2);
            sorted_states_suc{i}{j}(k,:) = horzcat(s1_suc_ind{i}{j}(k,:),flip(s3_s2_suc_ind{i}{j}{k}(:,1))');

            sorted_binned_gammas_suc{i}{j}(:,:,k) = mean_binned_gammas_suc{i}{j}(:,sorted_states_suc{i}{j}(k,:),k);

            MFc_suc_sorted{i}{j}(:,:,k) = MFc_suc{i}{j}(sorted_states_suc{i}{j}(k,:));
            MFr_suc_sorted{i}{j}(:,:,k) = MFr_suc{i}{j}(sorted_states_suc{i}{j}(k,:));
            MBc_suc_sorted{i}{j}(:,:,k) = MBc_suc{i}{j}(sorted_states_suc{i}{j}(k,:));
            MBr_suc_sorted{i}{j}(:,:,k) = MBr_suc{i}{j}(sorted_states_suc{i}{j}(k,:));
            bias_suc_sorted{i}{j}(:,:,k) = bias_suc{i}{j}(sorted_states_suc{i}{j}(k,:));

            corrected_MFc_suc{i}{j}(k,:) = mean(MFc_suc_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_suc{i}{j}(:,:,k));
            corrected_MFr_suc{i}{j}(k,:) = mean(MFr_suc_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_suc{i}{j}(:,:,k));
            corrected_MBc_suc{i}{j}(k,:) = mean(MBc_suc_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_suc{i}{j}(:,:,k));
            corrected_MBr_suc{i}{j}(k,:) = mean(MBr_suc_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_suc{i}{j}(:,:,k));
            corrected_bias_suc{i}{j}(k,:) = mean(bias_suc_sorted{i}{j}(:,:,k) .* sorted_binned_gammas_suc{i}{j}(:,:,k));

        end

        mean_sorted_binned_gammas_suc{i}{j} = mean(sorted_binned_gammas_suc{i}{j},3);

        mean_MFc_suc_sorted{i}{j} = mean(MFc_suc_sorted{i}{j},3);
        mean_MFr_suc_sorted{i}{j} = mean(MFr_suc_sorted{i}{j},3);
        mean_MBc_suc_sorted{i}{j} = mean(MBc_suc_sorted{i}{j},3);
        mean_MBr_suc_sorted{i}{j} = mean(MBr_suc_sorted{i}{j},3);
        mean_bias_suc_sorted{i}{j} = mean(bias_suc_sorted{i}{j},3);

        mean_corrected_MFc_suc{i}{j} = mean(corrected_MFc_suc{i}{j});
        mean_corrected_MFr_suc{i}{j} = mean(corrected_MFr_suc{i}{j});
        mean_corrected_MBc_suc{i}{j} = mean(corrected_MBc_suc{i}{j});
        mean_corrected_MBr_suc{i}{j} = mean(corrected_MBr_suc{i}{j});
        mean_corrected_bias_suc{i}{j} = mean(corrected_bias_suc{i}{j});

        % sort transition probabilities

        % fentanyl sessions

        trans_prob_fent{i}{j} = get_transition_probs(max_states,allResults.(cohort_struct_fent{i}).(rat_fieldnames{i}{j}));
        trans_prob_fent_nStates3{i}{j} = trans_prob_fent{i}{j}{3};

        for k = 1:size(mean_binned_gammas_fent{i}{j},3) 

            trans_prob_fent_nStates3_sorted{i}{j}{k} = trans_prob_fent_nStates3{i}{j}(sorted_states_fent{i}{j}(k,:),sorted_states_fent{i}{j}(k,:));
        end

        mean_trans_prob_fent_sorted{i}{j} = mean(cat(3,trans_prob_fent_nStates3_sorted{i}{j}{:}),3);

        % sucrose sessions

        trans_prob_suc{i}{j} = get_transition_probs(max_states,allResults.(cohort_struct_suc{i}).(rat_fieldnames{i}{j}));
        trans_prob_suc_nStates3{i}{j} = trans_prob_suc{i}{j}{3};

        for k = 1:size(mean_binned_gammas_suc{i}{j},3) 

            trans_prob_suc_nStates3_sorted{i}{j}{k} = trans_prob_suc_nStates3{i}{j}(sorted_states_suc{i}{j}(k,:),sorted_states_suc{i}{j}(k,:));
        end

        mean_trans_prob_suc_sorted{i}{j} = mean(cat(3,trans_prob_suc_nStates3_sorted{i}{j}{:}),3);


    end

    % concatenate data within each cohort

    sorted_binned_gammas_fent_coh{i} = cat(3,mean_sorted_binned_gammas_fent{i}{:}); 
    sorted_binned_gammas_suc_coh{i} = cat(3,mean_sorted_binned_gammas_suc{i}{:}); 

    mean_MFc_fent_coh{i} = vertcat(mean_MFc_fent_sorted{i}{:});
    mean_MFr_fent_coh{i} = vertcat(mean_MFr_fent_sorted{i}{:});
    mean_MBc_fent_coh{i} = vertcat(mean_MBc_fent_sorted{i}{:});
    mean_MBr_fent_coh{i} = vertcat(mean_MBr_fent_sorted{i}{:});
    mean_bias_fent_coh{i} = vertcat(mean_bias_fent_sorted{i}{:});
    mean_MFc_suc_coh{i} = vertcat(mean_MFc_suc_sorted{i}{:});
    mean_MFr_suc_coh{i} = vertcat(mean_MFr_suc_sorted{i}{:});
    mean_MBc_suc_coh{i} = vertcat(mean_MBc_suc_sorted{i}{:});
    mean_MBr_suc_coh{i} = vertcat(mean_MBr_suc_sorted{i}{:});
    mean_bias_suc_coh{i} = vertcat(mean_bias_suc_sorted{i}{:});

    mean_corrected_MFc_fent_coh{i} = vertcat(mean_corrected_MFc_fent{i}{:});
    mean_corrected_MFr_fent_coh{i} = vertcat(mean_corrected_MFr_fent{i}{:});
    mean_corrected_MBc_fent_coh{i} = vertcat(mean_corrected_MBc_fent{i}{:});
    mean_corrected_MBr_fent_coh{i} = vertcat(mean_corrected_MBr_fent{i}{:});
    mean_corrected_bias_fent_coh{i} = vertcat(mean_corrected_bias_fent{i}{:});
    mean_corrected_MFc_suc_coh{i} = vertcat(mean_corrected_MFc_suc{i}{:});
    mean_corrected_MFr_suc_coh{i} = vertcat(mean_corrected_MFr_suc{i}{:});
    mean_corrected_MBc_suc_coh{i} = vertcat(mean_corrected_MBc_suc{i}{:});
    mean_corrected_MBr_suc_coh{i} = vertcat(mean_corrected_MBr_suc{i}{:});
    mean_corrected_bias_suc_coh{i} = vertcat(mean_corrected_bias_suc{i}{:});

    trans_prob_fent_cat{i} = cat(3,mean_trans_prob_fent_sorted{i}{:});
    trans_prob_suc_cat{i} = cat(3,mean_trans_prob_suc_sorted{i}{:});

end

% concatentate over cohorts

sorted_binned_gammas_combined_fent = cat(3,sorted_binned_gammas_fent_coh{:});
sorted_binned_gammas_combined_suc = cat(3,sorted_binned_gammas_suc_coh{:});

trans_prob_sorted_fent = cat(3,trans_prob_fent_cat{:});
trans_prob_sorted_suc = cat(3,trans_prob_suc_cat{:});

gammas_sorted_all = (sorted_binned_gammas_combined_fent+sorted_binned_gammas_combined_suc)/2;
gammas_s1 = squeeze(gammas_sorted_all(:,1,:))';
gammas_s2 = squeeze(gammas_sorted_all(:,2,:))';
gammas_s3 = squeeze(gammas_sorted_all(:,3,:))';

%% label by group and sex

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

% raw agent weights
MFc_fent_cat = vertcat(mean_MFc_fent_coh{:});
MFr_fent_cat = vertcat(mean_MFr_fent_coh{:});
MBc_fent_cat = vertcat(mean_MBc_fent_coh{:});
MBr_fent_cat = vertcat(mean_MBr_fent_coh{:});
bias_fent_cat = vertcat(mean_bias_fent_coh{:});
MFc_suc_cat = vertcat(mean_MFc_suc_coh{:});
MFr_suc_cat = vertcat(mean_MFr_suc_coh{:});
MBc_suc_cat = vertcat(mean_MBc_suc_coh{:});
MBr_suc_cat = vertcat(mean_MBr_suc_coh{:});
bias_suc_cat = vertcat(mean_bias_suc_coh{:});

% corrected agent weights
MFc_weighted_fent_cat = vertcat(mean_corrected_MFc_fent_coh{:});
MFr_weighted_fent_cat = vertcat(mean_corrected_MFr_fent_coh{:});
MBc_weighted_fent_cat = vertcat(mean_corrected_MBc_fent_coh{:});
MBr_weighted_fent_cat = vertcat(mean_corrected_MBr_fent_coh{:});
bias_weighted_fent_cat = vertcat(mean_corrected_bias_fent_coh{:});
MFc_weighted_suc_cat = vertcat(mean_corrected_MFc_suc_coh{:});
MFr_weighted_suc_cat = vertcat(mean_corrected_MFr_suc_coh{:});
MBc_weighted_suc_cat = vertcat(mean_corrected_MBc_suc_coh{:});
MBr_weighted_suc_cat = vertcat(mean_corrected_MBr_suc_coh{:});
bias_weighted_suc_cat = vertcat(mean_corrected_bias_suc_coh{:});

%% plot Fig 1B-D

figure('Position', [500 200 420 400]);

subplot(2,2,1)
shadedErrorBar(1:size(gammas_s1,2),mean(gammas_s1),std(gammas_s1)/sqrt(length(gammas_s1)),...
    'lineProps',{'-','Color', 'r', 'LineWidth', 1.5})
hold on;
shadedErrorBar(1:size(gammas_s1,2),mean(gammas_s2),std(gammas_s2)/sqrt(length(gammas_s2)),...
    'lineProps',{'-','Color', 'b', 'LineWidth', 1.5})
shadedErrorBar(1:size(gammas_s1,2),mean(gammas_s3),std(gammas_s3)/sqrt(length(gammas_s3)),...
    'lineProps',{'-','Color', 'k', 'LineWidth', 1.5})
xticks([0 5 9])
xticklabels({'0','0.5','1'})
xlabel('session time (normalized)','fontsize',12)
ylabel('state probability','fontsize',12)
xlim([0 size(gammas_s1,2)+1])
ylim([.28 .42])
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

subplot(2,2,2)
means = [mean(mean(gammas_s1,2)), mean(mean(gammas_s2,2)), mean(mean(gammas_s3,2))];

b = bar(means,...
    'LineWidth', 1);
hold on;
plot([1 2 3],[mean(gammas_s1,2) mean(gammas_s2,2) mean(gammas_s3,2)],'color',[.7 .7 .7])
b(1).EdgeColor = 'k';
b(1).FaceColor = 'k';
xlabel('states')
ylabel('state probability','fontsize',12)
%ylim([.25 .4])
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

ax = subplot(2,2,3);
trans_prob_sorted = (trans_prob_sorted_fent + trans_prob_sorted_suc)/2;

heatmap_lim = [.22 .45];

imagesc(mean(trans_prob_sorted,3), heatmap_lim)
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

subplot(2,2,4)
means = [mean((MBr_fent_cat+MBr_suc_cat)/2); mean((MFr_fent_cat+MFr_suc_cat)/2); mean((MFc_fent_cat+MFc_suc_cat)/2); ...
    mean((bias_fent_cat+bias_suc_cat)/2); mean((MBc_fent_cat+MBc_suc_cat)/2)];
sems = [std((MBr_fent_cat+MBr_suc_cat)/2)/sqrt(35); std((MFr_fent_cat+MFr_suc_cat)/2)/sqrt(35);std((MFc_fent_cat+MFc_suc_cat)/2)/sqrt(35);...
    std((bias_fent_cat+bias_suc_cat)/2)/sqrt(35);std((MBc_fent_cat+MBc_suc_cat)/2)/sqrt(35)];
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
ylim([-3 5])
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

%% plot Fig S1

ex_i = 1;
ex_j = 2;
ex_k = 2;
trials = 1:30;

example_id = 2;

sorted_example = gammas_per_sess_suc{ex_i}{ex_j}{ex_k}(trials,[sorted_states_suc{ex_i}{ex_j}(ex_k,:)]);

[mean(sorted_example(trials,1)), mean(sorted_example(trials,2)), mean(sorted_example(trials,3))]

figure('Position', [500 500 700 200]);
subplot(1,3,1)
plot(sorted_example(:,1),'r','linewidth',1.5)
hold on;
plot(sorted_example(:,2),'b','linewidth',1.5)
plot(sorted_example(:,3),'k','linewidth',1.5)
title('example session','fontsize',12)
xlabel('trials','fontsize',12)
ylabel('state probability','fontsize',12)
ax = gca;
ax.FontSize = 10; 

ax = subplot(1,3,2);
imagesc(trans_prob_sorted_suc(:,:,example_id))
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

subplot(1,3,3)

means = [MBr_suc_cat(example_id,:); MFr_suc_cat(example_id,:); MFc_suc_cat(example_id,:); bias_suc_cat(28,:); MBc_suc_cat(example_id,:)];
sems = [0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0];
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
%ylim([-3 5])
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

%% plot Fig S2

MB_fent_male = MBr_weighted_fent_cat(sex_cat == 1,:);
MB_suc_male = MBr_weighted_suc_cat(sex_cat == 1,:);
MB_fent_female = MBr_weighted_fent_cat(sex_cat == 0,:);
MB_suc_female = MBr_weighted_suc_cat(sex_cat == 0,:);

y_lim = [-5.1 4];

figure('Position', [500 200 420 600]);
subplot(3,2,1)
plot(MB_fent_male','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(MB_fent_male(:,1)) mean(MB_fent_male(:,1))],'color','r','linewidth',3)
plot([1.8 2.2],[mean(MB_fent_male(:,2)) mean(MB_fent_male(:,2))],'color','b','linewidth',3)
plot([2.8 3.2],[mean(MB_fent_male(:,3)) mean(MB_fent_male(:,3))],'color','k','linewidth',3)
xlabel('states','fontsize',12)
xlim([0 4])
xticks(1:3)
ylabel('MB weight')
title('fentanyl sessions (male)')
ax = gca;
ax.FontSize = 10; 
ylim(y_lim)

subplot(3,2,2)
plot(MB_suc_male','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(MB_suc_male(:,1)) mean(MB_suc_male(:,1))],'color','r','linewidth',3)
plot([1.8 2.2],[mean(MB_suc_male(:,2)) mean(MB_suc_male(:,2))],'color','b','linewidth',3)
plot([2.8 3.2],[mean(MB_suc_male(:,3)) mean(MB_suc_male(:,3))],'color','k','linewidth',3)
xlabel('states','fontsize',12)
xlim([0 4])
xticks(1:3)
ylabel('MB weight')
title('sucrose sessions (male)')
ax = gca;
ax.FontSize = 10; 
ylim(y_lim)

subplot(3,2,3)
plot(MB_fent_female','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(MB_fent_female(:,1)) mean(MB_fent_female(:,1))],'color','r','linewidth',3)
plot([1.8 2.2],[mean(MB_fent_female(:,2)) mean(MB_fent_female(:,2))],'color','b','linewidth',3)
plot([2.8 3.2],[mean(MB_fent_female(:,3)) mean(MB_fent_female(:,3))],'color','k','linewidth',3)
xlabel('states','fontsize',12)
xlim([0 4])
xticks(1:3)
ylabel('MB weight')
title('fentanyl sessions (female)')
ax = gca;
ax.FontSize = 10; 
ylim(y_lim)

subplot(3,2,4)
plot(MB_suc_female','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(MB_suc_female(:,1)) mean(MB_suc_female(:,1))],'color','r','linewidth',3)
plot([1.8 2.2],[mean(MB_suc_female(:,2)) mean(MB_suc_female(:,2))],'color','b','linewidth',3)
plot([2.8 3.2],[mean(MB_suc_female(:,3)) mean(MB_suc_female(:,3))],'color','k','linewidth',3)
xlabel('states','fontsize',12)
xlim([0 4])
xticks(1:3)
ylabel('MB weight')
title('sucrose sessions (female)')
ax = gca;
ax.FontSize = 10; 
ylim(y_lim)

subplot(3,2,5)
plot([MB_fent_female(:,1) MB_suc_female(:,1)]','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(MB_fent_female(:,1)) mean(MB_fent_female(:,1))],'color','r','linewidth',3)
plot([1.8 2.2],[mean(MB_suc_female(:,1)) mean(MB_suc_female(:,1))],'color','r','linewidth',3)
xlabel('session','fontsize',12)
xlim([0 3])
xticks(1:2)
xticklabels({'fentanyl','sucrose'})
xtickangle(45)
ylabel('MB weight')
title('state 1 (female)')
ax = gca;
ax.FontSize = 10; 
ylim([-3.2 3.2])

subplot(3,2,6)
plot([MB_fent_female(:,2) MB_suc_female(:,2)]','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(MB_fent_female(:,2)) mean(MB_fent_female(:,2))],'color','b','linewidth',3)
plot([1.8 2.2],[mean(MB_suc_female(:,2)) mean(MB_suc_female(:,2))],'color','b','linewidth',3)
xlabel('session','fontsize',12)
xlim([0 3])
xticks(1:2)
xticklabels({'fentanyl','sucrose'})
xtickangle(45)
ylabel('MB weight')
title('state 2 (female)')
ax = gca;
ax.FontSize = 10; 
ylim([-3.2 3.2])


