% Written by Eric Garr

%% fentanyl sessions

load gammas_3state % contains state labels and gammas
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

num_cohorts = 9;

training_rewards = {'fent','suc','fent','fent','suc','suc','suc','fent','suc'};

for i = 1:num_cohorts % loop over cohorts
    
    cohort_struct_fent{i} = ['coh',num2str(i),'_',training_rewards{i},'_train_fent'];
    
    rat_fieldnames{i} = fieldnames(allResults.(cohort_struct_fent{i}));
    
    for j = 1:length(fieldnames(allResults.(cohort_struct_fent{i}))) % loop over rats within a cohort
        
        addpath(['data-cleaned\',cohort_struct_fent{i}])
        fent_data{i}{j} = readmatrix(rat_fieldnames{i}{j});
        fent_sess_start{i}{j} = find(fent_data{i}{j}(:,4)); % session boundaries
        
        for k = 1:length(fent_sess_start{i}{j}) % loop over fentanyl sessions

            if k ~= length(fent_sess_start{i}{j})

                data_per_sess = fent_data{i}{j}(fent_sess_start{i}{j}(k):fent_sess_start{i}{j}(k+1)-1,:);

            else

                data_per_sess = fent_data{i}{j}(fent_sess_start{i}{j}(k):length(fent_data{i}{j}(:,4)),:);

            end
            
            c_curr_fent{i}{j}{k} = (data_per_sess(2:end,1))*-1+2;
            c_prev_fent{i}{j}{k} = (data_per_sess(1:end-1,1))*-1+2;
            stay_fent{i}{j}{k} = c_curr_fent{i}{j}{k} == c_prev_fent{i}{j}{k};

            outcome_fent{i}{j}{k} = data_per_sess(1:end-1,3);
            trans_fent{i}{j}{k} = data_per_sess(1:end-1,2);
            
            gammas_fent_sess{i}{j}{k} = gammas_per_sess_fent{i}{j}{k}(2:end,sorting_order_fent{i}{j});
            
        end
        
        c_curr_cat_fent{i}{j} = vertcat(c_curr_fent{i}{j}{:});
        c_prev_cat_fent{i}{j} = vertcat(c_prev_fent{i}{j}{:});
        stay_fent_cat{i}{j} = vertcat(stay_fent{i}{j}{:});

        outcome_fent_cat{i}{j} = vertcat(outcome_fent{i}{j}{:});
        trans_fent_cat{i}{j} = vertcat(trans_fent{i}{j}{:});

        gammas_cat_fent{i}{j} = vertcat(gammas_fent_sess{i}{j}{:});
        
    end
end

%% sucrose sessions

for i = 1:num_cohorts % loop over cohorts
    
    cohort_struct_suc{i} = ['coh',num2str(i),'_',training_rewards{i},'_train_suc'];
    
    rat_fieldnames{i} = fieldnames(allResults.(cohort_struct_suc{i}));
    
    for j = 1:length(fieldnames(allResults.(cohort_struct_suc{i}))) % loop over rats within a cohort
        
        addpath(['data-cleaned\',cohort_struct_suc{i}])
        suc_data{i}{j} = readmatrix(rat_fieldnames{i}{j});
        suc_sess_start{i}{j} = find(suc_data{i}{j}(:,4)); % session boundaries
        
        for k = 1:length(suc_sess_start{i}{j}) % loop over sucanyl sessions

            if k ~= length(suc_sess_start{i}{j})

                data_per_sess = suc_data{i}{j}(suc_sess_start{i}{j}(k):suc_sess_start{i}{j}(k+1)-1,:);

            else

                data_per_sess = suc_data{i}{j}(suc_sess_start{i}{j}(k):length(suc_data{i}{j}(:,4)),:);

            end
            
            c_curr_suc{i}{j}{k} = (data_per_sess(2:end,1))*-1+2;
            c_prev_suc{i}{j}{k} = (data_per_sess(1:end-1,1))*-1+2;
            stay_suc{i}{j}{k} = c_curr_suc{i}{j}{k} == c_prev_suc{i}{j}{k};

            outcome_suc{i}{j}{k} = data_per_sess(1:end-1,3);
            trans_suc{i}{j}{k} = data_per_sess(1:end-1,2);
            
            gammas_suc_sess{i}{j}{k} = gammas_per_sess_suc{i}{j}{k}(2:end,sorting_order_suc{i}{j});
            
        end
        
        c_curr_cat_suc{i}{j} = vertcat(c_curr_suc{i}{j}{:});
        c_prev_cat_suc{i}{j} = vertcat(c_prev_suc{i}{j}{:});
        stay_suc_cat{i}{j} = vertcat(stay_suc{i}{j}{:});

        outcome_suc_cat{i}{j} = vertcat(outcome_suc{i}{j}{:});
        trans_suc_cat{i}{j} = vertcat(trans_suc{i}{j}{:});

        gammas_cat_suc{i}{j} = vertcat(gammas_suc_sess{i}{j}{:});
        
    end
end

%% regress MB coder against gammas

tracker = 0;

for i = 1:num_cohorts % loop over cohorts

    for j = 1:length(fieldnames(allResults.(cohort_struct_fent{i}))) % loop over rats within a cohort

        tracker = tracker+1;

        gammas_cat{i}{j} = vertcat(gammas_cat_fent{i}{j},gammas_cat_suc{i}{j});
        stay_cat{i}{j} = vertcat(stay_fent_cat{i}{j},stay_suc_cat{i}{j});
        outcome_cat{i}{j} = vertcat(outcome_fent_cat{i}{j},outcome_suc_cat{i}{j});
        trans_cat{i}{j} = vertcat(trans_fent_cat{i}{j},trans_suc_cat{i}{j});

        % construct MB coder

        stay_RC{i}{j} = stay_cat{i}{j} & outcome_cat{i}{j} & trans_cat{i}{j}; 
        stay_OR{i}{j} = stay_cat{i}{j} & ~outcome_cat{i}{j} & ~trans_cat{i}{j}; 
        switch_RR{i}{j} = ~stay_cat{i}{j} & outcome_cat{i}{j} & ~trans_cat{i}{j};
        switch_OC{i}{j} = ~stay_cat{i}{j} & ~outcome_cat{i}{j} & trans_cat{i}{j};

        MB_vec{i}{j} = zeros(length(stay_cat{i}{j}),1);

        MB_vec{i}{j}(find(stay_RC{i}{j})) = 1;
        MB_vec{i}{j}(find(stay_OR{i}{j})) = 1;
        MB_vec{i}{j}(find(switch_RR{i}{j})) = 1;
        MB_vec{i}{j}(find(switch_OC{i}{j})) = 1;

        % logistic regression

        [b{tracker},dev, stats{tracker}] = glmfit(gammas_cat{i}{j},MB_vec{i}{j},'binomial','link','logit');

        b_s1_new(tracker) = b{tracker}(2);
        b_s2_new(tracker) = b{tracker}(3);
        b_s3_new(tracker) = b{tracker}(4);

    end
end

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

%% regression coefficients broken down by group

fent_train_b_s1 = b_s1_new(group_cat == 1);
fent_train_b_s2 = b_s2_new(group_cat == 1);
fent_train_b_s3 = b_s3_new(group_cat == 1);

suc_train_b_s1 = b_s1_new(group_cat == 0);
suc_train_b_s2 = b_s2_new(group_cat == 0);
suc_train_b_s3 = b_s3_new(group_cat == 0);

fent_train_b = horzcat(fent_train_b_s1',fent_train_b_s2',fent_train_b_s3');
suc_train_b = horzcat(suc_train_b_s1',suc_train_b_s2',suc_train_b_s3');

figure('Position', [500 200 425 200]);
subplot(1,3,1)
plot(fent_train_b','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(fent_train_b(:,1)) mean(fent_train_b(:,1))],'color','r','linewidth',3)
plot([1.8 2.2],[mean(fent_train_b(:,2)) mean(fent_train_b(:,2))],'color','b','linewidth',3)
plot([2.8 3.2],[mean(fent_train_b(:,3)) mean(fent_train_b(:,3))],'color','k','linewidth',3)
xlabel('states','fontsize',12)
xlim([0 4])
xticks(1:3)
ylabel('b coefficient')
title('fentanyl trained')
ax = gca;
ax.FontSize = 10; 
ylim([-5 3])

subplot(1,3,2)
plot(suc_train_b','color',[.7 .7 .7])
hold on;
plot([.8 1.2],[mean(suc_train_b(:,1)) mean(suc_train_b(:,1))],'color','r','linewidth',3)
plot([1.8 2.2],[mean(suc_train_b(:,2)) mean(suc_train_b(:,2))],'color','b','linewidth',3)
plot([2.8 3.2],[mean(suc_train_b(:,3)) mean(suc_train_b(:,3))],'color','k','linewidth',3)
xlabel('states','fontsize',12)
ylabel('b coefficient')
xlim([0 4])
xticks(1:3)
title('sucrose trained')
ax = gca;
ax.FontSize = 10; 
ylim([-5 3])

subplot(1,3,3)
scatter(ones(length(fent_train_b),1),fent_train_b(:,1),75,'markeredgecolor',[.7 .7 .7])
hold on;
scatter(ones(length(suc_train_b),1)*2,suc_train_b(:,1),75,'markeredgecolor',[.7 .7 .7])
plot([.8 1.2],[mean(fent_train_b(:,1)),mean(fent_train_b(:,1))],'color',[.2745 .5098 .7059],'linewidth',3)
plot([1.8 2.2],[mean(suc_train_b(:,1)),mean(suc_train_b(:,1))],'color',[.9412 .502 .502],'linewidth',3)
xlim([0 3])
ylim([-4.5 2.5])
xticks(1:2)
xticklabels({'fentanyl','sucrose'})
xlabel('training group')
ylabel('b coefficient')
title('state 1')
ax = gca;

ax.FontSize = 10; 
