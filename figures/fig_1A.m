% Written by Eric Garr (2025)

load 4LR

cohorts = 9;
max_states = 7;

training_rewards = {'fent','suc','fent','fent','suc','suc','suc','fent','suc'};

learning_rates = 4;
agent_weights = 5;
init_state_params = horzcat(0,[2:7]); 
state_trans_params = horzcat(0,(2:7).^2);
state_params = learning_rates + agent_weights +  init_state_params + state_trans_params;

for i = 1:cohorts % loop over cohorts
 
    cohort_struct_fent{i} = ['coh',num2str(i),'_',training_rewards{i},'_train_fent'];
    cohort_struct_suc{i} = ['coh',num2str(i),'_',training_rewards{i},'_train_suc'];
    
    rat_fieldnames{i} = fieldnames(allResults.(cohort_struct_fent{i}));
    
    for j = 1:length(fieldnames(allResults.(cohort_struct_fent{i}))) % loop over rats within a cohort
        
        % nll

        ll_fent{i}(j,:) = get_ll(max_states,allResults.(cohort_struct_fent{i}).(rat_fieldnames{i}{j}));
        ll_suc{i}(j,:) = get_ll(max_states,allResults.(cohort_struct_suc{i}).(rat_fieldnames{i}{j}));
        
        addpath(['data-cleaned\',cohort_struct_fent{i}])
        num_trials_fent{i}(j) = length(readmatrix(rat_fieldnames{i}{j}));
        rmpath(['data-cleaned\',cohort_struct_fent{i}])
        
        addpath(['data-cleaned\',cohort_struct_suc{i}])
        num_trials_suc{i}(j) = length(readmatrix(rat_fieldnames{i}{j}));
        rmpath(['data-cleaned\',cohort_struct_suc{i}])
        
        ll_fent_norm{i}(j,:) = exp(ll_fent{i}(j,:) / num_trials_fent{i}(j));
        ll_suc_norm{i}(j,:) = exp(ll_suc{i}(j,:) / num_trials_suc{i}(j));
        
        % compute AIC scores

        AIC_fent{i}(j,:) = -2 * ll_fent{i}(j,:) + 2*state_params;
        AIC_suc{i}(j,:) = -2 * ll_suc{i}(j,:) + 2*state_params;

        AIC_comb{i}(j,:) = mean([AIC_fent{i}(j,:);AIC_suc{i}(j,:)]);
        AIC_comb_relative{i}(j,:) = AIC_comb{i}(j,:) - AIC_comb{i}(j,1);
    end
    
    
end

%% plot AIC scores across reward types

aics_comb = vertcat(AIC_comb_relative{:});

figure('Position', [500 500 420 213]);
subplot(1,2,1)
plot(aics_comb','color',[.7 .7 .7])
hold on;
plot(mean(aics_comb),'k','linewidth',3)
xlabel('number of states')
ylabel('change in AIC')
ylim([-80 100])
xlim([0 8])
xticks(1:7)
xline(3,'--')
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';
ax.YColor = 'black';

for i = 1:6

    prop_aic_comb(i) = sum(aics_comb(:,i+1) < 0) / length(aics_comb);

end

subplot(1,2,2)
plot(prop_aic_comb*100,'k-o','MarkerFaceColor','k','linewidth',3)
xlim([0 7])
xticks(1:6)
xticklabels({'2','3','4','5','6','7'})
xlabel('number of states')
ylabel({'% improvement',' over single state'})
yline(68.57,'--')
ax = gca;
ax.FontSize = 10; 
ax.XColor = 'black';

ax.YColor = 'black';
