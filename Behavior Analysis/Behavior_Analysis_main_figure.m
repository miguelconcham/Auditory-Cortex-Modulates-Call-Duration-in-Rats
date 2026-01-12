load('Behavior DATA.mat')


ALL_CALLS_TOGETHER.Condition(ismember(ALL_CALLS_TOGETHER.Condition, '1 dB')) = {'0 dB'};
NoiseCondition2Pool = {'-40 dB','-30 dB','-20 dB','-10 dB','0 dB'};
NoiseConditionValues = -40:10:0;
properties2compare = {'PrincipalFrequencykHz', 'MeanPowerdBHz', 'CallLengths'};
TableVariableNames = { 'Date','DateExp', 'NoiseIndex', ...
    'PrincipalFrequencykHzDIF', 'MeanPowerdBHzDIF', 'CallLengthsDIF', ...
    'PrincipalFrequencykHzNOISE', 'MeanPowerdBHzNOISE', 'CallLengthsNOISE', ...
    'PrincipalFrequencykHzBASE', 'MeanPowerdBHzBASE', 'CallLengthsBASE'};
max_freq      = 30;
min_length    = 0.025;
max_interval  = 0.5;

%%

ALL_NOISE_COMPARISON = [];
properties2report = { 'Experiment','Condition','BeginTimes','EndTimes'};
properties2average  = {'PrincipalFrequencykHz', 'MeanPowerdBHz'};
properties2sum      = {'CallLengths'};


max_calling_time =  3;
for date2extract= unique(ALL_CALLS_TOGETHER.Date*10 + ALL_CALLS_TOGETHER.Experiment)'
    date = floor(date2extract/10);
    experiment = date2extract - 10*date;
    CallStats = ALL_CALLS_TOGETHER(ALL_CALLS_TOGETHER.Date == date & ALL_CALLS_TOGETHER.Experiment == experiment,:);
   
    experiment_list = unique(CallStats.Experiment);
    
    for experiment=unique(CallStats.Experiment)'
        max_freq = mean(CallStats.PrincipalFrequencykHz(CallStats.Experiment==experiment)) + 2.8*std(CallStats.PrincipalFrequencykHz(CallStats.Experiment==experiment));
        CallStats(CallStats.Experiment==experiment, :) = CallCount(CallStats(CallStats.Experiment==experiment, :),max_freq,min_length,max_interval);
    end
    CallStats = CallStats(ismember(CallStats.Condition,[ NoiseCondition2Pool,'Baseline']),:);
    
    call_number = 1;
    range2average = [1 size(CallStats,1)];
    while call_number<size(CallStats,1)
        searching = true;
        corrected = false;
        while call_number<size(CallStats,1) & searching
            if ~strcmp(CallStats.Condition(range2average(1)), CallStats.Condition(call_number+1))
                range2average(2) = call_number;
                searching = false;
                
                if CallStats.BeginTimes(range2average(2))-CallStats.BeginTimes(range2average(1)) >max_calling_time
                    disp(['Selecting last ' num2str(max_calling_time), 'seconds of calls'])
                    total_range = range2average(1):range2average(2);
                    range2average(1) = total_range(min(find(CallStats.BeginTimes(range2average(1):range2average(2))> (CallStats.EndTimes(range2average(2))-max_calling_time))));
                    corrected = true;
                    
                end
                
            else
                call_number = call_number+1;
            end
            
            
        end
        
        if range2average(2)-range2average(1)>=1
            ALL_NOISE_COMPARISON = [ALL_NOISE_COMPARISON; [date, 10*date+experiment, corrected, ...
                table2cell(CallStats(range2average(1),properties2report)),table2cell(CallStats(range2average(2),properties2report)), ...
                num2cell([nanmean(CallStats{range2average(1):range2average(2), properties2average}) nansum(CallStats{range2average(1):range2average(2), properties2sum}) range2average(2)-range2average(1)+1])]];
        else
            ALL_NOISE_COMPARISON = [ALL_NOISE_COMPARISON; [date, 10*date+experiment,corrected, ...
                table2cell(CallStats(range2average(1),properties2report)),table2cell(CallStats(range2average(2),properties2report)), ...
                num2cell([(CallStats{range2average(1):range2average(2),[ properties2average,properties2sum]}) range2average(2)-range2average(1)+1])]];
            
        end
        range2average(1:2) = call_number+1;
        
        call_number = call_number+1;
    end
    
    
end


ALL_NOISE_COMPARISON= cell2table(ALL_NOISE_COMPARISON);
ALL_NOISE_COMPARISON.Properties.VariableNames = {'Date','DateExp', 'Corrected',...
    'ExperimentFirstCall', 'ConditionFirstCall','BeginTimesFirstCall','EndTimesFirstCall' , ...
    'ExperimentLastCall', 'ConditionLastCall','BeginTimesLastCall','EndTimesLastCall', ...
    'PrincipalFrequencykHzMean', 'MeanPowerdBHzMean', 'CallLengthsSum', 'NCalls'};


%%

NoiseProperties =    {'PrincipalFrequencykHzMean', 'MeanPowerdBHzMean', 'CallLengthsSum'};
% for        date2plot =  unique(ALL_NOISE_COMPARISON.Date*10 + ALL_NOISE_COMPARISON.ExperimentFirstCall)'

%     Matrix2Compare = ALL_NOISE_COMPARISON(ALL_NOISE_COMPARISON.Date == date & ALL_NOISE_COMPARISON.ExperimentFirstCall== experiment,:);
CorrelationValues = nan(numel(unique(ALL_NOISE_COMPARISON.Date)), 12);
date_index = 1;
for date2plot = unique(ALL_NOISE_COMPARISON.Date)'
    date = date2plot;
    experiment = 0;
%     date = floor(date2plot/10);
%     experiment = date2plot - date*10;
    

    Matrix2Compare = ALL_NOISE_COMPARISON(ALL_NOISE_COMPARISON.Date == date2plot,:);
    NoiseChangeMatrix = [];
%      experiment = 
    
    for noise_type =NoiseCondition2Pool
        
        noise_list = find(ismember(Matrix2Compare.ConditionFirstCall, noise_type));
        baseline_list = noise_list+1;
        
        NoiseWoBaseline = find(~ismember(Matrix2Compare.ConditionFirstCall(baseline_list), 'Baseline'));
        noise_list(NoiseWoBaseline) = [];
        baseline_list(NoiseWoBaseline) = [];
        
       
        NoiseChangeMatrix = [NoiseChangeMatrix; [ repmat([noise_type, {date, experiment}],numel(noise_list),1), num2cell([Matrix2Compare{noise_list,NoiseProperties}-Matrix2Compare{baseline_list,NoiseProperties},Matrix2Compare{baseline_list,NoiseProperties}])]];
        
    end
    
    NoiseChangeMatrix = cell2table(NoiseChangeMatrix);
    NoiseChangeMatrix.Properties.VariableNames = [{'NoiseType', 'Date', 'Experiment'}, 'PrincipalFrequencykHzMean', 'MeanPowerdBHzMean', 'CallLengthsSum', 'BASEPrincipalFrequencykHzMean', 'BASEMeanPowerdBHzMean', 'BASECallLengthsSum'];
    
    noise_value = NoiseConditionValues(cell2mat(cellfun(@(x) find(ismember(NoiseCondition2Pool,x)),   NoiseChangeMatrix.NoiseType, 'UniformOutput', false)))';
    
    figure
    subplot(1,3,1)
    NoiseOrder = NoiseCondition2Pool(ismember(NoiseCondition2Pool,NoiseChangeMatrix.NoiseType));
    boxplot(NoiseChangeMatrix.CallLengthsSum, NoiseChangeMatrix.NoiseType ,'GroupOrder', NoiseOrder)
    
    hold on
    sub_position = cellfun(@(x) find(ismember(NoiseOrder, x)), NoiseChangeMatrix.NoiseType, 'UniformOutput', false);
    empty_sub_position = cell2mat(cellfun(@(x) numel(x), sub_position, 'UniformOutput', false));
    sub_position(empty_sub_position==0) = {nan};
    sub_position = cell2mat(sub_position);
    plot(sub_position, NoiseChangeMatrix.CallLengthsSum, 'k.', 'MarkerSize', 5)
    
    
    [c,p] = corr(noise_value ,NoiseChangeMatrix.CallLengthsSum, 'type', 'Spearman');
    CorrelationValues(date_index, [1 2]) = [c, p];
    CorrelationValues(date_index, 7) = nanmean(NoiseChangeMatrix.CallLengthsSum);
    [~, p] = ttest(NoiseChangeMatrix.CallLengthsSum(noise_value>-40));
    CorrelationValues(date_index, 10) = p;
    title(['c =',num2str(round(c,2)), ' p =',num2str(round(log10(p),2))])
    ylabel('Length')
    xtickangle(90)
    
    subplot(1,3,2)
    NoiseOrder = NoiseCondition2Pool(ismember(NoiseCondition2Pool,NoiseChangeMatrix.NoiseType));
    boxplot(NoiseChangeMatrix.PrincipalFrequencykHzMean, NoiseChangeMatrix.NoiseType ,'GroupOrder', NoiseOrder)
    
    hold on
    sub_position = cellfun(@(x) find(ismember(NoiseOrder, x)), NoiseChangeMatrix.NoiseType, 'UniformOutput', false);
    empty_sub_position = cell2mat(cellfun(@(x) numel(x), sub_position, 'UniformOutput', false));
    sub_position(empty_sub_position==0) = {nan};
    sub_position = cell2mat(sub_position);
    plot(sub_position,NoiseChangeMatrix.PrincipalFrequencykHzMean, 'k.', 'MarkerSize', 5)
    
    [c,p] = corr(noise_value ,NoiseChangeMatrix.PrincipalFrequencykHzMean, 'type', 'Spearman');
    CorrelationValues(date_index, [3 4]) = [c, p];
    CorrelationValues(date_index, 8) = nanmean(NoiseChangeMatrix.PrincipalFrequencykHzMean);
    [~, p] = ttest(NoiseChangeMatrix.PrincipalFrequencykHzMean(noise_value>-40));
    CorrelationValues(date_index, 11) = p;
    title(['c =',num2str(round(c,2)), ' p =',num2str(round(log10(p),2))])
    ylabel('Freq')
    xtickangle(90)
    
    subplot(1,3,3)
    
    NoiseOrder = NoiseCondition2Pool(ismember(NoiseCondition2Pool,NoiseChangeMatrix.NoiseType)) ;
    boxplot(NoiseChangeMatrix.MeanPowerdBHzMean, NoiseChangeMatrix.NoiseType ,'GroupOrder',NoiseOrder)
    
    
    hold on
    sub_position = cellfun(@(x) find(ismember(NoiseOrder, x)), NoiseChangeMatrix.NoiseType, 'UniformOutput', false);
    empty_sub_position = cell2mat(cellfun(@(x) numel(x), sub_position, 'UniformOutput', false));
    sub_position(empty_sub_position==0) = {nan};
    sub_position = cell2mat(sub_position);
    plot(sub_position, NoiseChangeMatrix.MeanPowerdBHzMean, 'k.', 'MarkerSize', 5)
    
    
    [c,p] = corr(noise_value ,NoiseChangeMatrix.MeanPowerdBHzMean, 'type', 'Spearman');
    CorrelationValues(date_index, [5 6]) = [c, p];    
    CorrelationValues(date_index, 9) = nanmean(NoiseChangeMatrix.MeanPowerdBHzMean);
    [~, p] = ttest(NoiseChangeMatrix.MeanPowerdBHzMean(noise_value>-40));
    CorrelationValues(date_index, 12) = p;
    title(['c =',num2str(round(c,2)), ' p =',num2str(round(log10(p),2))])
    ylabel('Amplitud')
    xtickangle(90)
    date_index = date_index+1;
    % suptitle(num2str(date2plot))
    
    
    
    figure('units','normalized','outerposition', [0 0 1 1]);
    data = [NoiseChangeMatrix.BASECallLengthsSum, NoiseChangeMatrix.BASECallLengthsSum+NoiseChangeMatrix.CallLengthsSum];
    y_lim = [.9*min(data(:)) 1.1*max(data(:))];
    sb_n=1;
    for noise_type= NoiseCondition2Pool
        subplot(3,numel(NoiseCondition2Pool), sb_n)
        index = ismember(NoiseChangeMatrix.NoiseType,noise_type);
        if sum(index)>0
            
            hold on
            er = errorbar([1 2],mean(data(index,:)),-std(data(index,:)),std(data(index,:)), 'LineWidth',3, 'Color', 'b');
            
            hold on
            plot([1 2],data(index,:), 'k', 'Linewidth', .1);
            
            [h, ~, ~]= kstest(data(index, 2)-data(index, 1));
            if h
                [~, p, ~] =ttest(data(index, 2)-data(index, 1))
            else
                [p, ~, ~] =signrank(data(index, 2)-data(index, 1))
            end            
            
        end
        xlim([.5 2.5])
        xticks([1 2])
        xticklabels([])
        ylim(y_lim)
        if strcmp(noise_type, '0 dB')
            ylabel('Call Length')
        end
        title({noise_type{1}, num2str(p)})
        sb_n=sb_n+1;
        
    end
    data = [NoiseChangeMatrix.BASEPrincipalFrequencykHzMean, NoiseChangeMatrix.PrincipalFrequencykHzMean+NoiseChangeMatrix.BASEPrincipalFrequencykHzMean];
    y_lim = [.9*min(data(:)) 1.1*max(data(:))];
    
    for noise_type= NoiseCondition2Pool
        subplot(3,numel(NoiseCondition2Pool), sb_n)
        index = ismember(NoiseChangeMatrix.NoiseType,noise_type);
        if sum(index)>0
            
            hold on
            er = errorbar([1 2],mean(data(index,:)),-std(data(index,:)),std(data(index,:)), 'LineWidth',3, 'Color', 'b');
            
            hold on
            plot([1 2],data(index,:), 'k', 'Linewidth', .1);
            [h, ~, ~]= kstest(data(index, 2)-data(index, 1));
            if h
                [~, p, ~] =ttest(data(index, 2)-data(index, 1))
            else
                [p, ~, ~] =signrank(data(index, 2)-data(index, 1))
            end        
        end
        xlim([.5 2.5])
        xticks([1 2])
        xticklabels([])
        ylim(y_lim)
        if strcmp(noise_type, '0 dB')
            ylabel('Frequency')
        end
        title( num2str(p))
        sb_n=sb_n+1;
        
    end
    data = [NoiseChangeMatrix.BASEMeanPowerdBHzMean, NoiseChangeMatrix.MeanPowerdBHzMean+NoiseChangeMatrix.BASEMeanPowerdBHzMean];
    y_lim = [min(data(:)) max(data(:))];
    
    for noise_type= NoiseCondition2Pool
        subplot(3,numel(NoiseCondition2Pool), sb_n)
        index = ismember(NoiseChangeMatrix.NoiseType,noise_type);
        if sum(index)>0
            hold on
            er = errorbar([1 2],mean(data(index,:)),-std(data(index,:)),std(data(index,:)), 'LineWidth',3, 'Color', 'b');
            
            hold on
            plot([1 2],data(index,:), 'k', 'Linewidth', .1);
            [h, ~, ~]= kstest(data(index, 2)-data(index, 1));
            if h
                [~, p, ~] =ttest(data(index, 2)-data(index, 1))
            else
                [p, ~, ~] =signrank(data(index, 2)-data(index, 1))
            end        
        end
        xlim([.5 2.5])
        xticks([1 2])
        xticklabels({'Baseline', 'Noise'})
        xtickangle(45)
        if strcmp(noise_type, '0 dB')
            ylabel('Amplitud')
        end
        ylim(y_lim)
        title( num2str(p))
        sb_n=sb_n+1;
        
    end
    % suptitle(num2str(date2plot))
    
end



%% Correlation plot 
order4plot = [5 3];
rand_disp = (rand(size(CorrelationValues,1),numel(order4plot))-.5)*.2;
figure('units','normalized','outerposition', [0 0 1 1]);
x_tick_labels = {'PrincipalFrequency','CallLengths', 'MeanPow'};


subplot(1,2,1)
boxplot(CorrelationValues(:, order4plot))
hold on
plot(ones(size(CorrelationValues,1),1)*(1:numel(order4plot)) +rand_disp, CorrelationValues(:, order4plot), 'k.', 'MarkerSize', 20)
plot((ones(size(CorrelationValues,1),1)*(1:numel(order4plot)) +rand_disp)', (CorrelationValues(:, order4plot))', 'k:', 'LineWidth', 1)

% plot(ones(3,1)*[1 2 3], CorrelationValues(8:10, order4plot), 'r.', 'MarkerSize', 20)

for j=1:numel(order4plot)
    index_01 = CorrelationValues(:,order4plot(j)+1)<=0.01;
index_05 = CorrelationValues(:,order4plot(j)+1)<=0.05;
plot(ones(sum(index_01),1)*j + rand_disp(index_01,j), CorrelationValues(index_01, order4plot(j)), '.', 'MarkerSize', 25, 'Color', [1 0 0] )
plot(ones(sum(index_05 & ~index_01),1)*j + rand_disp(index_05 & ~index_01,j), CorrelationValues(index_05 & ~index_01, order4plot(j)), '.', 'MarkerSize', 25, 'Color', [1 .65 0] )

end



xticklabels(x_tick_labels(floor(order4plot/2)+1))
xtickangle(45)
ylabel('Spearman Correlation')
set(gca, 'FontSize', 24)

subplot(1,2,2)
boxplot(log10(CorrelationValues(:, order4plot+1)))
hold on
plot(ones(size(CorrelationValues,1),1)*(1:numel(order4plot)) + rand_disp, log10(CorrelationValues(:, order4plot+1)), 'k.', 'MarkerSize', 20)
% plot((ones(size(CorrelationValues,1),1)*(1:numel(order4plot)) + rand_disp)', log10(CorrelationValues(:, [4 2 6]))', 'k:', 'LineWidth', 1)
% plot(ones(3,1)*(1:numel(order4plot)), log10(CorrelationValues(8:10, [4 2 6])), 'r.', 'MarkerSize', 20)
plot([.5 3.5], log10([.05 0.05]), ':k', 'LineWidth', 2, 'Color', [1 .65 0] )
plot([.5 3.5], log10([.01 0.01]), ':k', 'LineWidth', 3, 'Color', [1 0 0] )
xticklabels(x_tick_labels(floor(order4plot/2)+1))
xtickangle(45)
ylabel('P-Value')
set(gca, 'FontSize', 24)
%% Amplitud plot
figure
subplot(1,2,1)
plot(CorrelationValues(:, 3),CorrelationValues(:, 9), 'k.', 'MarkerSize', 20)
hold on
plot(CorrelationValues(CorrelationValues(:,6)<=0.01, 3),CorrelationValues(CorrelationValues(:,6)<=0.01, 9), 'r.', 'MarkerSize', 20)


subplot(1,2,2)
plot(CorrelationValues(:, 3),CorrelationValues(:, 9), 'k.', 'MarkerSize', 20)
hold on
plot(CorrelationValues(CorrelationValues(:,12)<=0.01, 3),CorrelationValues(CorrelationValues(:,12)<=0.01, 9), 'r.', 'MarkerSize', 20)


%% Mean noise effect per animal

mean_values_CALL_TRAIN = [];
mean_values_figure = figure;

for date2plot = unique(ALL_NOISE_COMPARISON.Date)'
    date = date2plot;
    experiment = 0;
%     date = floor(date2plot/10);
%     experiment = date2plot - date*10;
    

    Matrix2Compare = ALL_NOISE_COMPARISON(ALL_NOISE_COMPARISON.Date == date2plot,:);
    
    basline_cond    = ismember(Matrix2Compare.ConditionFirstCall, 'Baseline');
    noise_cond      = ~ ismember(Matrix2Compare.ConditionFirstCall, 'Baseline');
    
    mean_values_CALL_TRAIN = [mean_values_CALL_TRAIN;[ date2plot, nanmean(Matrix2Compare.CallLengthsSum(basline_cond)),nanmean(Matrix2Compare.CallLengthsSum(noise_cond)), ...
                   nanmean(Matrix2Compare.PrincipalFrequencykHzMean(basline_cond)),nanmean(Matrix2Compare.PrincipalFrequencykHzMean(noise_cond)), ...
                   nanmean(Matrix2Compare.MeanPowerdBHzMean(basline_cond)),nanmean(Matrix2Compare.MeanPowerdBHzMean(noise_cond))]]
end

random_values_mean = .2*(rand(size(mean_values_CALL_TRAIN,1), 2) - .5) *0;
  
    subplot(1,3,1)
%     boxplot(mean_values_CALL_TRAIN(:, 2:3))
     hold on
    plot(repmat([1 2],size(mean_values_CALL_TRAIN,1),1)' + random_values_mean',mean_values_CALL_TRAIN(:, 2:3)', ':k')   
    plot(repmat([1 2],size(mean_values_CALL_TRAIN,1),1) + random_values_mean,mean_values_CALL_TRAIN(:, 2:3), '.k', 'MarkerSize', 20)
   errorbar([1 2],nanmean(mean_values_CALL_TRAIN(:, 2:3)), nanstd(mean_values_CALL_TRAIN(:, 2:3)))

    
    xlim([.5 2.5])
     xticks([1 2])
     ylim([.15 1.5])
     title('Total Call Length')
    xticklabels({'Base', 'Noise'})
    
     subplot(1,3,2)
%       boxplot(mean_values_CALL_TRAIN(:, 4:5))
     hold on
    plot(repmat([1 2],size(mean_values_CALL_TRAIN,1),1)' + random_values_mean',mean_values_CALL_TRAIN(:, 4:5)', ':k')
    hold on
    plot(repmat([1 2],size(mean_values_CALL_TRAIN,1),1) + random_values_mean,mean_values_CALL_TRAIN(:, 4:5), '.k', 'MarkerSize', 20)
    errorbar([1 2],nanmean(mean_values_CALL_TRAIN(:, 4:5)), nanstd(mean_values_CALL_TRAIN(:, 4:5)))

    xticks([1 2])
    xlim([.5 2.5])
    ylim([22 34])
    title('Freq')
    xticklabels({'Base', 'Noise'})

    subplot(1,3,3)
%       boxplot(mean_values_CALL_TRAIN(:, 4:5))
     hold on
    plot(repmat([1 2],size(mean_values_CALL_TRAIN,1),1)' + random_values_mean',mean_values_CALL_TRAIN(:, 6:7)', ':k')
    hold on
    plot(repmat([1 2],size(mean_values_CALL_TRAIN,1),1) + random_values_mean,mean_values_CALL_TRAIN(:, 6:7), '.k', 'MarkerSize', 20)
    errorbar([1 2],nanmean(mean_values_CALL_TRAIN(:, 6:7)), nanstd(mean_values_CALL_TRAIN(:, 6:7)))

    xticks([1 2])
    xlim([.5 2.5])
    % ylim([22 34])
    title('Amplitud')
    xticklabels({'Base', 'Noise'})

%% Call number Plots
noise_level_order=  {'0 dB','-10 dB','-20 dB','-30 dB','-40 dB'};    
Date_combinations = unique(ALL_NOISE_COMPARISON.Date);
summary_values = [];
for date_n = 1:numel(Date_combinations)
experiment_index = ALL_NOISE_COMPARISON.Date == Date_combinations(date_n);

SUB_COMP = ALL_NOISE_COMPARISON(experiment_index,:);
noise_levels = unique(SUB_COMP.ConditionFirstCall)';
noise_levels(strcmp(noise_levels, 'Baseline')) = [];

noise_levels = noise_level_order(ismember(noise_level_order,noise_levels));

figure('units','normalized','outerposition',[0 0 1 1]);
all_noise_comp = [];

for nl =1:numel(noise_levels)
    noise_level_indexes = find(ismember(SUB_COMP.ConditionFirstCall, noise_levels(nl)));

    noise_comp = nan(numel(noise_level_indexes),3);
    rand_pos = .1*(rand(numel(noise_level_indexes),1)-.5);
    for n=1:numel(noise_level_indexes)

        noise_ref_index = noise_level_indexes(n)-1;
        if noise_ref_index==0
            noise_ref_index = noise_ref_index+2;
        end
        if  SUB_COMP.NCalls(noise_ref_index)==1
            noise_comp(n,:) = [NaN NaN NaN];
        else
            noise_comp(n,:) = [SUB_COMP.NCalls(noise_level_indexes(n))   SUB_COMP.NCalls(noise_ref_index)  nl];
        end
        plot([0 .5] + rand_pos(n)+nl,noise_comp(n,[2 1]) , ':k' )
        hold on
        plot([0 .5] + rand_pos(n)+nl,noise_comp(n,[2 1]) , '.', 'MarkerSize',6 )
    end
    plot([0 .5]+ nl, max(max(noise_comp(:, [2 1])))*[1.1 1.1], 'k')
    p = signrank(noise_comp(:, 2),noise_comp(:, 1));
    text(.1 + nl, max(max(noise_comp(:, [2 1])))*1.1 + .05, num2str(round(p,3)))
    all_noise_comp = [all_noise_comp;[noise_comp]];
 
end
   ylim([0 (max(max(all_noise_comp(:, [1 2])))*1.1 + .1)])
   no_nan_index = ~any(isnan(all_noise_comp(:, [1 2])),2);
[c,p] = corr(all_noise_comp(no_nan_index,2)-all_noise_comp(no_nan_index,1), all_noise_comp(no_nan_index,3), 'type','Spearman');
title(['C = ', num2str(c), ' P = ', num2str(p)])
pause(.1)

summary_values = [summary_values;[nanmean(all_noise_comp(all_noise_comp(:,3)<5,2)) nanmean(all_noise_comp(all_noise_comp(:,3)<5,1)) c p]];
end
%%
rand_values = .1*(rand(size(summary_values,1),1)-.5);
figure
subplot(1,2,1)
plot([1 2]' + [rand_values rand_values]', summary_values(:,[1 2])', ':k')
hold on
plot([1 2]' + [rand_values rand_values]', summary_values(:,[1 2])', '.k', 'MarkerSize', 6)
xlim([.5 2.5])
[~, p,~, t_stat] = ttest(summary_values(:,1),summary_values(:,2));
title([' P = ', num2str(p), ' t_val =', num2str(t_stat.tstat), ' d.f=', num2str(t_stat.df)])

subplot(1,2,2)

plot(rand_values , summary_values(:,3), '.k', 'MarkerSize', 6)
hold on
plot(rand_values(summary_values(:,4)<=0.01) , summary_values(summary_values(:,4)<=0.01,3), '.r', 'MarkerSize', 6)
plot([-.1 .1], [1 1]*mean(summary_values(:,3)), 'r')
xlim([-.5 .5])
[h,p]=ttest(summary_values(:,3)






    %%
    
    figure
      boxplot(mean_values_CALL_TRAIN(:, 6:7))
     hold on
    plot(repmat([1 2],size(mean_values_CALL_TRAIN,1),1)' + random_values_mean',mean_values_CALL_TRAIN(:, 6:7)', ':k')
    hold on
    plot(repmat([1 2],size(mean_values_CALL_TRAIN,1),1) + random_values_mean,mean_values_CALL_TRAIN(:, 6:7), '.k', 'MarkerSize', 20)
   
    xticks([1 2])
    [H,P,CI,STATS] = ttest(mean_values_CALL_TRAIN(:, 6),mean_values_CALL_TRAIN(:, 7))
    xlim([.5 2.5])
    title(['AMplitud t = ', num2str(STATS.tstat), ' p = ', num2str(P)])
    xticklabels({'Base', 'Noise'})
%% Model plot? XD
ModelSummary = nan(numel(unique(ALL_NOISE_COMPARISON.Date)),4);
% ALL_COMPARISON.NoiseIndex(ALL_COMPARISON.Date==20221026 & ALL_COMPARISON.NoiseIndex==1) = NaN;

data_index=1;
for  date2plot= unique(ALL_NOISE_COMPARISON.Date)'
    
    ComparisonMatrix = ALL_NOISE_COMPARISON(ALL_NOISE_COMPARISON.Date == date2plot & ~isnan(ALL_NOISE_COMPARISON.NoiseIndex),:);
    [c, p] = corr(ComparisonMatrix.NoiseIndex, ComparisonMatrix.CallLengthsDIF, 'type', 'Pearson');
    ModelSummary(data_index, 1) = c;
    ModelSummary(data_index, 2) = p;
    [c, p] = corr(ComparisonMatrix.NoiseIndex, ComparisonMatrix.PrincipalFrequencykHzDIF, 'type', 'Pearson');
    ModelSummary(data_index, 3) = c;
    ModelSummary(data_index, 4) = p;
    
    data_index=data_index+1;
    
    
end

figure('units','normalized','outerposition', [0 0 1 1]);
subplot(1,2,1)
boxplot(ModelSummary(:, [1 3]))
hold on
plot(ones(size(ModelSummary,1),1)*[1 2], ModelSummary(:, [1 3]), 'k.', 'MarkerSize', 20)
xticklabels({'CallLengths', 'PrincipalFrequency'})
xtickangle(45)
ylabel('Spearman Correlation')
set(gca, 'FontSize', 24)


subplot(1,2,2)
boxplot(log10(ModelSummary(:, [2 4])))
hold on
plot([.5 2.5], log10([.05 0.05]), ':k', 'LineWidth', 2)
plot([.5 2.5], log10([.01 0.01]), ':k', 'LineWidth', 3)
hold on

plot(ones(size(ModelSummary,1),1)*[1 2], log10(ModelSummary(:, [2 4])),  'k.', 'MarkerSize', 20)
xticklabels({'CallLengths', 'PrincipalFrequency'})
xtickangle(45)
ylabel('P-Value')
set(gca, 'FontSize', 24)

% suptitle('Call by Call correlation values')
%%%%%%%%%%%%%%%%%%%
%% NOW CALL BY CALL
%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%


ALL_NOISE_COMPARISON_callbycall = [];
% max_freq = Inf;

for date= unique(ALL_CALLS_TOGETHER.Date)'
    
    CallStats = ALL_CALLS_TOGETHER(ALL_CALLS_TOGETHER.Date == date,:);
    
    experiment_list = unique(CallStats.Experiment);
    
    for experiment=unique(CallStats.Experiment)'
        max_freq = mean(CallStats.PrincipalFrequencykHz(CallStats.Experiment==experiment)) + 4*std(CallStats.PrincipalFrequencykHz(CallStats.Experiment==experiment));
        CallStats(CallStats.Experiment==experiment, :) = CallCount(CallStats(CallStats.Experiment==experiment, :),max_freq,min_length,max_interval);
    end
    
    
    BaselineCalls       = find(ismember(CallStats.Condition, 'Baseline'));
    ComparisonMatrix    = [];
    
    for noise_index     = 1:numel(NoiseCondition2Pool)
        noise_list      = find(ismember(CallStats.Condition, NoiseCondition2Pool(noise_index)) & CallStats.CallNumber>0);
        
        
        
        
        for noise_number    = 1:numel(noise_list)
            
            next_baseline_call = min(find(CallStats.BeginTimes(BaselineCalls)>CallStats.BeginTimes(noise_list(noise_number)) & CallStats.CallNumber(BaselineCalls)==CallStats.CallNumber(noise_list(noise_number))));
            
            if ~isempty(next_baseline_call)
                
                ComparisonMatrix = [ComparisonMatrix;[date,date*10 , noise_index ...
                    ,CallStats{noise_list(noise_number),properties2compare}-CallStats{BaselineCalls(next_baseline_call),properties2compare} ...
                    ,CallStats{noise_list(noise_number),properties2compare} ...                              %% noise properties
                    ,CallStats{BaselineCalls(next_baseline_call),properties2compare}  ]];                     %% baseline for freq properties
            end
            
        end
    end
    
    
    ComparisonMatrix = array2table(ComparisonMatrix);
    ComparisonMatrix.Properties.VariableNames = TableVariableNames;
    
    
    
    ALL_NOISE_COMPARISON_callbycall = [ALL_NOISE_COMPARISON_callbycall; ComparisonMatrix];
    
    
    
end

%% Frequency plot BOXES
date2plot =  20221007;
ComparisonMatrix = ALL_NOISE_COMPARISON_callbycall(ALL_NOISE_COMPARISON_callbycall.Date == date2plot,:);
for noise_index = 1:5
    figure
    x0=10;
    y0=10;
    width=650;
    height=550
    set(gcf,'position',[x0,y0,width,height])
    plot(ComparisonMatrix.PrincipalFrequencykHzNOISE(ComparisonMatrix.NoiseIndex==noise_index),ComparisonMatrix.PrincipalFrequencykHzBASE(ComparisonMatrix.NoiseIndex==noise_index), 'k.')
    % minimo = min(ComparisonMatrix.PrincipalFrequencykHzNOISE(ComparisonMatrix.NoiseIndex==noise_index));
    % minimo = min(minimo, min(ComparisonMatrix.PrincipalFrequencykHzBASE(ComparisonMatrix.NoiseIndex==noise_index)));
    %
    % maximo = max(ComparisonMatrix.PrincipalFrequencykHzNOISE(ComparisonMatrix.NoiseIndex==noise_index));
    % maximo = max(maximo, max(ComparisonMatrix.PrincipalFrequencykHzBASE(ComparisonMatrix.NoiseIndex==noise_index)));
    
    hold on
    plot([24 32], [24 32], 'k')
    axis([24 32 24 32])
    axis tight
    
    plot(mean(ComparisonMatrix.PrincipalFrequencykHzNOISE(ComparisonMatrix.NoiseIndex==noise_index)), mean(ComparisonMatrix.PrincipalFrequencykHzBASE(ComparisonMatrix.NoiseIndex==noise_index)), 'rx', 'MarkerSize', 25)
    xlabel('Frequency During Noise (kHz)')
    ylabel('Frequency During Baseline (kHz)')
    set(gca, 'FontSize', 14)
    title(NoiseCondition2Pool(noise_index))
    saveas(gcf, [num2str(date2plot), ' Call Frequency difference noise level ', NoiseCondition2Pool{noise_index}, '.svg'])
    
end

%% Call length plot BOXES

for noise_index = 1:5
    figure
    x0=10;
    y0=10;
    width=650;
    height=550
    set(gcf,'position',[x0,y0,width,height])
    plot(ComparisonMatrix.CallLengthsNOISE(ComparisonMatrix.NoiseIndex==noise_index),ComparisonMatrix.CallLengthsBASE(ComparisonMatrix.NoiseIndex==noise_index), 'k.')
    
    
    hold on
    plot([0 .6], [0 .6], 'k')
    axis tight
    
    plot(mean(ComparisonMatrix.CallLengthsNOISE(ComparisonMatrix.NoiseIndex==noise_index)), mean(ComparisonMatrix.CallLengthsBASE(ComparisonMatrix.NoiseIndex==noise_index)), 'rx', 'MarkerSize', 25)
    xlabel('Length During Noise (kHz)')
    ylabel('Length During Baseline (kHz)')
    set(gca, 'FontSize', 14)
    title(NoiseCondition2Pool(noise_index))
    % saveas(gcf, [num2str(date2plot), ' Call length difference noise level ', NoiseCondition2Pool{noise_index}, '.svg'])
    
end
%% Frequency example plot

ComparisonMatrix = ALL_NOISE_COMPARISON_callbycall(ALL_NOISE_COMPARISON_callbycall.Date == date2plot,:);
figure('units','normalized','outerposition', [0 0 1 1]);
data2violin = cell(5,1);
for noise_index = 1:5
    subplot(1,5,noise_index)
    data = [ComparisonMatrix.PrincipalFrequencykHzBASE(ComparisonMatrix.NoiseIndex==noise_index) ComparisonMatrix.PrincipalFrequencykHzNOISE(ComparisonMatrix.NoiseIndex==noise_index)];
    %    b = bar(mean(data))
    %    b.FaceAlpha = 0.25;
    %    b.FaceColor = [0 0 0];
    if ~isempty(data)
        
        hold on
        er = errorbar([1 2],mean(data),-std(data),std(data), 'LineWidth',3);
        
        hold on
        plot([1 2],data, 'k', 'Linewidth', .1);
        ylim([24 32])
        xlim([.5 2.5])
        if noise_index==1
            ylabel('Frequency')
        end
        xticks([1 2])
        xticklabels({'Baseline','Noise'})
        xtickangle(45)
        set(gca, 'FontSize', 24)
        %    [p, h, stats] =signrank(ComparisonMatrix.PrincipalFrequencykHzDIF((ComparisonMatrix.NoiseIndex==noise_index)))
        [h, p, stat]= kstest(ComparisonMatrix.PrincipalFrequencykHzDIF((ComparisonMatrix.NoiseIndex==noise_index)))
        if h
            [h, p, stats] =ttest(ComparisonMatrix.PrincipalFrequencykHzDIF((ComparisonMatrix.NoiseIndex==noise_index)))
        else
            [p, h, stats] =signrank(ComparisonMatrix.PrincipalFrequencykHzDIF((ComparisonMatrix.NoiseIndex==noise_index)))
        end
        
        
        title(p)
    end
    
end
% suptitle(num2str(date2plot))
saveas(gcf, [num2str(date2plot), ' Call Frequency difference ALL NOISE LEVLES together.svg'])


figure

NoiseOrder = NoiseCondition2Pool(unique(ComparisonMatrix.NoiseIndex));
boxplot(ComparisonMatrix.PrincipalFrequencykHzDIF, NoiseCondition2Pool(ComparisonMatrix.NoiseIndex)' ,'GroupOrder', NoiseOrder)

hold on
sub_position = ComparisonMatrix.NoiseIndex;

plot(sub_position, ComparisonMatrix.PrincipalFrequencykHzDIF, 'k.', 'MarkerSize', 5)


[c,p] = corr(noise_value ,NoiseChangeMatrix.CallLengthsSum, 'type', 'Spearman');
CorrelationValues(date_index, [1 2]) = [c, p];
CorrelationValues(date_index, 7) = nanmean(NoiseChangeMatrix.CallLengthsSum);
[~, p] = ttest(NoiseChangeMatrix.CallLengthsSum(noise_value>-40));
CorrelationValues(date_index, 10) = p;
title(['c =',num2str(round(c,2)), ' p =',num2str(round(log10(p),2))])
ylabel('Principal Freq (kHz)')
xtickangle(90)



%% call length example plot

figure('units','normalized','outerposition', [0 0 1 1]);
data2violin = cell(5,1);
for noise_index = 1:5
    subplot(1,5,noise_index)
    
    if ~isempty(data)
        data = [ComparisonMatrix.CallLengthsNOISE(ComparisonMatrix.NoiseIndex==noise_index) ComparisonMatrix.CallLengthsBASE(ComparisonMatrix.NoiseIndex==noise_index)];
        %    b = bar(mean(data))
        %    b.FaceAlpha = 0.25;
        %    b.FaceColor = [0 0 0];
        hold on
        er = errorbar([1 2],mean(data),-std(data),std(data), 'LineWidth',3);
        
        hold on
        plot([1 2],data, 'k', 'Linewidth', .1);
        ylim([0 .6])
        xlim([.5 2.5])
        if noise_index==1
            ylabel('Length')
        end
        xticks([1 2])
        xticklabels({'Noise', 'Baseline'})
        xtickangle(45)
        set(gca, 'FontSize', 24)
        [h, p, stat]= kstest(ComparisonMatrix.CallLengthsDIF((ComparisonMatrix.NoiseIndex==noise_index)))
        
        
        if h
            [h, p, stats] =ttest(ComparisonMatrix.CallLengthsDIF((ComparisonMatrix.NoiseIndex==noise_index)))
        else
            [p, h, stats] =signrank(ComparisonMatrix.CallLengthsDIF((ComparisonMatrix.NoiseIndex==noise_index)))
        end
        
        
        title(p)
    end
    
    
end
pause(.1)
saveas(gcf, [num2str(date2plot), ' Call length difference ALL NOISE LEVLES together.svg'])
%% CORRELATION COMPARISON FOR SINGLE CALLS


corr_values_single_call   = nan(numel(unique(ALL_NOISE_COMPARISON_callbycall.Date)'), 2);
corr_p_values_single_calls = nan(numel(unique(ALL_NOISE_COMPARISON_callbycall.Date)'), 2);
p_index = 1;
for date2plot = unique(ALL_NOISE_COMPARISON_callbycall.Date)'
ComparisonMatrix = ALL_NOISE_COMPARISON_callbycall(ALL_NOISE_COMPARISON_callbycall.Date == date2plot,:);

  [cl,pl] = corr(ComparisonMatrix.CallLengthsDIF, NoiseConditionValues(ComparisonMatrix.NoiseIndex)' , 'type', 'Spearman')   
   
   
  [cf,pf] = corr(ComparisonMatrix.PrincipalFrequencykHzDIF, NoiseConditionValues(ComparisonMatrix.NoiseIndex)', 'type', 'Spearman')
   corr_values_single_call(p_index,:) =[ cl,cf];
   corr_p_values_single_calls(p_index,:) =[ pl,pf];
  
   
   
    
p_index=p_index+1;

end
%% r and p-values plot for single calls

rand_values = (rand(size(corr_values_single_call,1), 2) -.5)*.2;
figure
subplot(1,2,1)
boxplot(corr_values_single_call)
hold on
plot([1 2] +rand_values, (corr_values_single_call), '.k', 'MarkerSize', 20)
hold on
plot([1 2] +rand_values(corr_p_values_single_calls(:,2)<=0.01,:), corr_values_single_call(corr_p_values_single_calls(:,2)<=0.01,:), 'r.', 'MarkerSize', 25)
xticklabels({'Noise Level', 'Freq'})
set(gca, 'FontSize', 24)

subplot(1,2,2)
boxplot(log10(corr_p_values_single_calls))
hold on
plot([1 2]+rand_values, log10(corr_p_values_single_calls), '.k', 'MarkerSize', 20)
plot([.5 2.5], log10([.05 0.05]), ':k', 'LineWidth', 2)
plot([.5 2.5], log10([.01 0.01]), ':k', 'LineWidth', 3)
xticklabels({'Noise Level', 'Freq'})
set(gca, 'FontSize', 24)

%% Correlation all together

order4plot = [3 1];
rand_disp = (rand(size(CorrelationValues,1),numel(order4plot))-.5)*.2;
figure('units','normalized','outerposition', [0 0 1 1]);


subplot(1,2,1)
boxplot([corr_values_single_call(:,2),CorrelationValues(:, order4plot)])




set(gca, 'FontSize', 24)

hold on
X = [(1+rand_values(:,2)),(ones(size(CorrelationValues,1),1)*((1:numel(order4plot)) +1) +rand_disp)];
Y = [corr_values_single_call(:,2),CorrelationValues(:, order4plot)];
plot(X, Y , 'k.', 'MarkerSize', 20)
plot(X', Y', 'k:', 'LineWidth', 1)

% plot(ones(3,1)*[1 2 3], CorrelationValues(8:10, order4plot), 'r.', 'MarkerSize', 20)

for j=1:numel(order4plot)
    index_01 = CorrelationValues(:,order4plot(j)+1)<=0.01;
index_05 = CorrelationValues(:,order4plot(j)+1)<=0.05;
plot(ones(sum(index_01),1)*(j +1)+ rand_disp(index_01,j), CorrelationValues(index_01, order4plot(j)), '.', 'MarkerSize', 25, 'Color', [1 0 0] )
plot(ones(sum(index_05 & ~index_01),1)*(j +1)+ rand_disp(index_05 & ~index_01,j), CorrelationValues(index_05 & ~index_01, order4plot(j)), '.', 'MarkerSize', 25, 'Color', [1 .65 0] )

end

hold on
plot(1 +rand_values(corr_p_values_single_calls(:,2)<=0.01,2), corr_values_single_call(corr_p_values_single_calls(:,2)<=0.01,2), 'r.', 'MarkerSize', 25)



xticklabels({'Freq ','Mean Freq', 'Total Call Length'})
xtickangle(45)
ylabel('Spearman Correlation')
set(gca, 'FontSize', 24)

subplot(1,2,2)
boxplot(log10([corr_p_values_single_calls(:,2),CorrelationValues(:, order4plot+1)]))
hold on
X = [(1+rand_values(:,2)),(ones(size(CorrelationValues,1),1)*((1:numel(order4plot)) +1) +rand_disp)];
Y = [corr_p_values_single_calls(:,2),CorrelationValues(:, order4plot+1)];
plot(X, log10(Y), 'k.', 'MarkerSize', 20)
% plot((ones(size(CorrelationValues,1),1)*(1:numel(order4plot)) + rand_disp)', log10(CorrelationValues(:, [4 2 6]))', 'k:', 'LineWidth', 1)
% plot(ones(3,1)*(1:numel(order4plot)), log10(CorrelationValues(8:10, [4 2 6])), 'r.', 'MarkerSize', 20)
plot([.5 4.5], log10([.05 0.05]), ':k', 'LineWidth', 2, 'Color', [1 .65 0] )
plot([.5 4.5], log10([.01 0.01]), ':k', 'LineWidth', 3, 'Color', [1 0 0] )
xticklabels({'Freq ','Mean Freq', 'Total Call Length'})
xtickangle(45)
ylabel('P-Value')
set(gca, 'FontSize', 24)

%% adding single call to 

mean_values = [];
figure(mean_values_figure)
for date2plot = unique(ALL_NOISE_COMPARISON_callbycall.Date)'
    date = date2plot;
    experiment = 0;
%     date = floor(date2plot/10);
%     experiment = date2plot - date*10;
    

    Matrix2Compare = ALL_NOISE_COMPARISON_callbycall(ALL_NOISE_COMPARISON_callbycall.Date == date2plot,:);
    
   
    
    mean_values = [mean_values;[ date2plot, nanmean(Matrix2Compare.PrincipalFrequencykHzBASE),nanmean(Matrix2Compare.PrincipalFrequencykHzNOISE)]];
end
                   
%   random_values_mean = .2*(rand(size(mean_values,1), 2) - .5);
  
    subplot(1,3,3)
%     boxplot(mean_values(:, 2:3))
     hold on
    plot(repmat([1 2],size(mean_values,1),1)' + random_values_mean',mean_values(:, 2:3)', ':k')   
    plot(repmat([1 2],size(mean_values,1),1) + random_values_mean,mean_values(:, 2:3), '.k', 'MarkerSize', 20)
     errorbar([1 2],nanmean(mean_values(:, 2:3)), nanstd(mean_values(:, 2:3)))
    xlim([.5 2.5])
     xticks([1 2])
     title('Single Call Freq')
    xticklabels({'Base', 'Noise'})

    ylim([22 34])
    


%% ANOVA COMPARISON

anova_p = nan(numel(unique(ALL_NOISE_COMPARISON_callbycall.Date)'), 2);
p_index = 1;
for date2plot = unique(ALL_NOISE_COMPARISON_callbycall.Date)'
ComparisonMatrix = ALL_NOISE_COMPARISON_callbycall(ALL_NOISE_COMPARISON_callbycall.Date == date2plot,:);

    data = [ComparisonMatrix.CallLengthsBASE;ComparisonMatrix.CallLengthsNOISE];
    group1 = [ ComparisonMatrix.NoiseIndex; ComparisonMatrix.NoiseIndex];
    group2 = [ones(size(ComparisonMatrix,1),1); zeros(size(ComparisonMatrix,1),1)];


       [p, table, stats] = anovan(data, [group1, group2], 'varnames', {['Noise Level ', num2str(date2plot)], ['Condition', num2str(date2plot)]})
   anova_p(p_index,:) =[ table{2:3,7}];
   
   figure
   multcompare(stats, 'Dimension', [1,2])
   
   % suptitle(num2str(date2plot))
    
p_index = p_index+1;

end

figure

boxplot(log10(anova_p))
hold on
plot([1 2], log10(anova_p), '.k', 'MarkerSize', 10)
plot([.5 2.5], log10([.05 0.05]), ':k', 'LineWidth', 2)
plot([.5 2.5], log10([.01 0.01]), ':k', 'LineWidth', 3)
xticklabels({'Noise Level', 'Condition'})
set(gca, 'FontSize', 24)

