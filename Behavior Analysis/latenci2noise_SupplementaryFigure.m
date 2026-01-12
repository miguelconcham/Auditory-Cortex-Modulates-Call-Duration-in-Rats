behavior_data  = '\\experimentfs.bccn-berlin.pri\experiment\share\Miguel\LOMBARD EFFECT NPX files and analysis\ANALYSIS UPDATED 2025\folder to share data\Behavior Analysis';

load('synch_model_spike2audio.mat')
spike2_file     = 'mc250207_3 exp3';

%% LOAD DATA


noise_onset_offset  = readtable('Noise offset and onset.xlsx');
CallStats           = readtable('merged_audio 2025-05-06 12_13 PM_Stats.xlsx');
CallStats.Properties.VariableNames = cellfun(@(x) strrep(x, '_', ''),CallStats.Properties.VariableNames, 'UniformOutput',false );

calls2analyze = {'B','N','DuringEndNoise','DuringNoise','PostNoise','Rebound'};
CallStats = CallStats(ismember(CallStats.Label,calls2analyze),:);


variable_list   = who('-file', [behavior_data,'\',spike2_file]);
stim_var_name   = cell2mat(cellfun(@(x) contains(x, 'stim'), variable_list, 'UniformOutput',false));

load( [behavior_data,'\',spike2_file],variable_list{stim_var_name})
stim_var = eval(variable_list{stim_var_name});


stim_start_end = [stim_var.times( stim_var.level==1)  stim_var.times( stim_var.level==0) ];

stim_start_end(:,1) = predict(synch_model_spike2audio,stim_start_end(:,1));
stim_start_end(:,2) = predict(synch_model_spike2audio,stim_start_end(:,2));

stim_end = [find(diff(stim_start_end(:,1))>0.1); size(stim_start_end(:,1),1)];
stim_start = [1;(find(diff(stim_start_end(:,1))>0.1)+1)];


stim_beg_end = [stim_start_end(stim_start,1) ,stim_start_end(stim_end,2)];



%% some preliminary plots 
 colors = {'r', 'b','b','b','b','y'};
call_type_n = 1;
figure
hold on
for call_type = calls2analyze

    indexes = find(ismember(CallStats.Label,call_type));

    for call_n = indexes'
        call_start  = CallStats.BeginTimes(call_n);
        call_end    = CallStats.EndTimes(call_n);
        fill([call_start call_end call_end call_start], [0 0 1 1],'k','FaceColor', colors{call_type_n}, 'FaceAlpha',.8, 'EdgeColor','none')
    end
    call_type_n = call_type_n+1;
end



for noise_n = 1:size(noise_onset_offset,1)
    noise_start  = noise_onset_offset.OnsetSecTotal(noise_n);
    noise_end    = noise_onset_offset.OffsetSecTotal(noise_n);
    fill([noise_start noise_end noise_end noise_start], [0 0 1 1],'k', 'FaceAlpha',.25, 'EdgeColor','none')
end


%%


BN_indexes = find(ismember(noise_onset_offset.NoiseType, 'N'));
noise_intervals = -noise_onset_offset.OffsetSecTotal(BN_indexes(1:end-1))+ noise_onset_offset.OnsetSecTotal(BN_indexes(2:end));
max_interval = max(noise_intervals);
NOISE_CALL_TABLE = [];
first_call_reduction = [];

for noise_n = 1:numel(BN_indexes)

    noise_onset     = noise_onset_offset.OnsetSecTotal(BN_indexes(noise_n));
    noise_offset    = noise_onset_offset.OffsetSecTotal(BN_indexes(noise_n));

    next_noise_onset = Inf;
    if BN_indexes(noise_n)<size(noise_onset_offset,1)
        next_noise_onset =  noise_onset_offset.OnsetSecTotal(BN_indexes(noise_n)+1);
    end
    prev_noise_offset = noise_onset-max_interval;

    if BN_indexes(noise_n)>1
        prev_noise_offset =  noise_onset_offset.OffsetSecTotal(BN_indexes(noise_n)-1);
    end




    this_noise_sequence      = find(CallStats.BeginTimes>noise_onset & CallStats.EndTimes<=noise_offset & ismember(CallStats.Label, 'N'));
    next_baseline_sequence  =  find(CallStats.BeginTimes>noise_offset & CallStats.EndTimes<=next_noise_onset & ismember(CallStats.Label, 'B'));
    prev_baseline_sequence  =  find(CallStats.BeginTimes>prev_noise_offset & CallStats.EndTimes<=noise_onset & ismember(CallStats.Label, 'B'));

    if ~isempty(this_noise_sequence) && ~isempty(next_baseline_sequence)


        next_basline_lengths    = CallStats.CallLengths(next_baseline_sequence);
        pre_baseline_lengths    = CallStats.CallLengths(prev_baseline_sequence);
        noise_lengths           = CallStats.CallLengths(this_noise_sequence);
        noise_call_onset        = CallStats.BeginTimes(this_noise_sequence(1)) ;
        noise_call_offset       = CallStats.EndTimes(this_noise_sequence(1)) ;
        baseline_call_onset     = CallStats.EndTimes(next_baseline_sequence(1)) ;

        next_basline_freq    = CallStats.PrincipalFrequencykHz(next_baseline_sequence);
        pre_baseline_freq    = CallStats.PrincipalFrequencykHz(prev_baseline_sequence);
        noise_freq           = CallStats.PrincipalFrequencykHz(this_noise_sequence);

        total_baseline_length   = sum(next_basline_lengths);
        total_noise_length      = sum(noise_lengths);
        mean_noise_freq         = mean(noise_freq );
        mean_baseline_freq      = mean(next_basline_freq);
        min_count               = min(numel(next_basline_lengths), numel(noise_lengths));
        mean_call_diff          = mean(next_basline_lengths(1:min_count) - noise_lengths(1:min_count));
        mean_freq_diff          = mean(next_basline_freq(1:min_count) - noise_freq(1:min_count));
        last_stim_noise         = max(stim_beg_end(stim_beg_end(:,1)<noise_call_onset,1));
        last_stim_base          = max(stim_beg_end(stim_beg_end(:,1)<baseline_call_onset,1));

        first_call_reduction = [first_call_reduction;[next_basline_lengths(1) noise_lengths(1) next_basline_freq(1)  noise_freq(1) ...
            noise_call_onset-noise_onset  noise_call_onset-noise_offset ...
            noise_call_offset-noise_onset  noise_call_offset-noise_offset ...
            noise_call_onset-last_stim_noise baseline_call_onset-last_stim_base BN_indexes(noise_n) ]];


        NOISE_CALL_TABLE = [NOISE_CALL_TABLE;[total_baseline_length total_noise_length mean_baseline_freq mean_noise_freq mean_call_diff mean_freq_diff numel(next_basline_lengths)  numel(noise_lengths) BN_indexes(noise_n)]];
    end
    %varaibles = {'1: total_baseline_length','2: total_noise_length',
    %             '3: mean_baseline_freq'   ,'4: mean_noise_freq',
    %             '5: mean_call_diff'       ,'6: mean_freq_diff',
    %             '7: num_base_calls'       ,'8: num_noise_calls',
    %             '9: noise_index'}

    %variables one call 5: noise_call_onset-noise_onset
    %                   8: noise_call_offset-noise_offset
    %                   9: noise call stim latency
    %                   10: baseline call stim latency

end

%% noise baseline standar plot
figure
subplot(1,3,1)

rand_val = (rand(size(NOISE_CALL_TABLE,1),2)-.5)*.2;
plot((repmat([1 2],size(NOISE_CALL_TABLE,1), 1 )+rand_val)',NOISE_CALL_TABLE(:, [1 2])', ':k'  )
hold on

plot((repmat([1 2],size(NOISE_CALL_TABLE,1), 1 )+rand_val)',NOISE_CALL_TABLE(:, [1 2])', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(NOISE_CALL_TABLE(:,1)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(NOISE_CALL_TABLE(:,2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(NOISE_CALL_TABLE(:, 1),NOISE_CALL_TABLE(:, 2));
title({num2str(p),num2str(s_stat.signedrank),num2str(s_stat.zval)})



subplot(1,3,2)
rand_val = (rand(size(NOISE_CALL_TABLE,1),2)-.5)*.2;
plot((repmat([1 2],size(NOISE_CALL_TABLE,1), 1 )+rand_val)',NOISE_CALL_TABLE(:, [1 2]+2)', ':k'  )
hold on

plot((repmat([1 2],size(NOISE_CALL_TABLE,1), 1 )+rand_val)',NOISE_CALL_TABLE(:, [1 2]+2)', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(NOISE_CALL_TABLE(:,1+2)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(NOISE_CALL_TABLE(:,2+2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(NOISE_CALL_TABLE(:, 1+2),NOISE_CALL_TABLE(:, 2+2));
title({num2str(p),num2str(s_stat.signedrank),num2str(s_stat.zval)})


subplot(1,3,3)
conditions2include = NOISE_CALL_TABLE(:,7)>1;
rand_val = (rand(sum(conditions2include),2)-.5)*.2;
plot((repmat([1 2],sum(conditions2include), 1 )+rand_val)',NOISE_CALL_TABLE(conditions2include, [1 2]+6)', ':k'  )
hold on

plot((repmat([1 2],sum(conditions2include), 1 )+rand_val)',NOISE_CALL_TABLE(conditions2include, [1 2]+6)', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(NOISE_CALL_TABLE(conditions2include,1+6)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(NOISE_CALL_TABLE(conditions2include,2+6)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(NOISE_CALL_TABLE(:, 1+6),NOISE_CALL_TABLE(:, 2+6));
title({num2str(p),num2str(s_stat.signedrank),num2str(s_stat.zval)})

%% latency figure
figure('units','normalized','outerposition',[.33 0 .5 1]);
subplot(3,2,1)
idnex2plot = first_call_reduction(:, 1 )>0.05;
rand_val = (rand(sum(idnex2plot),2)-.5)*.2;

plot((repmat([1 2],sum(idnex2plot), 1 )+rand_val)',first_call_reduction(idnex2plot, [1 2])', ':k'  )
hold on

plot((repmat([1 2],sum(idnex2plot), 1 )+rand_val)',first_call_reduction(idnex2plot, [1 2])', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(idnex2plot,1)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(idnex2plot,2)), 'r', 'LineWidth',2)
% [p,h,s_stat] = signrank(first_call_reduction(idnex2plot, 1),first_call_reduction(idnex2plot, 2));
% title({num2str(p),num2str(s_stat.signedrank),num2str(s_stat.zval)})

[h,p,ci,t_stat] = ttest(first_call_reduction(idnex2plot, 1),first_call_reduction(idnex2plot, 2));
title({num2str(p),num2str(t_stat.tstat),num2str(t_stat.df)})





subplot(3,2,2)
rand_val = (rand(sum(idnex2plot),2)-.5)*.2;
plot((repmat([1 2],sum(idnex2plot), 1 )+rand_val)',first_call_reduction(idnex2plot, [1 2]+2)', ':k'  )
hold on

plot((repmat([1 2],sum(idnex2plot), 1 )+rand_val)',first_call_reduction(idnex2plot, [1 2]+2)', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(idnex2plot,1+2)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(idnex2plot,2+2)), 'r', 'LineWidth',2)
% [p,h,s_stat] = signrank(first_call_reduction(idnex2plot, 1+2),first_call_reduction(idnex2plot, 2+2));
% title({num2str(p),num2str(s_stat.signedrank),num2str(s_stat.zval)})

[h,p,ci,t_stat] = ttest(first_call_reduction(idnex2plot, 1+2),first_call_reduction(idnex2plot, 2+2));

title({num2str(p),num2str(t_stat.tstat),num2str(t_stat.df)})

cases2include = first_call_reduction(:,1)-first_call_reduction(:,2)>0 & first_call_reduction(:,5)>4.9 & first_call_reduction(:,5)<5.2;
subplot(3,3,[4 7])
histogram(first_call_reduction(cases2include,5), 5:.005:5.2)
subplot(3,3,5)
plot(first_call_reduction(cases2include,5),first_call_reduction(cases2include,2), '.k')
ftited_reg = fitlm(first_call_reduction(cases2include,5),first_call_reduction(cases2include,2));
plot(ftited_reg, 'Marker','.', 'Color', 'k')
legend('off')
% plot(first_call_reduction(cases2include,5),first_call_reduction(cases2include,1)- first_call_reduction(cases2include,2), '.k')
[c,p] = corr(first_call_reduction(cases2include,5),first_call_reduction(cases2include,2), 'Type', 'Spearman')
title({num2str(c), num2str(p)})

subplot(3,3,6)
plot(first_call_reduction(cases2include,9),first_call_reduction(cases2include,2), '.k')

[c,p] = corr(first_call_reduction(cases2include,9),first_call_reduction(cases2include,1), 'Type', 'Spearman')
title({num2str(c), num2str(p)})




subplot(3,3,8)
plot(first_call_reduction(cases2include,5),first_call_reduction(cases2include,4), '.k')
% plot(first_call_reduction(cases2include,5),first_call_reduction(cases2include,1)- first_call_reduction(cases2include,2), '.k')
[c,p] = corr(first_call_reduction(cases2include,5),first_call_reduction(cases2include,4), 'Type', 'Spearman')
title({num2str(c), num2str(p)})

subplot(3,3,9)
plot(first_call_reduction(cases2include,9),first_call_reduction(cases2include,4), '.k')
[c,p] = corr(first_call_reduction(cases2include,9),first_call_reduction(cases2include,4), 'Type', 'Spearman')
title({num2str(c), num2str(p)})



%% same now for other noise configurations


latency_range = 2;
BN_indexes = find(ismember(noise_onset_offset.NoiseType, {'DuringNoise','DuringNoiseEnd'}));

first_call_reduction = [];

for noise_n = 1:numel(BN_indexes)

    noise_onset     = noise_onset_offset.OnsetSecTotal(BN_indexes(noise_n));
    noise_offset    = noise_onset_offset.OffsetSecTotal(BN_indexes(noise_n));

    next_noise_onset = Inf;
    if BN_indexes(noise_n)<size(noise_onset_offset,1)
        next_noise_onset =  noise_onset_offset.OnsetSecTotal(BN_indexes(noise_n)+1);
    end
    prev_noise_offset = noise_onset-max_interval;

    if BN_indexes(noise_n)>1
        prev_noise_offset =  noise_onset_offset.OffsetSecTotal(BN_indexes(noise_n)-1);
    end




    this_noise_sequence      = find(CallStats.BeginTimes>noise_onset-latency_range & CallStats.EndTimes<=next_noise_onset & ismember(CallStats.Label, {'DuringEndNoise','DuringNoise','PostNoise'}));
    next_baseline_sequence  =  find(CallStats.BeginTimes>noise_offset & CallStats.EndTimes<=next_noise_onset & ismember(CallStats.Label, 'B'));
        disp(~isempty(this_noise_sequence) && ~isempty(next_baseline_sequence))
    if ~isempty(this_noise_sequence) && ~isempty(next_baseline_sequence)


        next_basline_lengths    = CallStats.CallLengths(next_baseline_sequence);
        noise_lengths           = CallStats.CallLengths(this_noise_sequence);
        noise_call_onset        = CallStats.BeginTimes(this_noise_sequence(1)) ;
        noise_call_offset       = CallStats.EndTimes(this_noise_sequence(1)) ;
        baseline_call_onset     = CallStats.EndTimes(next_baseline_sequence(1)) ;

        next_basline_freq    = CallStats.PrincipalFrequencykHz(next_baseline_sequence);
        noise_freq           = CallStats.PrincipalFrequencykHz(this_noise_sequence);

          
        last_stim_noise         = max(stim_beg_end(stim_beg_end(:,1)<noise_call_onset,1));
        last_stim_base          = max(stim_beg_end(stim_beg_end(:,1)<baseline_call_onset,1));

        first_call_reduction = [first_call_reduction;[next_basline_lengths(1) noise_lengths(1) next_basline_freq(1)  noise_freq(1) ...
            -noise_call_onset+noise_onset  noise_call_onset-noise_offset ...
            noise_call_offset-noise_onset  noise_call_offset-noise_offset ...
            noise_call_onset-last_stim_noise baseline_call_onset-last_stim_base BN_indexes(noise_n) ]];


    end
    %varaibles = {'1: total_baseline_length','2: total_noise_length',
    %             '3: mean_baseline_freq'   ,'4: mean_noise_freq',
    %             '5: mean_call_diff'       ,'6: mean_freq_diff',
    %             '7: num_base_calls'       ,'8: num_noise_calls',
    %             '9: noise_index'}

    %variables one call 5: noise_call_onset-noise_onset
    %                   8: noise_call_offset-noise_offset
    %                   9: noise call stim latency
    %                   10: baseline call stim latency

end

%% now plot
figure('units','normalized','outerposition',[.33 0 .5 1]);
ranges2plot = [-.1 0 Inf ];
min_length = 0;
min_freq = 0;
subplot(3,4,1)
call_affected       = first_call_reduction(:,5)./first_call_reduction(:,2);
call_affected(first_call_reduction(:,5)<0) = first_call_reduction(first_call_reduction(:,5)<0,5);
condition2analysie = call_affected<ranges2plot(2) & call_affected>=ranges2plot(1) & first_call_reduction(:,1)>min_length & first_call_reduction(:,3)>min_freq;
sum(condition2analysie)
rand_val = (rand(sum(condition2analysie),2)-.5)*.2;
plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2])', ':k'  )
hold on

plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2])', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(condition2analysie,1)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(condition2analysie,2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(first_call_reduction(condition2analysie, 1),first_call_reduction(condition2analysie, 2));
title({num2str(p),num2str(s_stat.signedrank), num2str(s_stat.zval)})


subplot(3,4,2)
condition2analysie = call_affected<ranges2plot(3) & call_affected>=ranges2plot(2) & first_call_reduction(:,1)>min_length & first_call_reduction(:,3)>min_freq;
rand_val = (rand(sum(condition2analysie),2)-.5)*.2;
plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2])', ':k'  )
hold on

plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2])', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(condition2analysie,1)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(condition2analysie,2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(first_call_reduction(condition2analysie, 1),first_call_reduction(condition2analysie, 2));
title({num2str(p),num2str(s_stat.signedrank), num2str(s_stat.zval)})


subplot(3,4,3)
condition2analysie = call_affected<ranges2plot(2) & call_affected>=ranges2plot(1) & first_call_reduction(:,1)>min_length & first_call_reduction(:,3)>min_freq;

rand_val = (rand(sum(condition2analysie),2)-.5)*.2;
plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2]+2)', ':k'  )
hold on

plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2]+2)', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(condition2analysie,1+2)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(condition2analysie,2+2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(first_call_reduction(condition2analysie, 1+2),first_call_reduction(condition2analysie, 2+2));
title({num2str(p),num2str(s_stat.signedrank), num2str(s_stat.zval)})



subplot(3,4,4)
condition2analysie = call_affected<ranges2plot(3) & call_affected>=ranges2plot(2) & first_call_reduction(:,1)>min_length & first_call_reduction(:,3)>min_freq;
rand_val = (rand(sum(condition2analysie),2)-.5)*.2;
plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2]+2)', ':k'  )
hold on

plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2]+2)', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(condition2analysie,1+2)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(condition2analysie,2+2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(first_call_reduction(condition2analysie, 1+2),first_call_reduction(condition2analysie, 2+2));
title({num2str(p),num2str(s_stat.signedrank), num2str(s_stat.zval)})

condition2analysie = call_affected<Inf & call_affected>=ranges2plot(1) & first_call_reduction(:,1)>min_length & first_call_reduction(:,3)>min_freq;

cases2include = true(size(first_call_reduction(:,1))) & condition2analysie;

freq_differences    = first_call_reduction(cases2include,3)-first_call_reduction(cases2include,4);
length_differences  = first_call_reduction(cases2include,1)-first_call_reduction(cases2include,2);
call_affected       = call_affected(cases2include);

std_condition       = zscore(abs(length_differences))<Inf;

subplot(3,3,[4 7])
histogram(call_affected(std_condition),10)
xlabel('% Call Affected')
ylabel('Count')
subplot(3,3,[5 6])
plot(call_affected(std_condition),length_differences(std_condition), '.k')
% plot(first_call_reduction(cases2include(std_condition),5),first_call_reduction(cases2include(std_condition),1)- first_call_reduction(cases2include(std_condition),2), '.k')

hold on
plot([-.1 1], [0 0], 'b')
plot([.05 0.05], [-.1 .25], ':b')
[c,p] = corr(call_affected(std_condition),length_differences(std_condition), 'Type', 'Spearman');
title({num2str(c), num2str(p)})
xlim tight

% subplot(3,3,6)
% plot(first_call_reduction(cases2include(std_condition),9),first_call_reduction(cases2include(std_condition),1)-first_call_reduction(cases2include(std_condition),2), '.k')
% [c,p] = corr(first_call_reduction(cases2include(std_condition),9),first_call_reduction(cases2include(std_condition),1), 'Type', 'Spearman')
% title({num2str(c), num2str(p)})


std_condition       = abs(freq_differences)<Inf;

subplot(3,3,[8 9])
ftited_reg = fitlm(call_affected(std_condition),freq_differences(std_condition));
plot(ftited_reg, 'Marker','.', 'Color', 'k')
hold on
plot([-.4 1], [0 0], 'b')
plot([.05 0.05], [-3 2], ':b')
% plot(first_call_reduction(cases2include(std_condition),5),first_call_reduction(cases2include(std_condition),1)- first_call_reduction(cases2include(std_condition),2), '.k')
[c,p] = corr(call_affected(std_condition),(freq_differences(std_condition)), 'Type', 'Spearman')
xlim tight
legend('Off')
xlabel('Noise Latency (s)')
title({num2str(c), num2str(p)})
% 
% subplot(3,3,9)
% plot(first_call_reduction(cases2include(std_condition),9),first_call_reduction(cases2include(std_condition),3)-first_call_reduction(cases2include(std_condition),4), '.k')
% [c,p] = corr(first_call_reduction(cases2include(std_condition),9),first_call_reduction(cases2include(std_condition),4), 'Type', 'Spearman')
% title({num2str(c), num2str(p)})
%% same now for pre call noise



latency_range =0;
BN_indexes = find(ismember(noise_onset_offset.NoiseType, {'PostNoise'}));

first_call_reduction = [];

for noise_n = 1:numel(BN_indexes)

    noise_onset     = noise_onset_offset.OnsetSecTotal(BN_indexes(noise_n));
    noise_offset    = noise_onset_offset.OffsetSecTotal(BN_indexes(noise_n));

    next_noise_onset = Inf;
    if BN_indexes(noise_n)<size(noise_onset_offset,1)
        next_noise_onset =  noise_onset_offset.OnsetSecTotal(BN_indexes(noise_n)+1);
    end
    prev_noise_offset = noise_onset-max_interval;

    if BN_indexes(noise_n)>1
        prev_noise_offset =  noise_onset_offset.OffsetSecTotal(BN_indexes(noise_n)-1);
    end




    this_noise_sequence      = find(CallStats.BeginTimes>noise_onset-latency_range & CallStats.EndTimes<=next_noise_onset & ismember(CallStats.Label, {'PostNoise'}));
    next_baseline_sequence  =  find(CallStats.BeginTimes>noise_offset & CallStats.EndTimes<=next_noise_onset & ismember(CallStats.Label, 'B'));
        disp(~isempty(this_noise_sequence) && ~isempty(next_baseline_sequence))
    if ~isempty(this_noise_sequence) && ~isempty(next_baseline_sequence)


        next_basline_lengths    = CallStats.CallLengths(next_baseline_sequence);
        noise_lengths           = CallStats.CallLengths(this_noise_sequence);
        noise_call_onset        = CallStats.BeginTimes(this_noise_sequence(1)) ;
        noise_call_offset       = CallStats.EndTimes(this_noise_sequence(1)) ;
        baseline_call_onset     = CallStats.EndTimes(next_baseline_sequence(1)) ;

        next_basline_freq    = CallStats.PrincipalFrequencykHz(next_baseline_sequence);
        noise_freq           = CallStats.PrincipalFrequencykHz(this_noise_sequence);

          if numel(this_noise_sequence)>1
              if CallStats.BeginTimes(this_noise_sequence(2))-CallStats.EndTimes(this_noise_sequence(1))<0.05
                  noise_lengths(1) = sum(CallStats.CallLengths(this_noise_sequence(1:2)));
                  noise_freq(1) = mean(CallStats.PrincipalFrequencykHz(this_noise_sequence(1:2)));
              end
          end
          if numel(next_baseline_sequence)>1
              if CallStats.BeginTimes(next_baseline_sequence(2))-CallStats.EndTimes(next_baseline_sequence(1))<0.05
                  next_basline_lengths(1) = sum(CallStats.CallLengths(next_baseline_sequence(1:2)));
                  next_basline__freq(1) = mean(CallStats.PrincipalFrequencykHz(next_baseline_sequence(1:2)));
              end

          end


        last_stim_noise         = max(stim_beg_end(stim_beg_end(:,1)<noise_call_onset,1));
        last_stim_base          = max(stim_beg_end(stim_beg_end(:,1)<baseline_call_onset,1));

        first_call_reduction = [first_call_reduction;[next_basline_lengths(1) noise_lengths(1) next_basline_freq(1)  noise_freq(1) ...
            noise_call_onset-noise_onset  noise_call_onset-noise_offset ...
            noise_call_offset-noise_onset  noise_call_offset-noise_offset ...
            noise_call_onset-last_stim_noise baseline_call_onset-last_stim_base BN_indexes(noise_n) ]];


    end
    %varaibles = {'1: total_baseline_length','2: total_noise_length',
    %             '3: mean_baseline_freq'   ,'4: mean_noise_freq',
    %             '5: mean_call_diff'       ,'6: mean_freq_diff',
    %             '7: num_base_calls'       ,'8: num_noise_calls',
    %             '9: noise_index'}

    %variables one call 5: noise_call_onset-noise_onset
    %                   8: noise_call_offset-noise_offset
    %                   9: noise call stim latency
    %                   10: baseline call stim latency

end

%% now plot

figure('units','normalized','outerposition',[.33 0 .5 1]);

relevant_Ragne = abs(first_call_reduction(:,6))<.1 & min(first_call_reduction(:, [1 2]),[],2)>.4 & first_call_reduction(:, 1)<0.74;
time_Ranges = [-0.0 0.0];
x_lim = [-.1 .1];
subplot(3,4,1)
condition2analysie = first_call_reduction(:,6)<time_Ranges(1) & relevant_Ragne;
sum(condition2analysie)
rand_val = (rand(sum(condition2analysie),2)-.5)*.2;
plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2])', ':k'  )
hold on

plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2])', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(condition2analysie,1)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(condition2analysie,2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(first_call_reduction(condition2analysie, 1),first_call_reduction(condition2analysie, 2));
if ~ismember('zval', fields(s_stat))
    s_stat.zval = norminv(p);
end
title({num2str(p),num2str(s_stat.signedrank), num2str(s_stat.zval)})

subplot(3,4,2)
condition2analysie = first_call_reduction(:,6)>time_Ranges(2) & relevant_Ragne;
sum(condition2analysie)
rand_val = (rand(sum(condition2analysie),2)-.5)*.2;
plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2])', ':k'  )
hold on

plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2])', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(condition2analysie,1)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(condition2analysie,2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(first_call_reduction(condition2analysie, 1),first_call_reduction(condition2analysie, 2));
if ~ismember('zval', fields(s_stat))
    s_stat.zval = norminv(p);
end

title({num2str(p),num2str(s_stat.signedrank), num2str(s_stat.zval)})
disp('ttest')
[h,p] = ttest(first_call_reduction(condition2analysie, 1),first_call_reduction(condition2analysie, 2))



subplot(3,4,3)
condition2analysie = first_call_reduction(:,6)<time_Ranges(1) & relevant_Ragne;
rand_val = (rand(sum(condition2analysie),2)-.5)*.2;
plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2]+2)', ':k'  )
hold on

plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2]+2)', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(condition2analysie,1+2)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(condition2analysie,2+2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(first_call_reduction(condition2analysie, 1+2),first_call_reduction(condition2analysie, 2+2));
if ~ismember('zval', fields(s_stat))
    s_stat.zval = norminv(p);
end
title({num2str(p),num2str(s_stat.signedrank), num2str(s_stat.zval)})


subplot(3,4,4)
condition2analysie = first_call_reduction(:,6)>time_Ranges(2) & relevant_Ragne;
rand_val = (rand(sum(condition2analysie),2)-.5)*.2;
plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2]+2)', ':k'  )
hold on

plot((repmat([1 2],sum(condition2analysie), 1 )+rand_val)',first_call_reduction(condition2analysie, [1 2]+2)', '.k', 'MarkerSize',14  )
xticks([1 2])
xticklabels({'Basline','0dB'})
xlim([.5 2.5])
hold on
plot([-.2 .2]+1, [1 1]*median(first_call_reduction(condition2analysie,1+2)), 'r', 'LineWidth',2)
plot([-.2 .2]+2, [1 1]*median(first_call_reduction(condition2analysie,2+2)), 'r', 'LineWidth',2)
[p,h,s_stat] = signrank(first_call_reduction(condition2analysie, 1+2),first_call_reduction(condition2analysie, 2+2));
if ~ismember('zval', fields(s_stat))
    s_stat.zval = norminv(p);
end
title({num2str(p),num2str(s_stat.signedrank), num2str(s_stat.zval)})




freq_differences    = first_call_reduction(relevant_Ragne,3)-first_call_reduction(relevant_Ragne,4);
length_differences  = first_call_reduction(relevant_Ragne,1)-first_call_reduction(relevant_Ragne,2);
noise_offset_latency = first_call_reduction(relevant_Ragne,6);
noise_onset_latency  = first_call_reduction(relevant_Ragne,5);

x = -noise_offset_latency;

std_condition       = abs(length_differences)<Inf;

subplot(3,3,[4 7])
histogram(x(std_condition),10)
xlabel('% Call Affected')
ylabel('Count')
subplot(3,3,[5 6])

ftited_reg = fitlm(x(std_condition),length_differences(std_condition));
plot(ftited_reg, 'Marker','.', 'Color', 'k')


% plot(call_affected(std_condition),length_differences(std_condition), '.k')
% plot(first_call_reduction(cases2include(std_condition),5),first_call_reduction(cases2include(std_condition),1)- first_call_reduction(cases2include(std_condition),2), '.k')

hold on
legend('Off')
plot(x_lim, [0 0], 'b')
[c,p] = corr(x(std_condition),length_differences(std_condition));
title({num2str(c), num2str(p)})
xlim tight

% subplot(3,3,6)
% plot(first_call_reduction(cases2include(std_condition),9),first_call_reduction(cases2include(std_condition),1)-first_call_reduction(cases2include(std_condition),2), '.k')
% [c,p] = corr(first_call_reduction(cases2include(std_condition),9),first_call_reduction(cases2include(std_condition),1), 'Type', 'Spearman')
% title({num2str(c), num2str(p)})


std_condition       = abs(freq_differences)<Inf;

subplot(3,3,[8 9])
ftited_reg = fitlm(x(std_condition),freq_differences(std_condition));
plot(ftited_reg, 'Marker','.', 'Color', 'k')
hold on
plot(x_lim, [0 0], 'b')
% plot(first_call_reduction(cases2include(std_condition),5),first_call_reduction(cases2include(std_condition),1)- first_call_reduction(cases2include(std_condition),2), '.k')
[c,p] = corr(x(std_condition),(freq_differences(std_condition)), 'Type', 'Spearman');
xlim tight
legend('Off')

xlabel('Noise Latency (s)')
title({num2str(c), num2str(p)})