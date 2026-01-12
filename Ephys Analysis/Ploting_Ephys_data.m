%% I LOAD DATASET
disp('############################')
disp('############################')
disp('SECTION # 1')

data_Set_list = dir('*DataSet.mat');


load(data_Set_list(1).name)
DataSet.Experiment =str2double(data_Set_list(1).name(3:10));
fields = fieldnames(DataSet);

if ~ismember( 'NPX_type',fields)
    DataSet.NPX_type = 1;
end

if ~ismember('y_pos',fields)
    DataSet.y_pos = ceil(DataSet.CH/2)*20;

end


ALL_DATA(1) = DataSet;



for file_n=1:numel(data_Set_list)

    load(data_Set_list(file_n).name)
    fields = fieldnames(DataSet);

    if ~ismember(fields, 'NPX_type')
        DataSet.NPX_type = 1;
    end

    if ~ismember('y_pos',fields)
        DataSet.y_pos = ceil(DataSet.CH/2)*20;

    end
    DataSet.Experiment = str2double(data_Set_list(file_n).name(3:10));
    ALL_DATA(file_n) = DataSet;

end
table_data = [];

for data_set_n=1:numel(ALL_DATA)
    ResponseTypes               = ALL_DATA(data_set_n).ResponseTypes;
    ResponseTypes.Experiment    = ones(size(ResponseTypes,1), 1)*ALL_DATA(data_set_n).Experiment;
    ResponseTypes.DataSetN      = ones(size(ResponseTypes,1), 1)*data_set_n;
    ResponseTypes.DataSetIndex  = ones(size(ResponseTypes,1), 1)*data_set_n;
    ResponseTypes.BrainArea     = repmat({'Cortex'}, size(ResponseTypes,1),1);
    NeuropixelsDepth            = ALL_DATA(data_set_n).NeuropixelsDepth;
    channels                    = ResponseTypes.Channel;
    if ALL_DATA(data_set_n).NPX_type==1
        if ~isempty(NeuropixelsDepth)
            [CortexChann, ~]    = assignBrainArea(NeuropixelsDepth);
            ChannDepth         = assign_depth(NeuropixelsDepth,ResponseTypes.Channel);
            ResponseTypes.BrainArea(ResponseTypes.Channel<CortexChann) = {'SubCortical'};
            ResponseTypes.NeuronDepth = ChannDepth;
        else
            dips(['NO NeuropixelsDepth for ', nzum2str(data_set_n)])
        end
    elseif ALL_DATA(data_set_n).NPX_type==2
        cortex_row      = ismember(NeuropixelsDepth.Location, 'Cortex');
        brain_row       = ismember(NeuropixelsDepth.Location, 'Brain');
        thickness_row   = ismember(NeuropixelsDepth.Location, 'Thickness');
        brain_dist2tip = NeuropixelsDepth.Depth(brain_row) ;
        npxincortex =(NeuropixelsDepth.Depth(brain_row) - NeuropixelsDepth.Depth(cortex_row))/NeuropixelsDepth.Depth(thickness_row);
        ChannDepth         = brain_dist2tip-ALL_DATA(data_set_n).y_pos;
        ChannDepth = ChannDepth/npxincortex;
        % cortex_limit =
        ResponseTypes.NeuronDepth = ChannDepth/1000;
        ResponseTypes.BrainArea(ChannDepth>NeuropixelsDepth.Depth(thickness_row)   ) = {'SubCortical'};

    end
    ResponseTypes.ClusterQuality = repmat({'NotAssigned'}, size(ResponseTypes,1), 1);

    for cp=1:size(ResponseTypes,1)
        if ismember(ResponseTypes.ID(cp), ALL_DATA(data_set_n).good_clusters)
            ResponseTypes.ClusterQuality(cp) = {'Good'};
        elseif ismember(ResponseTypes.ID(cp), ALL_DATA(data_set_n).mua_clusters)
            ResponseTypes.ClusterQuality(cp) = {'Mua'};
        end
    end
    % ResponseTypes.ID = ResponseTypes.ID + ResponseTypes.DataSetIndex*1000;
    table_data                  = [table_data;ResponseTypes];
end

table_data(~ismember(table_data.BrainArea, 'Cortex'), :) = [];

table_data(ismember(table_data.ClusterQuality, 'NotAssigned'),:)= [];

ResponseTypeLabels  = unique(table_data.ResponseType);
ResponseSign        = unique(table_data.Sign);
ResponseAlignment   = unique(table_data.Alignment);

ResponseTypeLabels(ismember(ResponseTypeLabels, 'Non')) =  [];
ResponseSign(ismember(ResponseSign, 'Non'))             =  [];
ResponseAlignment(ismember(ResponseAlignment, 'Non'))   =  [];

Combination2mix = { 'After' ,'-' , 'Off','During' ,'-' , 'On';
    'After' ,'+' , 'Off','During' ,'-' , 'On'};

for nn=1:size(table_data, 1)

    for comb_index = 1:size(Combination2mix,1)
        if strcmp(table_data.ResponseType{nn}, Combination2mix{comb_index,1}) && ...
                strcmp(table_data.Sign{nn}, Combination2mix{comb_index,2}) && ...
                strcmp(table_data.Alignment{nn}, Combination2mix{comb_index,3})

            table_data.ResponseType{nn} = Combination2mix{comb_index,4};
            table_data.Sign{nn} = Combination2mix{comb_index,5};
            table_data.Alignment{nn} = Combination2mix{comb_index,6};


            disp(table_data(nn,:))
        end
    end
end



Combinations = unique(table_data(:, {'ResponseType',  'Sign'}));


ALL_PSTH_ONSET  =   [];
ALL_PSTH_OFFSET =   [];
psth_type_code = [];
table_indexes_Bas = [];

type_n =1;
indexes_noise = true(size(table_data,1),1);
for feature_n = 1:size(Combinations,2)
    var2compare = Combinations.Properties.VariableNames{feature_n};
    indexes_noise(~ismember(table_data.(var2compare), Combinations.(var2compare)(type_n))) = false;

end
indexes_noise = find(indexes_noise);

table_data(indexes_noise(3),{'ResponseType'}) = {'During'};
Combinations = unique(table_data(:, {'ResponseType',  'Sign'}));



neuron_type_table = readtable('NeuronTypesAfterRevision.xlsx');

%% II SET PARAMETERS


next_stim =1;
onset_correlation_range             = [0 0.1];
onset_correlation_range2BeforeCells = [-.20 0];
offset_correlation_window           = .1;
offset_correlation_range            = [-.05 .45];
correlation_window_step             = .01;
offset_correlation_example_range    = [0.0 0.05];


onset_peak_range = [-.01 0.05];
call_number_range = [1  5];

numel_PB_calls = 10;
histogram_edges = [-1 1];
x_lim2plot_onset           = [-.1 .25];
x_lim2plot_offset          = [-.1 .25];

baseline_window = [-.2 0];
bin_size        = .01;
bin_edges = histogram_edges(1):bin_size:histogram_edges(2);
smooth_wind     = 1;

time2plot = (histogram_edges(1)+.5*bin_size):bin_size:histogram_edges(2);

c_lim_baseANDpb = [-4 4];
c_lim_NOISE       = [-2 2];

onset_rate_comparison = [-.02 0];
onset_rate_reference = [.0 0.05];
color_map = jet(256);
sign_range = [-0.05 0.05];
pre_index = time2plot>=sign_range(1) & time2plot<=0;
post_index = time2plot<=sign_range(2) & time2plot>=0;


onset_correlation_range_index   = time2plot>=onset_correlation_range(1) & time2plot<=onset_correlation_range(2);
onset_correlation_range_indexBeforeCells  = time2plot>=onset_correlation_range2BeforeCells(1) & time2plot<=onset_correlation_range2BeforeCells(2);;

onset_peak_range_index          = time2plot>=onset_peak_range(1) & time2plot<=onset_peak_range(2);

onset_rate_comparison_index     = time2plot>=onset_rate_comparison(1) & time2plot<=onset_rate_comparison(2);
onset_rate_reference_index      = time2plot>=onset_rate_reference(1) & time2plot<=onset_rate_reference(2);

onset_peak_time                 = time2plot(onset_peak_range_index);
call_properties2corr = {'PrincipalFrequencykHz','MeanPowerdBHz', 'CallLengths'};
neuron_type_names = {'Ramping', 'Before', 'Onset', 'Non Responsive'};

freq_range4corr =  [20 30];


baseline_conditions =  {'Calls','-10 dB','-20 dB','-30 dB','-40 dB','0 dB'};
noise_conditions =  {'-40 dB','-30 dB','-20 dB','-10 dB','0 dB'};

n_surrogates = 1000;
call_stim_margin_onset      = .2;
call_stim_margin_offset     = .2;
time_after_stim     = 2;
prevCalldist     = 0;
baseline_index = time2plot<baseline_window(2) & time2plot>baseline_window(1);
neuron_types = {'A', 'B', 'D', {'N', 'T'}};




%% III ESTIMATE of ONSET AND OFFSET to call (and plot)
ONSET_OFFSET_PSTH = cell(numel(neuron_types),5);

type_n = 1;


correlation_values = nan(size(neuron_type_table,1),2*numel(call_properties2corr) +1);
figure('units','normalized','outerposition',[0 0 .33 1]);
colormap(color_map)
for type = neuron_types
    index_list_neg = find(ismember(neuron_type_table.Type,type{1}))';
    % index_list = find(ismember(neuron_type_table.SecondType,type))';
    type_psth_onset = zeros(numel(neuron_types),diff(histogram_edges)/bin_size);
    type_psth_offset = zeros(numel(neuron_types),diff(histogram_edges)/bin_size);
    offset_index    = zeros(numel(index_list_neg),2);
    onset_index     = zeros(numel(index_list_neg),2);
    rate_index      = zeros(numel(index_list_neg),1);
    if type_n==3
        rates = nan(numel(index_list_neg),1);
    end
    neuron_n = 1;
    for index_n = index_list_neg
    
        dsN             =  table_data.DataSetIndex(index_n);

        spikes_time_sec = double(ALL_DATA(dsN).spike_times(ALL_DATA(dsN).spike_clusters==table_data.ID(index_n)))/30000;
        spikes_time_sec = predict(ALL_DATA(dsN).synch_models.AUDIO_NPX, spikes_time_sec);
         if type_n==3
             non_repeated_spikes = spikes_time_sec(2:end);
             non_repeated_spikes(diff(spikes_time_sec)<0.002) = [];
     

        rates(neuron_n)=numel(non_repeated_spikes)/(max(non_repeated_spikes)-min(non_repeated_spikes));
         end

        STIM_TABLE =  ALL_DATA(dsN).STIM_TIMES;
        CallStats = ALL_DATA(dsN).CallStats;

        isINStim = false(size(CallStats,1),1);
        for call=1:size(CallStats,1)
            isINStim(call) = any(STIM_TABLE.StimStart<=CallStats.BeginTimes(call) & (STIM_TABLE.StimEnd+call_stim_margin_onset)>=CallStats.BeginTimes(call)) ;
        end
        prevCalldist_condition = false(size(isINStim));
        prevCalldist_condition([1;1+find(CallStats.BeginTimes(2:end) -  CallStats.EndTimes(1:end-1)>prevCalldist)]) = true;

        Properties2Test = CallStats(ismember(CallStats.NoiseType, baseline_conditions) & ismember(CallStats.Condition, 'Baseline') ...
            &  ~isINStim & prevCalldist_condition,call_properties2corr);

        ALL_CallBeg_BASE = CallStats.BeginTimes(ismember(CallStats.NoiseType, baseline_conditions) & ismember(CallStats.Condition, 'Baseline') ...
            &  ~isINStim & prevCalldist_condition);

        ALL_CallEnd_BASE = CallStats.EndTimes(ismember(CallStats.NoiseType, baseline_conditions) & ismember(CallStats.Condition, 'Baseline') ...
            &  ~isINStim & prevCalldist_condition);

        ALL_CallLength_BASE = ALL_CallEnd_BASE - ALL_CallBeg_BASE;

        [ALL_CallLength_BASE, order1] = sort(ALL_CallLength_BASE);
        ALL_CallBeg_BASE             = ALL_CallBeg_BASE(order1);
        ALL_CallEnd_BASE             = ALL_CallEnd_BASE(order1);
        Properties2Test              = Properties2Test(order1,:);
        Properties2Test{:,:}            = zscore(Properties2Test{:,:});

        psth_onset = zeros(numel(ALL_CallBeg_BASE),diff(histogram_edges)/bin_size);

        n = 1;
        for call_beg = ALL_CallBeg_BASE'
            spikes2raster       = spikes_time_sec(spikes_time_sec>call_beg+histogram_edges(1) & spikes_time_sec<call_beg+histogram_edges(2)) - call_beg;
            psth_onset(n,:)     = histcounts(spikes2raster, bin_edges);
            n = n+1;
        end

        rate2correlate = zscore(mean(psth_onset(:,onset_correlation_range_index),2));
        for corr_n = 1:numel(call_properties2corr)

            [c,p2_t] = corr(Properties2Test{:,call_properties2corr{corr_n}}, rate2correlate, 'type', 'Spearman');
            correlation_values(index_n,2*corr_n - 1 + [0 1]) = [c,p2_t];
        end
        correlation_values(index_n,end) = type_n;
        mean_psth_onset = mean(psth_onset)/bin_size;


        psth_offset = zeros(size(ALL_CallBeg_BASE,1),diff(histogram_edges)/bin_size);
        n = 1;
        for call_end = ALL_CallEnd_BASE'

            spikes2raster       = spikes_time_sec(spikes_time_sec>call_end+histogram_edges(1) & spikes_time_sec<call_end+histogram_edges(2)) - call_end;
            psth_offset(n,:)     = histcounts(spikes2raster, bin_edges);
            n = n+1;
        end
        mean_psth_offset = mean(psth_offset)/bin_size;

        rate_index(neuron_n) = .5*(sum(psth_onset(:)) + sum(psth_offset(:)))/(numel(psth_offset)*bin_size);

        mean2norm   = mean(mean_psth_onset(baseline_index));
        var2norm    = std(mean_psth_onset(baseline_index));
        if var2norm==0
            mean2norm = mean(mean_psth_onset);
            var2norm = std(mean_psth_onset);
        end

        if var2norm==0
            smoothed_psth(:)=0;
        else
            smoothed_psth = smooth((mean_psth_onset -mean2norm)/var2norm,smooth_wind);
        end
        if smooth_wind>1
            psth2smooth = nan(size(mean_psth_onset,2),1);
            psth2smooth(ceil(smooth_wind*.5)+1:end-(ceil(smooth_wind*.5))) = smoothed_psth(1:end-(2*ceil(smooth_wind*.5)));
        else
            psth2smooth = (mean_psth_onset -mean2norm)/var2norm;
        end
        type_psth_onset(neuron_n,:)     = psth2smooth;


        if var2norm==0
            smoothed_psth(:)=0;
        else
            smoothed_psth = smooth((mean_psth_offset -mean2norm)/var2norm,smooth_wind);
        end
        if smooth_wind>1
            psth2smooth = nan(size(mean_psth_onset,2),1);
            psth2smooth(ceil(smooth_wind*.5)+1:end-(ceil(smooth_wind*.5))) = smoothed_psth(1:end-(2*ceil(smooth_wind*.5)));
        else
            smoothed_psth = (mean_psth_offset -mean2norm)/var2norm;
        end
        type_psth_offset(neuron_n,:)    = smoothed_psth;




        offset_index(neuron_n,[1 2])    = [mean(type_psth_offset(neuron_n,pre_index))  mean(type_psth_offset(neuron_n,post_index))];
        onset_index(neuron_n, [1 2])    = [mean(type_psth_onset(neuron_n,pre_index))  mean(type_psth_onset(neuron_n,post_index))];

        neuron_n = neuron_n+1;
    end

    ONSET_OFFSET_PSTH{type_n,1} = type_psth_onset;
    ONSET_OFFSET_PSTH{type_n,2} = type_psth_offset;
    ONSET_OFFSET_PSTH{type_n,3} = onset_index;
    ONSET_OFFSET_PSTH{type_n,4} = offset_index;
    ONSET_OFFSET_PSTH{type_n,5}  = rate_index;


    subplot(numel(neuron_types),2,2*type_n -1)
    if type_n==1       
         if exist('Index_table', 'var')
            [index_value , index_order] = sort(Index_table.("Base-PB Offset pre")(index_list_neg));
        else

        [index_value , index_order] = sort(offset_index(:,1));
         end
        separation_point = max(find(index_value<0));
    elseif type_n==2
        [index_value , index_order] = sort(onset_index(:,1));
    elseif type_n==3
        if exist('Index_table', 'var')
            [index_value , index_order] = sort(Index_table.("Base-PB Onset post")(index_list_neg));
           
        else
            [index_value , index_order] = sort(onset_index(:,2));
        end
         separation_point = max(find(index_value<0));
    elseif type_n==4
        [index_value , index_order] = sort(onset_index(:,2));
        separation_point = [];
    end

    imagesc(time2plot, 1:size(type_psth_onset,1),type_psth_onset(index_order,:))
    yticks(1:size(type_psth_offset,1))
    yticklabels(index_list_neg(index_order))


    set(gca, 'FontSize', 8)
    clim(c_lim_baseANDpb)
    a = ylabel(neuron_type_names {type_n}, 'FontSize', 14);


    if type_n<numel(neuron_types)
        xticklabels([])
    else
        xlabel('Time (s)')
    end
    xlim(x_lim2plot_onset)
    axis xy
    hold on
    plot([0 0], [.5 size(type_psth_offset,1)+.5], 'w', 'LineWidth',3)
    if ~isempty(separation_point)
        plot(x_lim2plot_onset,[separation_point separation_point],':w', 'LineWidth',3)
    end
    title(numel(index_list_neg))

    subplot(numel(neuron_types),2,2*type_n)
    imagesc(time2plot, 1:size(type_psth_offset,1),type_psth_offset(index_order,:))
    yticks(1:size(type_psth_offset,1))
    yticklabels(index_value(index_order))
    set(gca, 'FontSize', 8)
    if type_n<numel(neuron_types)
        xticklabels([])
    else
        xlabel('Time (s)', 'FontSize',16)
    end

    axis xy
    hold on
    plot([0 0], [.5 size(type_psth_offset,1)+.5], 'w', 'LineWidth',3)
    if ~isempty(separation_point)
        plot(x_lim2plot_onset,[separation_point separation_point],':w', 'LineWidth',3)
    end
    clim(c_lim_baseANDpb)
    xlim(x_lim2plot_onset)

    pause(.5)
    type_n = type_n+1;

end
%% IV ESTIMATE PB (and plot)

type_n = 1;

figure('units','normalized','outerposition',[0.33 0 .33 1]);
ONSET_OFFSET_PSTH_PB = cell(numel(neuron_types),62);
colormap(color_map)
for type = neuron_types
    index_list_neg = find(ismember(neuron_type_table.Type,type{1}))';
    % index_list = find(ismember(neuron_type_table.SecondType,type))';
    type_psth_onset_PB = zeros(numel(neuron_types),diff(histogram_edges)/bin_size);
    type_psth_offset_PB = zeros(numel(neuron_types),diff(histogram_edges)/bin_size);

    neuron_n = 1;
    for index_n = index_list_neg

        dsN             =  table_data.DataSetIndex(index_n);

        spikes_time_sec = double(ALL_DATA(dsN).spike_times(ALL_DATA(dsN).spike_clusters==table_data.ID(index_n)))/30000;
        spikes_time_sec = predict(ALL_DATA(dsN).synch_models.AUDIO_NPX, spikes_time_sec);

        STIM_TABLE =  ALL_DATA(dsN).STIM_TIMES;
        CallStats = ALL_DATA(dsN).CallStats;

        isINStim = false(size(CallStats,1),1);
        for call=1:size(CallStats,1)
            isINStim(call) = any(STIM_TABLE.StimStart<=CallStats.BeginTimes(call) & (STIM_TABLE.StimEnd+call_stim_margin_onset)>=CallStats.BeginTimes(call)) ;
        end
        prevCalldist_condition = false(size(isINStim));
        prevCalldist_condition([1;1+find(CallStats.BeginTimes(2:end) -  CallStats.EndTimes(1:end-1)>prevCalldist)]) = true;


        CallBeg = CallStats.BeginTimes(ismember(CallStats.NoiseType, 'Calls'));


        PB_CallBeg = CallStats.BeginTimes(ismember(CallStats.NoiseType, 'PB'));


        lag = mean(PB_CallBeg(1:numel_PB_calls) - CallBeg(1:numel_PB_calls));

        CallBeg = CallStats.BeginTimes(prevCalldist_condition & ~isINStim & ismember(CallStats.NoiseType, 'Calls'));
        CallEnd = CallStats.EndTimes(prevCalldist_condition & ~isINStim & ismember(CallStats.NoiseType, 'Calls'));

        PB_CallBeg = CallBeg + lag;
        PB_CallEnd = CallEnd + lag;

        %Estimating Call
        psth_onset_PB = zeros(numel(PB_CallBeg),diff(histogram_edges)/bin_size);
        n = 1;
        for call_beg = PB_CallBeg'
            spikes2raster       = spikes_time_sec(spikes_time_sec>call_beg+histogram_edges(1) & spikes_time_sec<call_beg+histogram_edges(2)) - call_beg;
            psth_onset_PB(n,:)     = histcounts(spikes2raster, bin_edges);
            n = n+1;
        end
        mean_psth_onset_PB = mean(psth_onset_PB)/bin_size;

        psth_offset_PB = zeros(size(ALL_CallBeg_BASE,1),diff(histogram_edges)/bin_size);
        n = 1;
        for call_end = PB_CallEnd'
            spikes2raster       = spikes_time_sec(spikes_time_sec>call_end+histogram_edges(1) & spikes_time_sec<call_end+histogram_edges(2)) - call_end;
            psth_offset_PB(n,:)     = histcounts(spikes2raster, bin_edges);
            n = n+1;
        end
        mean_psth_offset_PB = mean(psth_offset_PB)/bin_size;



        mean2norm   = mean(mean_psth_onset_PB(baseline_index));
        var2norm    = std(mean_psth_onset_PB(baseline_index));
        if var2norm==0
            mean2norm = mean(mean_psth_onset_PB);
            var2norm = std(mean_psth_onset_PB);
        end

        if var2norm==0
            smoothed_psth(:)=0;
        else
            smoothed_psth = smooth((mean_psth_onset_PB -mean2norm)/var2norm,smooth_wind);
        end
        if smooth_wind>1
            psth2smooth = nan(size(mean_psth_onset_PB,2),1);
            psth2smooth(ceil(smooth_wind*.5)+1:end-(ceil(smooth_wind*.5))) = smoothed_psth(1:end-(2*ceil(smooth_wind*.5)));
        else
            psth2smooth =   (mean_psth_onset_PB -mean2norm)/var2norm;
        end
        type_psth_onset_PB(neuron_n,:)     = psth2smooth;


        if var2norm==0
            smoothed_psth(:)=0;
        else
            smoothed_psth = smooth((mean_psth_offset_PB -mean2norm)/var2norm,smooth_wind);
        end
        if smooth_wind>1
            psth2smooth = nan(size(mean_psth_offset_PB,2),1);
            psth2smooth(ceil(smooth_wind*.5)+1:end-(ceil(smooth_wind*.5))) = smoothed_psth(1:end-(2*ceil(smooth_wind*.5)));
        else
            psth2smooth =   (mean_psth_offset_PB -mean2norm)/var2norm;
        end
        type_psth_offset_PB(neuron_n,:)    = smoothed_psth;


        neuron_n = neuron_n+1;
    end



    onset_index     = ONSET_OFFSET_PSTH{type_n,3};
    offset_index    = ONSET_OFFSET_PSTH{type_n,4};


    subplot(numel(neuron_types),2,2*type_n -1)
   if type_n==1       
         if exist('Index_table', 'var')
            [index_value , index_order] = sort(Index_table.("Base-PB Offset pre")(index_list_neg));
        else

        [index_value , index_order] = sort(offset_index(:,1));
         end
         separation_point = max(find(index_value<0));
    elseif type_n==2
        [index_value , index_order] = sort(onset_index(:,1));
    elseif type_n==3
        if exist('Index_table', 'var')
            [index_value , index_order] = sort(Index_table.("Base-PB Onset post")(index_list_neg));
            separation_point = max(find(index_value<0));
        else
            [index_value , index_order] = sort(onset_index(:,2));
        end
    elseif type_n==4
        [index_value , index_order] = sort(onset_index(:,2));
        separation_point = [];
    end


    imagesc(time2plot, 1:size(type_psth_onset_PB,1),type_psth_onset_PB(index_order,:))
    yticks(1:size(type_psth_offset_PB,1))
    yticklabels(index_list_neg)

    set(gca, 'FontSize', 8)
    clim(c_lim_baseANDpb)
    a = ylabel(neuron_type_names{type_n}, 'FontSize', 14);

    if type_n<4
        xticklabels([])
    else
        xlabel('Time (s)')
    end
    axis xy
    hold on
    plot([0 0], [.5 size(type_psth_offset_PB,1)+.5], 'w', 'LineWidth',3)
    if ~isempty(separation_point)
        plot(x_lim2plot_onset,[separation_point separation_point],':w', 'LineWidth',3)
    end
    xlim(x_lim2plot_onset)

    subplot(numel(neuron_types),2,2*type_n)
    imagesc(time2plot, 1:size(type_psth_offset_PB,1),type_psth_offset_PB(index_order,:))
    if type_n<4
        xticklabels([])
    else
        xlabel('Time (s)')
    end
    axis xy
    hold on
    plot([0 0], [.5 size(type_psth_offset_PB,1)+.5], 'w', 'LineWidth',3)
    if ~isempty(separation_point)
        plot(x_lim2plot_onset,[separation_point separation_point],':w', 'LineWidth',3)
    end
    clim(c_lim_baseANDpb)
    yticks([])
    pause(.5)
    ONSET_OFFSET_PSTH_PB{type_n,1} = type_psth_onset_PB;
    ONSET_OFFSET_PSTH_PB{type_n,2} = type_psth_offset_PB;
    xlim(x_lim2plot_onset)
    type_n = type_n+1;

end

%% V color plot and CI of BASELINE-PB
color_plot_figure = figure('units','normalized','outerposition',[0 0 .33 1]);
colormap(jet(256))
ci_figure = figure('units','normalized','outerposition',[.33 0 .33 1]);
rate_comp_figure = figure('units','normalized','outerposition',[.66 0 .33 1]);

sign_range = [-0.05 0.05];
pre_index = time2plot>=sign_range(1) & time2plot<0;
post_index = time2plot<=sign_range(2) & time2plot>0;
YLims = [3 3 4 1 8 4 1 1;-2 -2 -4 -1 -6 -1 -.5 -.5];
for type_n = 1:numel(neuron_types)
    figure(color_plot_figure)
    type = neuron_types{type_n};
    index_list_neg = find(ismember(neuron_type_table.Type,type))';
    subplot(numel(neuron_types),2,2*type_n -1)
    onset_index     = ONSET_OFFSET_PSTH{type_n,3};
    offset_index    = ONSET_OFFSET_PSTH{type_n,4};
    color_plot = ONSET_OFFSET_PSTH{type_n,1} - ONSET_OFFSET_PSTH_PB{type_n,1};
    % onset_index_BminN = (mean(color_plot(:, post_index),2) -  mean(color_plot(:, pre_index),2))./(mean(color_plot(:, post_index),2) +  mean(color_plot(:, pre_index),2));
    post_onset_index_BminN = mean(color_plot(:, post_index),2);

    if type_n==1       
         if exist('Index_table', 'var')
            [index_value , index_order] = sort(Index_table.("Base-PB Offset pre")(index_list_neg));
        else

        [index_value , index_order] = sort(offset_index(:,1));
         end
           index_order = index_order(index_value<0);
           separation_point = max(find(index_value<0));

    elseif type_n==2
        [index_value , index_order] = sort(onset_index(:,1));
    elseif type_n==3
        if exist('Index_table', 'var')
            [index_value , index_order] = sort(Index_table.("Base-PB Onset post")(index_list_neg));
            separation_point = max(find(index_value<0));
            index_order = index_order(index_value<0);


           this_Rates= rates(index_order);
        else
            [index_value , index_order] = sort(onset_index(:,2));
        end
    elseif type_n==4
        [index_value , index_order] = sort(onset_index(:,2));
        separation_point = [];
    end

    PB2subtract = ONSET_OFFSET_PSTH_PB{type_n,1};
    PB2subtract(sum(isnan(PB2subtract),2) == size(PB2subtract,2),:) = 0;
    color_plot = ONSET_OFFSET_PSTH{type_n,1} - PB2subtract;

 % index_order = index_order(ismember(index_list(index_order),poleimc_cases_OA))
    imagesc(time2plot, 1:numel(index_order),color_plot(index_order,:))
    hold on
    plot([0 0],[.5 numel(index_order)+.5], 'w', 'LineWidth',3 )
    clim(c_lim_baseANDpb)
    yticks(1:numel(index_order))
    yticklabels(index_list_neg(index_order))
    set(gca, 'FontSize', 7)
    ylabel(neuron_type_names {type_n}, 'FontSize', 14);
    if type_n<numel(neuron_types)
        xticklabels([])
    else
        xlabel('Time (s)')
    end
    xlim(x_lim2plot_onset)
    axis xy
    ylim([.5 numel(index_order)+.5])
    if ~isempty(separation_point)
        plot(x_lim2plot_onset,[separation_point separation_point],':w', 'LineWidth',3)
    end
    title(numel(index_order))


    figure(ci_figure)
    [h,p, ci] = ttest(color_plot(index_order,:));
    h = p<=0.05;

    subplot(numel(neuron_types),2,2*type_n -1)
    no_nan_values = ~isnan(ci(1,:));
    fill([time2plot(no_nan_values) fliplr(time2plot(no_nan_values))],[ci(1,(no_nan_values)) fliplr(ci(2,(no_nan_values)))],'k', 'FaceAlpha',.5, 'EdgeColor','none')
    hold on
    plot(time2plot, nanmean(color_plot(index_order,:)), 'k', 'Linewidth', 3)
    plot(x_lim2plot_onset, [0 0], 'b:')

    y_lim = sort(YLims(:,2*type_n -1))';
    plot([0 0], y_lim, 'r')

    p_locations = find(h);
    p_locations= unique([p_locations,p_locations+1]);
    event_end = [p_locations(find(diff(p_locations)>1)),p_locations(end)];
    event_start = p_locations([1,find(diff(p_locations)>1)+1]);
    plot(bin_edges([event_start;event_end]),y_lim([2 2])*.95, 'r')

     p_locations = find(p<0.01);
     if ~isempty(p_locations)
         p_locations= unique([p_locations,p_locations+1]);
         event_end = [p_locations(find(diff(p_locations)>1)),p_locations(end)];
         event_start = p_locations([1,find(diff(p_locations)>1)+1]);
         plot(bin_edges([event_start;event_end]),y_lim([2 2])*.95, 'm')
     end


      p_locations = find(p<0.001);
     if ~isempty(p_locations)
         p_locations= unique([p_locations,p_locations+1]);
         event_end = [p_locations(find(diff(p_locations)>1)),p_locations(end)];
         event_start = p_locations([1,find(diff(p_locations)>1)+1]);
         plot(bin_edges([event_start;event_end]),y_lim([2 2])*.95, 'g')
     end

    axis([x_lim2plot_onset y_lim])
    ylabel(neuron_types{type_n}, 'FontSize', 14);
    if type_n<numel(neuron_types)
        xticklabels([])
    else
        xlabel('Time (s)')
    end
    xlim(x_lim2plot_onset)

    figure(rate_comp_figure)
    subplot(numel(neuron_types),1,type_n )
    mean_rate       = mean(color_plot(index_order,onset_rate_comparison_index),2);
    mean_rate = mean_rate(~isnan(mean_rate));
    mean_rate_ref   = mean(color_plot(index_order,onset_rate_reference_index),2);
    mean_rate_ref = mean_rate_ref(~isnan(mean_rate_ref));

    rand_pos_rate   = .2*(rand(size(mean_rate))-.5);
    rand_pos_ref    = .2*(rand(size(mean_rate_ref))-.5);

    plot(2 +rand_pos_rate, mean_rate, 'b.' )
    hold on
    plot(1 +rand_pos_ref, mean_rate_ref, 'k.' )

    if kstest(mean_rate)
        [h,p,~,t_stat] = ttest(mean_rate);
        title_legend{1} = ['T-test P= ', num2str(h), ' t-val= ', num2str(t_stat.tstat), ' df= ',num2str(t_stat.df)];
    else
        [h,p,z_val] = signrank(mean_rate);
        title_legend{1} = ['RANK P= ', num2str(h), ' z-val= ', num2str(z_val.signedrank)];
    end
    xticks([1 2])
    xticklabels({'Reference', '20 ms bofre'})



    if kstest(rand_pos_ref)
        [h,p,~,t_stat] = ttest(rand_pos_ref);
        title_legend{2} = ['REF T-test P= ', num2str(h), ' t-val= ', num2str(t_stat.tstat), ' df= ',num2str(t_stat.df)];
    else
        [h,p,z_val] = signrank(rand_pos_ref);
        title_legend{2} = ['REF RANK P= ', num2str(h), ' z-val= ', num2str(z_val.signedrank)];
    end
    if type_n==4
        ylabel({[neuron_types{type_n}{:}], 'CAll-PB difference'}, 'FontSize', 14);
    else
        ylabel({neuron_types{type_n}, 'CAll-PB difference'}, 'FontSize', 14);
    end





    title(title_legend)
    xlim([.5 2.5])




    figure(color_plot_figure)
    subplot(numel(neuron_types),2,2*type_n )

    PB2subtract = ONSET_OFFSET_PSTH_PB{type_n,2};
    PB2subtract(sum(isnan(PB2subtract),2) == size(PB2subtract,2),:) = 0;


    color_plot = ONSET_OFFSET_PSTH{type_n,2} - PB2subtract;
    imagesc(time2plot, 1:numel(index_order),color_plot(index_order,:))
    hold on
    plot([0 0],[.5 numel(index_order)+.5], 'w' , 'LineWidth',3 )
    clim(c_lim_baseANDpb)
    if type_n<numel(neuron_types)
        xticklabels([])
    else
        xlabel('Time (s)')
    end
    if ~isempty(separation_point)
        plot(x_lim2plot_onset,[separation_point separation_point],':w', 'LineWidth',3)
    end
    xlim(x_lim2plot_onset)
    axis xy
    yticks(1:numel(index_order))
    yticklabels((index_value- nanmean(index_value))/nanstd(index_value))
    ylim([.5 numel(index_order)+.5])


    figure(ci_figure)
    [h,p, ci] = ttest(color_plot(index_order,:));

    h = p<=0.05;
    subplot(numel(neuron_types),2,2*type_n )
    no_nan_values = ~isnan(ci(1,:));
    fill([time2plot(no_nan_values) fliplr(time2plot(no_nan_values))],[ci(1,(no_nan_values)) fliplr(ci(2,(no_nan_values)))],'k', 'FaceAlpha',.5, 'EdgeColor','none')
    hold on
    plot(time2plot, nanmean(color_plot(index_order,:)), 'k', 'Linewidth', 3)
    plot(x_lim2plot_onset, [0 0], 'b:')
   
    p_locations = find(h);
    p_locations= unique([p_locations,p_locations+1])
    event_end = [p_locations(find(diff(p_locations)>1)),p_locations(end)];
    event_start = p_locations([1,find(diff(p_locations)>1)+1]);
    plot(bin_edges([event_start;event_end]),y_lim([2 2])*.95, 'r')


    p_locations = find(p<0.01);
    if ~isempty(p_locations)
        p_locations= unique([p_locations,p_locations+1]);
        event_end = [p_locations(find(diff(p_locations)>1)),p_locations(end)];
        event_start = p_locations([1,find(diff(p_locations)>1)+1]);
        plot(bin_edges([event_start;event_end]),y_lim([2 2])*.95, 'm')
    end


      p_locations = find(p<0.001);
     if ~isempty(p_locations)
         p_locations= unique([p_locations,p_locations+1]);
         event_end = [p_locations(find(diff(p_locations)>1)),p_locations(end)];
         event_start = p_locations([1,find(diff(p_locations)>1)+1]);
         plot(bin_edges([event_start;event_end]),y_lim([2 2])*.95, 'g')
     end

    y_lim = sort(YLims(:,2*type_n))';
    plot([0 0], y_lim, 'r')
    axis([x_lim2plot_onset y_lim])

    if type_n<numel(neuron_types)
        xticklabels([])
    else
        xlabel('Time (s)')
    end
    xlim(x_lim2plot_onset)

end


%% VI ploting CALL and PB CI together

Index_table = nan(size(neuron_type_table,1),10);

ci_figure_2cond = figure('units','normalized','outerposition',[.33 0 .33 1]);
for type_n = 1:numel(neuron_types)

    onset_index     = ONSET_OFFSET_PSTH{type_n,3};
    offset_index    = ONSET_OFFSET_PSTH{type_n,4};
    PB2subtract     = ONSET_OFFSET_PSTH_PB{type_n,1};
    PB2subtract(sum(isnan(PB2subtract),2) == size(PB2subtract,2),:) = 0;
    pb_index        = mean(PB2subtract(:, post_index),2);
    color_plot = ONSET_OFFSET_PSTH{type_n,1} - PB2subtract;

    post_onset_index_BminN = mean(color_plot(:, post_index),2);
    pre_onset_index_BminN   = mean(color_plot(:, pre_index),2);
    PB2subtract = ONSET_OFFSET_PSTH_PB{type_n,2};
    PB2subtract(sum(isnan(PB2subtract),2) == size(PB2subtract,2),:) = 0;
    color_plot = ONSET_OFFSET_PSTH{type_n,1} - PB2subtract;

    color_plot = ONSET_OFFSET_PSTH{type_n,2} - PB2subtract;
    pre_offset_index_BminN =  mean(color_plot(:, pre_index),2);
    post_offset_index_BminN =  mean(color_plot(:, post_index),2);
    rate_index = ONSET_OFFSET_PSTH{type_n,5};
    Index_table(ismember(neuron_type_table.Type, neuron_types{type_n}),:) = [pre_onset_index_BminN,post_onset_index_BminN,pre_offset_index_BminN,post_offset_index_BminN,onset_index(:,1),onset_index(:,2),offset_index(:,1),offset_index(:,2),pb_index,rate_index];

    if type_n==1       
         if exist('Index_table', 'var') && istable(Index_table)
            [index_value , index_order] = sort(Index_table.(" Base-PB Offset pre")(index_list_neg));
        else

        [index_value , index_order] = sort(offset_index(:,1));
         end
    elseif type_n==2
        [index_value , index_order] = sort(onset_index(:,1));
    elseif type_n==3
        % [index_value , index_order] = sort(onset_index(:,2));
        % index_order(index_value<0)= [];
        [index_value , index_order] = sort(post_onset_index_BminN);
        index_order(index_value<0)= [];

    elseif type_n==4
        [index_value , index_order] = sort(post_onset_index_BminN);
    end

    figure(ci_figure_2cond)

    color_plot = ONSET_OFFSET_PSTH{type_n,1};
    [~,~, ci] = ttest(color_plot(index_order,:));
    subplot(numel(neuron_types),2,2*type_n -1)
    no_nan_values = ~isnan(ci(1,:));
    fill([time2plot(no_nan_values) fliplr(time2plot(no_nan_values))],[ci(1,(no_nan_values)) fliplr(ci(2,(no_nan_values)))],'b', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time2plot, nanmean(color_plot(index_order,:)), 'b', 'Linewidth', 2)


    color_plot = ONSET_OFFSET_PSTH_PB{type_n,1};

    [~,~, ci] = ttest(color_plot(index_order,:));
    no_nan_values = ~isnan(ci(1,:));
    fill([time2plot(no_nan_values) fliplr(time2plot(no_nan_values))],[ci(1,(no_nan_values)) fliplr(ci(2,(no_nan_values)))],'c', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time2plot, nanmean(color_plot(index_order,:)), 'c', 'Linewidth', 2)

    plot(x_lim2plot_onset, [0 0], 'k:')
    y_lim = ylim;
    plot([0 0], y_lim, 'r')
    axis([x_lim2plot_onset y_lim])
    ylabel(neuron_type_names {type_n}, 'FontSize', 14);

    if type_n<numel(neuron_types)
        xticklabels([])
    else
        xlabel('Time (s)')
    end
    xlim(x_lim2plot_onset)



    color_plot = ONSET_OFFSET_PSTH{type_n,2};
    [~,~, ci] = ttest(color_plot(index_order,:));
    subplot(numel(neuron_types),2,2*type_n )
    no_nan_values = ~isnan(ci(1,:));
    fill([time2plot(no_nan_values) fliplr(time2plot(no_nan_values))],[ci(1,(no_nan_values)) fliplr(ci(2,(no_nan_values)))],'b', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time2plot, nanmean(color_plot(index_order,:)), 'b', 'Linewidth', 2)


    color_plot =ONSET_OFFSET_PSTH_PB{type_n,2};
    [~,~, ci] = ttest(color_plot(index_order,:));
    no_nan_values = ~isnan(ci(1,:));
    fill([time2plot(no_nan_values) fliplr(time2plot(no_nan_values))],[ci(1,(no_nan_values)) fliplr(ci(2,(no_nan_values)))],'c', 'FaceAlpha',.25, 'EdgeColor','none')
    hold on
    plot(time2plot, nanmean(color_plot(index_order,:)), 'c', 'Linewidth', 2)
    plot(x_lim2plot_onset, [0 0], 'k:')
    y_lim = ylim;
    plot([0 0], y_lim, 'r')
    axis([x_lim2plot_onset y_lim])
    if type_n<numel(neuron_types)
        xticklabels([])
    else
        xlabel('Time (s)')
    end
    xlim(x_lim2plot_onset)

end

Index_table = array2table(Index_table);
Index_table.Properties.VariableNames = {'Base-PB Onset pre','Base-PB Onset post', 'Base-PB Offset pre','Base-PB Offset post','Onset pre','Onset post', 'Offset pre', 'Offset post', 'PlayBack index', 'Rate'};

type_n = 4;
for call_n=1:size(Index_table,2)-1

    Index_table{~isnan(Index_table{:,call_n}),call_n} = (Index_table{~isnan(Index_table{:,call_n}),call_n}  - mean(Index_table{~isnan(Index_table{:,call_n}) & ismember(neuron_type_table.Type, neuron_types{type_n}),call_n}))./...
        std(Index_table{~isnan(Index_table{:,call_n}) & ismember(neuron_type_table.Type, neuron_types{type_n}),call_n});

    Index_table{isnan(Index_table{:,call_n}),call_n} =0;
end

Index_table.("Offset RI") = Index_table{:,'Offset post'}-Index_table{:,'Offset pre'};

%% VII Cell type classification
figure
shapes =  {'d', 's', 'o', '.'};
colors =  {'b', 'r', 'm', 'k'};
% limit_cases = [4  105];
% limit_cases = [ 83 102 103 ];
% limit_cases = [1 36 37 110];
% limit_cases = [37];
% limit_cases = [10 19 28 61 50 42 47 72 117];
% 10 is low firing rate
% limit_cases =  [112 ];clear all
% limit_cases = [72 117];
% limit_cases = [ 82 ]
% limit_cases = 105;
limit_cases = [];
current_type = 1;
subplot(2,2,3)
hold on
neuron_type_table = readtable('NeuronTypesAfterRevision.xlsx');
for type_n = 1:4
    sub_type_index = ismember(neuron_type_table.Type, neuron_types{type_n});
    if type_n == current_type

        plot( Index_table{sub_type_index,'Base-PB Offset pre'}, Index_table{sub_type_index,'Offset pre'},shapes{type_n}, 'Color',  colors{type_n})
        for lc = limit_cases
            plot(Index_table{lc,'Offset RI'},Index_table{lc,'Offset pre'},'.g', 'MarkerSize', 20)
            % text(Index_table{lc,'Offset RI'}-.1,Index_table{lc,'Offset pre'}+.1, num2str(lc), 'Color', 'g')
            text(Index_table{lc,'Base-PB Offset pre'}-.1,Index_table{lc,'Offset pre'}+.1, num2str(lc), 'Color', 'g')
            %
        end
    else
        plot(Index_table{sub_type_index,'Base-PB Offset pre'}, Index_table{sub_type_index,'Offset pre'},shapes{type_n}, 'Color',  'k')
    end

    for j = find(sub_type_index)'
        text(Index_table{j,'Base-PB Offset pre'}, Index_table{j,'Offset pre'}, num2str(j))
    end
end

xlabel('Offset Rrebound Index')
ylabel('Offset pre')
axis_lim = axis;
axis_lim = axis_lim*1.2;
plot(axis_lim(1:2), [1.96 1.96], 'k:')
plot(axis_lim(1:2), [-1.96 -1.96], 'k:')
plot([1.96 1.96],axis_lim(3:4), 'k:')
plot([-1.96 -1.96],axis_lim(3:4), 'k:')
% axis([-6 6 -6 8])
axis tight
title('Ramping')


% axis square
current_type = 2;
subplot(2,2,2)
hold on
for type_n = 1:4
    sub_type_index = ismember(neuron_type_table.Type, neuron_types{type_n});
    if type_n == current_type
        plot(Index_table{sub_type_index,'Base-PB Onset pre'}, Index_table{sub_type_index,'Onset pre'},shapes{type_n}, 'Color',  colors{type_n})
        plot(Index_table{limit_cases,'Base-PB Onset pre'}, Index_table{limit_cases,'Onset pre'},   '.g', 'MarkerSize', 20)


    else

        plot(Index_table{sub_type_index,'Base-PB Onset pre'}, Index_table{sub_type_index,'Onset pre'},shapes{type_n}, 'Color',  'k')
    end
end
xlabel('Base-PB Onset pre')
ylabel('Onset pre')
axis_lim = axis;
axis_lim = axis_lim*1.2;
plot(axis_lim(1:2), [0 0], 'k:')
plot([0 0],axis_lim(3:4), 'k:')
axis tight
title('Before')
% axis square

current_type = 3;
subplot(2,2,1)
hold on
for type_n = 1:4

    sub_type_index = ismember(neuron_type_table.Type, neuron_types{type_n});
    if type_n == current_type
        plot(Index_table{sub_type_index,'Base-PB Onset post'}, Index_table{sub_type_index,'Onset post'},shapes{type_n}, 'Color',  colors{type_n})
        hold on
        % plot(Index_table{sub_type_index & ismember(neuron_type_table.SecondType, 'N'),'Base-PB Onset post'}, Index_table{sub_type_index & ismember(neuron_type_table.SecondType, 'N'),'Onset post'},shapes{type_n}, 'Color',  'm')
        plot(Index_table{limit_cases,'Base-PB Onset post'}, Index_table{limit_cases,'Onset post'}, 'g.', 'MarkerSize', 20)


    else

        plot(Index_table{sub_type_index,'Base-PB Onset post'}, Index_table{sub_type_index,'Onset post'},shapes{type_n}, 'Color',  'k')

    end
    
    for call_n = find(sub_type_index)'
        text(Index_table{call_n,'Base-PB Onset post'}, Index_table{call_n,'Onset post'}, num2str(call_n))
    end
end
xlabel('Base-PB Onset post')
ylabel('Onset post')
axis_lim = axis;
axis_lim = axis_lim*1.2;
plot(axis_lim(1:2), [1.96 1.96], 'k:')
plot(axis_lim(1:2), [-1.96 -1.96], 'k:')
plot([1.96 1.96],axis_lim(3:4), 'k:')
plot([-1.96 -1.96],axis_lim(3:4), 'k:')
axis tight
title('Onset +/-')



subplot(2,2,4)
fill([-1.96 1.96 1.96 -1.96], [-1.96 -1.96 1.96 1.96], 'k', 'FaceAlpha', .2, 'EdgeColor', 'none')
hold on
for type_n = 1:4
    sub_type_index = ismember(neuron_type_table.Type, neuron_types{5-type_n});
    if type_n == 4
        plot( Index_table{sub_type_index,'PlayBack index'}, Index_table{sub_type_index,'Base-PB Onset post'},shapes{5-type_n}, 'Color',  colors{5-type_n}, 'MarkerSize', 6)
    else
        plot( Index_table{sub_type_index,'PlayBack index'}, Index_table{sub_type_index,'Base-PB Onset post'},shapes{5-type_n}, 'Color',  colors{5-type_n})
    end
end
xlabel('PlayBack index')
ylabel('Base-PB Onset post')
axis_lim = axis;
axis_lim = axis_lim*1.2;
plot(axis_lim(1:2), [1.96 1.96], 'k:')
plot(axis_lim(1:2), [-1.96 -1.96], 'k:')
plot([1.96 1.96],axis_lim(3:4), 'k:')
plot([-1.96 -1.96],axis_lim(3:4), 'k:')
axis tight
title('Putative Sensory')


% axis square



%% VIIIa ploting depth
figure
sorted_depth = sort(table_data.NeuronDepth);
positions = [1 2];
reference_Depth = mean(table_data.NeuronDepth);
for type_n=1:numel(neuron_types)
    subplot(1,4,type_n )
    sub_type        = ismember(neuron_type_table.Type,neuron_types{type_n});

    if type_n==3

        group_index      = Index_table{:,'Base-PB Onset post'}>0;
        
        sub_type_depth_positive  = table_data.NeuronDepth(sub_type & group_index);
        for sn=1:4
            disp(['During + :', num2str(sum(sub_type & group_index & table_data.DataSetN==sn)), ' at dataset ', num2str(sn)])
            disp(['During - :', num2str(sum(sub_type & ~group_index & table_data.DataSetN==sn)), ' at dataset ', num2str(sn)])
        end
        sub_type_depth_negative  = table_data.NeuronDepth(sub_type & ~group_index);

        others_depth    = table_data.NeuronDepth(~sub_type);

        rand_pos_sub_type = (rand(numel(sub_type_depth_positive),1) - .5)*.2;
        plot(positions(1) +rand_pos_sub_type+.5, -sub_type_depth_positive, '.r', 'MarkerSize',8 )
        hold on
        rand_pos_sub_type = (rand(numel(sub_type_depth_negative),1) - .5)*.2;
        plot(positions(1) +rand_pos_sub_type, -sub_type_depth_negative, '.b', 'MarkerSize', 8 )
        rand_pos_others= (rand(sum(~sub_type),1) - .5)*.2;
        plot(positions(2) +rand_pos_others, -table_data.NeuronDepth(~sub_type ), '.k', 'MarkerSize', 8 )

        legend({'During +', 'During -', 'Others'})


        [~,p1, ~,t_stat] = ttest2(abs(sub_type_depth_positive -  reference_Depth), abs(others_depth - reference_Depth));
        % [~,p2_sr,~,t_stat] = ttest2(sub_type_depth_positive , others_depth );
        % title_parts{1} = ['+ Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{1} = ['+ Dist to mean depth:  ',num2str(mean_dist), ' p=' , num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];


        [~,p1 ,~,t_stat] = ttest2(abs(sub_type_depth_negative -  reference_Depth), abs(others_depth - reference_Depth));
        mean_dist = mean(abs(sub_type_depth_negative -  reference_Depth));
        % [~,p2_sr,~,t_stat] = ttest2(sub_type_depth_negative , others_depth );
        % title_parts{2} = ['- Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{2} = ['- Dist to mean depth: ',num2str(mean_dist), ' p='  num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];
        title(title_parts)

    elseif type_n==1

        group_index      = Index_table{:,'Offset pre'}>0;



        sub_type_depth_positive  = table_data.NeuronDepth(sub_type & group_index);
        sub_type_depth_negative  = table_data.NeuronDepth(sub_type & ~group_index);
        others_depth    = table_data.NeuronDepth(~sub_type);
        sub_type_depth  = table_data.NeuronDepth(sub_type);
        others_depth    = table_data.NeuronDepth(~sub_type);


        rand_pos_sub_type = (rand(numel(sub_type_depth_positive),1) - .5)*.2;
        plot(positions(1) +rand_pos_sub_type+.5, -sub_type_depth_positive, '.r', 'MarkerSize',8 )
        hold on
        rand_pos_sub_type = (rand(numel(sub_type_depth_negative),1) - .5)*.2;
        plot(positions(1) +rand_pos_sub_type, -sub_type_depth_negative, '.b', 'MarkerSize', 8 )

        rand_pos_others= (rand(sum(~sub_type),1) - .5)*.2;
        plot(positions(2) +rand_pos_others, -table_data.NeuronDepth(~sub_type ), '.k', 'MarkerSize', 8 )
        legend({'Offset +', 'Offset -', 'Others'})

        mean_dist = mean(abs(sub_type_depth_positive -  reference_Depth));

        % [~,p1, ~,t_stat] = ttest2(abs(sub_type_depth_positive -  reference_Depth), abs(others_depth - reference_Depth));
        [~,p1,~,t_stat] = ttest2(sub_type_depth_positive , others_depth );
        % title_parts{1} = ['+ Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{1} = ['+ Dist to mean depth:  ',num2str(mean_dist), ' p=' , num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];


        % [~,p1 ~,t_stat] = ttest2(abs(sub_type_depth_negative -  reference_Depth), abs(others_depth - reference_Depth));
        [~,p1,~,t_stat] = ttest2(sub_type_depth_negative , others_depth );
        % title_parts{2} = ['- Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{2} = ['- Dist to mean depth: ',num2str(mean_dist), ' p='  num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];
        title(title_parts)

    else



        sub_type_depth  = table_data.NeuronDepth(sub_type);
        others_depth    = table_data.NeuronDepth(~sub_type);
        sub_type_depth  = table_data.NeuronDepth(sub_type);
        others_depth    = table_data.NeuronDepth(~sub_type);
        [~,p1] = ttest2(abs(sub_type_depth - reference_Depth), abs(others_depth - reference_Depth));
        [~,p2_sr,~, t_stat] = ttest2(sub_type_depth , others_depth );

        rand_pos_sub_type = (rand(sum(sub_type),1) - .5)*.2;
        plot(positions(1) +rand_pos_sub_type, -table_data.NeuronDepth(sub_type), '.k', 'MarkerSize', 8 )
        hold on

        rand_pos_others= (rand(sum(~sub_type),1) - .5)*.2;
        plot(positions(2) +rand_pos_others, -table_data.NeuronDepth(~sub_type ), '.k', 'MarkerSize', 8 )
        % title(['Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr,2)])
        title(['Depth: Mean ',num2str(mean(table_data.NeuronDepth(sub_type))), ' p=' num2str(p2_sr,2), ' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)])
        hold on

    end
    plot([.5 2.5], -[reference_Depth reference_Depth], ':k')

    plot([.5 2.5], -[sorted_depth(round(.75*117)) sorted_depth(round(.75*117))], ':g')
    plot([.5 2.5], -[sorted_depth(round(.25*117)) sorted_depth(round(.25*117))], ':g')
    xlim([.5 2.5])

end

%% VIIIb ploting depth using pdf instead of observations
figure
evaluation_points = -1.4:0.01:0.5;
sorted_depth = sort(table_data.NeuronDepth);
positions = [1 5 9];
reference_Depth = mean(table_data.NeuronDepth);
min_sep = 0.05;
for type_n=1:numel(neuron_types)
    subplot(1,4,type_n )
    sub_type        = ismember(neuron_type_table.Type,neuron_types{type_n});

    if type_n==3

        group_index      = Index_table{:,'Base-PB Onset post'}>0;
        sub_type_depth_positive  = table_data.NeuronDepth(sub_type & group_index);
        disp('Onset ACtivated')
        numel(sub_type_depth_positive)
        sub_type_depth_negative  = table_data.NeuronDepth(sub_type & ~group_index);
        disp('Onset Suppressd')
        numel(sub_type_depth_negative)
        others_depth    = table_data.NeuronDepth(~sub_type);

        [f,xf] = kde(-sub_type_depth_positive, 'Bandwidth',.1, 'EvaluationPoints', evaluation_points);
        f = f';         xf = xf';                f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(1), [xf;flipud(xf)]', 'r','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on
        swarmchart(sub_type_depth_positive*0 +positions(1) , -sub_type_depth_positive, '.r')

        [f,xf] = kde(-sub_type_depth_negative, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f';         xf = xf';                 f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(2), [xf;flipud(xf)]', 'b','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on
        swarmchart(sub_type_depth_negative*0 +positions(2) , -sub_type_depth_negative, '.b')

        [f,xf] = kde(-others_depth, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f';         xf = xf';                  f(xf>=max(-others_depth) |  xf<-1.4)=[];
        xf(xf>=max(-others_depth) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(3), [xf;flipud(xf)]', 'k','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on
        swarmchart(others_depth*0 +positions(3) , -others_depth, '.k')
        legend({'During +', 'During -', 'Others'})


        [~,p1, ~,t_stat] = ttest2(abs(sub_type_depth_positive -  reference_Depth), abs(others_depth - reference_Depth));
        mean_dist = mean(abs(sub_type_depth_positive -  reference_Depth));
         % [~,p1, ~,t_stat] = ttest2(sub_type_depth_positive , others_depth );
         % mean_dist = mean(sub_type_depth_positive);
        % [~,p1,~,t_stat] = ttest2(sub_type_depth_positive , others_depth );
        title_parts{1} = ['+ Dist to mean depth:  ',num2str(mean_dist), ' p=' , num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];
        % title_parts{1} = ['+ Mean depth: ', num2str(mean_dist),  ' p=' , num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];


        [~,p1 ,~,t_stat] = ttest2(abs(sub_type_depth_negative -  reference_Depth), abs(others_depth - reference_Depth));
        mean_dist = mean(abs(sub_type_depth_negative -  reference_Depth));
        % [~,p2_sr,~,t_stat] = ttest2(sub_type_depth_negative , others_depth );
        % title_parts{2} = ['- Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{2} = ['- Dist to mean depth: ',num2str(mean_dist), ' p='  num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];
        title(title_parts)

    elseif type_n==1

        group_index      = Index_table{:,'Base-PB Offset pre'}>0;



        sub_type_depth_positive  = table_data.NeuronDepth(sub_type & group_index);
        sub_type_depth_negative  = table_data.NeuronDepth(sub_type & ~group_index);
        others_depth    = table_data.NeuronDepth(~sub_type);

        sub_type_depth  = table_data.NeuronDepth(sub_type);

        disp('Rampoiung ACtivated')
        numel(sub_type_depth_positive)
        disp('Ramping Suppressd')
        numel(sub_type_depth_negative)

        [f,xf] = kde(-sub_type_depth_positive, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f';         xf = xf';         f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(1), [xf;flipud(xf)]', 'r','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on
        swarmchart(sub_type_depth_positive*0 +positions(1) , -sub_type_depth_positive, '.r')

        [f,xf] = kde(-sub_type_depth_negative, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f';         xf = xf';                  f(xf>=max(-sub_type_depth_negative) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_negative) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(2), [xf;flipud(xf)]', 'b','FaceAlpha',.25, 'EdgeColor', 'None',  'HandleVisibility','off')
        hold on
        swarmchart(sub_type_depth_negative*0 +positions(2) , -sub_type_depth_negative, '.b')

        [f,xf] = kde(-others_depth, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f';         xf = xf';                  f(xf>=max(-others_depth) |  xf<-1.4)=[];
        xf(xf>=max(-others_depth) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(3), [xf;flipud(xf)]', 'k','FaceAlpha',.25, 'EdgeColor', 'None' , 'HandleVisibility','off')
        hold on
        swarmchart(others_depth*0 +positions(3) , -others_depth, '.k')
        legend({'During +', 'During -', 'Others'})


        legend({'Offset +', 'Offset -', 'Others'})

        mean_dist = mean(abs(sub_type_depth_negative -  reference_Depth));

        % % [~,p1, ~,t_stat] = ttest2(abs(sub_type_depth_positive -  reference_Depth), abs(others_depth - reference_Depth));
        % % mean_dist = mean(abs(sub_type_depth_positive -  reference_Depth));
         % [~,p1, ~,t_stat] = ttest2(sub_type_depth_positive , others_depth );
     
        [~,p1,~,t_stat] = ttest2(sub_type_depth_positive , others_depth );
        mean_dist = mean(sub_type_depth_positive);
        % title_parts{1} = ['+ Dist to mean depth:  ',num2str(mean_dist), ' p=' , num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];
        title_parts{1} = ['+ Mean depth: ', num2str(mean_dist),  ' p=' , num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];


        % [~,p1 ~,t_stat] = ttest2(abs(sub_type_depth_negative -  reference_Depth), abs(others_depth - reference_Depth));
        [~,p1,~,t_stat] = ttest2(sub_type_depth_negative , others_depth );
        % title_parts{2} = ['- Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{2} = ['- Depth: ',num2str(mean_dist), ' p='  num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];
        title(title_parts)

    else



        sub_type_depth  = table_data.NeuronDepth(sub_type);
        others_depth    = table_data.NeuronDepth(~sub_type);
        sub_type_depth  = table_data.NeuronDepth(sub_type);
        others_depth    = table_data.NeuronDepth(~sub_type);
        [~,p1] = ttest2(abs(sub_type_depth - reference_Depth), abs(others_depth - reference_Depth));
        [~,p2_sr,~, t_stat] = ttest2(sub_type_depth , others_depth );

        rand_pos_sub_type = (rand(sum(sub_type),1) - .5)*.2;
        numel(sub_type_depth)
        numel(others_depth)



        [f,xf] = kde(-sub_type_depth, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f';         xf = xf';                  f(xf>=max(-sub_type_depth) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(1), [xf;flipud(xf)]', 'r','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on
        swarmchart(sub_type_depth*0 +positions(1) , -sub_type_depth, '.r')


        [f,xf] = kde(-others_depth, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f';         xf = xf';                  f(xf>=max(-others_depth) |  xf<-1.4)=[];
        xf(xf>=max(-others_depth) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(3), [xf;flipud(xf)]', 'k','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on
        swarmchart(others_depth*0 +positions(3) , -others_depth, '.k')
        legend({'During +', 'During -', 'Others'})


        % title(['Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr,2)])
        title(['Depth: Mean ',num2str(mean(table_data.NeuronDepth(sub_type))), ' p=' num2str(p2_sr,2), ' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)])
        hold on

    end
    % plot([.5 2.5], -[reference_Depth reference_Depth], ':k')

    % plot([.5 2.5], -[sorted_depth(round(.75*117)) sorted_depth(round(.75*117))], ':g')
    % plot([.5 2.5], -[sorted_depth(round(.25*117)) sorted_depth(round(.25*117))], ':g')
    % xlim([.5 2.5])

end

%% plot depth per session

evaluation_points = -1.4:0.01:0.5;
evaluation_points_um = 0:40:3840;


for sn = 1:3
    figure('units','normalized','outerposition',[0 0 .5 1]);
subplot(1,2,1 )




group_index      = table_data.DataSetN==sn;
sesion_name = num2str(unique(table_data.Experiment(group_index)));

this_session_deppth  = table_data.NeuronDepth( group_index);


[f,xf] = kde(-this_session_deppth, 'Bandwidth',.1, 'EvaluationPoints', evaluation_points);
f = f';         xf = xf';                f(xf>=max(-this_session_deppth) |  xf<-1.4)=[];
xf(xf>=max(-this_session_deppth) |  xf<-1.4) = [];
fill([f+min_sep; flipud(-f-min_sep)]+positions(1), [xf;flipud(xf)]', 'k','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
hold on
swarmchart(this_session_deppth*0 +positions(1) , -this_session_deppth, '.k')
ylim tight
title(sesion_name)
subplot(1,2,2 )
hold on



group_index           = table_data.DataSetN==sn;
this_session_deppth     = floor(table_data.Channel( group_index)/2)*20;
evaluation_points_um = linspace((min(this_session_deppth)-400), ...
                                max(this_session_deppth), numel(xf));

% evaluation_points_um = (min(this_session_deppth)-200):40:(3840+100);
% [~,xf]                  = kde(this_session_deppth,'Bandwidth',5, 'EvaluationPoints', evaluation_points_um);
xf = evaluation_points_um';

fill([f+min_sep; flipud(-f-min_sep)]+positions(1), [xf;flipud(xf)]', 'k','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
hold on
swarmchart(this_session_deppth*0 +positions(1) , this_session_deppth,'.k' )
ylim tight
end

sn = 4
 figure('units','normalized','outerposition',[0 0 .5 1]);
subplot(1,2,1 )




group_index      = table_data.DataSetN==sn;
sesion_name = num2str(unique(table_data.Experiment(group_index)));

this_session_deppth  = table_data.NeuronDepth( group_index);


[f,xf] = kde(-this_session_deppth, 'Bandwidth',.1, 'EvaluationPoints', evaluation_points);
f = f';         xf = xf';                f(xf>=max(-this_session_deppth) |  xf<-1.4)=[];
xf(xf>=max(-this_session_deppth) |  xf<-1.4) = [];
fill([f+min_sep; flipud(-f-min_sep)]+positions(1), [xf;flipud(xf)]', 'k','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
hold on
swarmchart(this_session_deppth*0 +positions(1) , -this_session_deppth, '.k')
ylim tight
title(sesion_name)
%% depth normlazigin by depth distribution
all_f = kde(-table_data.NeuronDepth, 'Bandwidth',.1, 'EvaluationPoints', evaluation_points);
figure
for type_n=1:numel(neuron_types)
    subplot(1,4,type_n )
    sub_type        = ismember(neuron_type_table.Type,neuron_types{type_n});

    if type_n==3
        group_index      = Index_table{:,'Base-PB Onset post'}>0;
        sub_type_depth_positive  = table_data.NeuronDepth(sub_type & group_index);
        disp('Onset ACtivated')
        numel(sub_type_depth_positive)
        sub_type_depth_negative  = table_data.NeuronDepth(sub_type & ~group_index);
        disp('Onset Suppressd')
        numel(sub_type_depth_negative)
        others_depth    = table_data.NeuronDepth(~sub_type);

        [f,xf] = kde(-sub_type_depth_positive, 'Bandwidth',.1, 'EvaluationPoints', evaluation_points);
        f = f./all_f;
        f = f';         xf = xf';                f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(1), [xf;flipud(xf)]', 'r','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on

        [f,xf] = kde(-sub_type_depth_negative, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f./all_f;
        f = f';         xf = xf';                 f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(2), [xf;flipud(xf)]', 'b','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on

        [f,xf] = kde(-others_depth, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f./all_f;
        f = f';         xf = xf';                  f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(3), [xf;flipud(xf)]', 'k','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on
        % legend({'During +', 'During -', 'Others'})


        [~,p1, ~,t_stat] = ttest2(abs(sub_type_depth_positive -  reference_Depth), abs(others_depth - reference_Depth));
        % [~,p2_sr,~,t_stat] = ttest2(sub_type_depth_positive , others_depth );
        % title_parts{1} = ['+ Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{1} = ['+ Dist to mean depth:  ',num2str(mean_dist), ' p=' , num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];


        [~,p1 ,~,t_stat] = ttest2(abs(sub_type_depth_negative -  reference_Depth), abs(others_depth - reference_Depth));
        mean_dist = mean(abs(sub_type_depth_negative -  reference_Depth));
        % [~,p2_sr,~,t_stat] = ttest2(sub_type_depth_negative , others_depth );
        % title_parts{2} = ['- Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{2} = ['- Dist to mean depth: ',num2str(mean_dist), ' p='  num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];
        title(title_parts)

    elseif type_n==1

        group_index      = Index_table{:,'Offset pre'}>0;
        sub_type_depth_positive  = table_data.NeuronDepth(sub_type & group_index);
        sub_type_depth_negative  = table_data.NeuronDepth(sub_type & ~group_index);
        others_depth    = table_data.NeuronDepth(~sub_type);

        sub_type_depth  = table_data.NeuronDepth(sub_type);
        others_depth    = table_data.NeuronDepth(~sub_type);

        disp('Rampoiung ACtivated')
        numel(sub_type_depth_positive)
        sub_type_depth_negative  = table_data.NeuronDepth(sub_type & ~group_index);
        disp('Ramping Suppressd')
        numel(sub_type_depth_negative)

        [f,xf] = kde(-sub_type_depth_positive, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f./all_f;
        f = f';         xf = xf';         f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(1), [xf;flipud(xf)]', 'r','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on

        [f,xf] = kde(-sub_type_depth_negative, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f./all_f;
        f = f';         xf = xf';                  f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(2), [xf;flipud(xf)]', 'b','FaceAlpha',.25, 'EdgeColor', 'None',  'HandleVisibility','off')
        hold on

        [f,xf] = kde(-others_depth, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f./all_f;
        f = f';         xf = xf';                  f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(3), [xf;flipud(xf)]', 'k','FaceAlpha',.25, 'EdgeColor', 'None' , 'HandleVisibility','off')
        hold on
        % legend({'During +', 'During -', 'Others'})


        % legend({'Offset +', 'Offset -', 'Others'})

        mean_dist = mean(abs(sub_type_depth_negative -  reference_Depth));

        % [~,p1, ~,t_stat] = ttest2(abs(sub_type_depth_positive -  reference_Depth), abs(others_depth - reference_Depth));
        [~,p1,~,t_stat] = ttest2(sub_type_depth_positive , others_depth );
        % title_parts{1} = ['+ Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{1} = ['+ Depth:  ',num2str(mean_dist), ' p=' , num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];


        % [~,p1 ~,t_stat] = ttest2(abs(sub_type_depth_negative -  reference_Depth), abs(others_depth - reference_Depth));
        [~,p1,~,t_stat] = ttest2(sub_type_depth_negative , others_depth );
        % title_parts{2} = ['- Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr)];
        title_parts{2} = ['- Dist to mean depth: ',num2str(mean_dist), ' p='  num2str(p1),' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)];
        title(title_parts)

    else



        sub_type_depth  = table_data.NeuronDepth(sub_type);
        others_depth    = table_data.NeuronDepth(~sub_type);
        sub_type_depth  = table_data.NeuronDepth(sub_type);
        others_depth    = table_data.NeuronDepth(~sub_type);
        [~,p1] = ttest2(abs(sub_type_depth - reference_Depth), abs(others_depth - reference_Depth));
        [~,p2_sr,~, t_stat] = ttest2(sub_type_depth , others_depth );

        rand_pos_sub_type = (rand(sum(sub_type),1) - .5)*.2;
        numel(sub_type_depth)
        numel(others_depth)



        [f,xf] = kde(-sub_type_depth, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f./all_f;
        f = f';         xf = xf';                  f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(1), [xf;flipud(xf)]', 'r','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on


        [f,xf] = kde(-others_depth, 'Bandwidth',.1, 'EvaluationPoints',evaluation_points);
        f = f./all_f;
        f = f';         xf = xf';                  f(xf>=max(-sub_type_depth_positive) |  xf<-1.4)=[];
        xf(xf>=max(-sub_type_depth_positive) |  xf<-1.4) = [];
        fill([f+min_sep; flipud(-f-min_sep)]+positions(3), [xf;flipud(xf)]', 'k','FaceAlpha',.25, 'EdgeColor', 'None', 'HandleVisibility','off')
        hold on
        % legend({'During +', 'During -', 'Others'})


        % title(['Dist to mean depth: ', num2str(p1), ' / Depth: ', num2str(p2_sr,2)])
        title(['Depth: Mean ',num2str(mean(table_data.NeuronDepth(sub_type))), ' p=' num2str(p2_sr,2), ' t_stats = ' num2str(t_stat.tstat), ' df =', num2str(t_stat.df)])
        hold on

    end
    % plot([.5 2.5], -[reference_Depth reference_Depth], ':k')

    plot([positions(1)-2 positions(end)+2], -[1.4 1.4], ':g')
    plot([positions(1)-2 positions(end)+2], -[0 0], ':g')
    axis([positions(1)-2 positions(end)+2 -1.5 .1])
    % plot([.5 2.5], -[sorted_depth(round(.25*117)) sorted_depth(round(.25*117))], ':g')
    % xlim([.5 2.5])

end
%% Parameters for  correlations between rate and call length (new dataset)


depth_lim = -1;
call_offset_stim_margin = .1;
SD_limit = 2.5;
prev_call_range = 0.5;
y_lim_dzscore = [-3 3];
condition_type = 'Baseline';
property = 'PrincipalFrequencykHz'; %CallLengths;
max_time_after_stim= 2;
call_number_range =[1 3];
lim2com = [-0.05 -0 0.05];
ptcl_range =0.25;
full_table = [];
ALL_LENGTH = [];
pooled_data = [];

%% correlations parameters (new dataset)

% types2plot = [1 4];
for sign_val = [-1 1]
    for tt = 1:4
        types2plot = [tt 4];
        call_stim_margin_onset_figure3 = 0.1;

        code_per_dataset = {'sr', 'dc', 'og'};

        all_data_together = [];
        cell_psth_list = cell(2,2);
        n_d = 1;
        psth4ploting        = [];
        prop4ploting        = [];
        noise_level_ploting = [];
        n_units_per_ds = nan(2,1);

        %% ploting correlations (new dataset)

        for dsN             =4

            % sign_label = 'ACTIVATED';
            % Index_table{:,'Base-PB Onset post'}>0;
            % index_list = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Base-PB Onset post'}<0);


            % index_list = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Base-PB Onset post'}<0);
            type_n=types2plot(1);
            type = neuron_types{type_n};

            onset_index2use  = onset_correlation_range_index;
            if type_n==3
                % index_list = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Base-PB Onset post'}<0);
                index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Base-PB Onset post'}*sign_val<0 & table_data.NeuronDepth>-1);

                % index_list = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN );
                sub_tit_neg = 'ONSET INHIBITED';
            elseif type_n==1

                index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Base-PB Offset pre'}*sign_val<0 & table_data.NeuronDepth>depth_lim);
                % index_list = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN );
                sub_tit_neg = 'RAMPING INHIBITED';


            elseif type_n==2
                index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & table_data.NeuronDepth>depth_lim);
                sub_tit_neg= 'BEFORE';
                onset_index2use = onset_correlation_range_indexBeforeCells;
            elseif type_n==4


                index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & table_data.NeuronDepth>depth_lim);

                sub_tit_neg = 'NON RESPONSIVE';
            end




            spikes_time_sec = double(ALL_DATA(dsN).spike_times(ismember(ALL_DATA(dsN).spike_clusters,table_data.ID(index_list_neg))))/30000;
            spikes_time_sec = predict(ALL_DATA(dsN).synch_models.AUDIO_NPX, spikes_time_sec);

            spike_times_cell = cell(numel(index_list_neg),1);

            for in = 1:numel(index_list_neg)

                spike_times_cell{in} = double(ALL_DATA(dsN).spike_times(ALL_DATA(dsN).spike_clusters==table_data.ID(index_list_neg(in))))/30000;
                spike_times_cell{in} = predict(ALL_DATA(dsN).synch_models.AUDIO_NPX, spike_times_cell{in});
            end


            STIM_TABLE =  ALL_DATA(dsN).STIM_TIMES;
            CallStats = ALL_DATA(dsN).CallStats;


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%% PB CALL SELECTION %%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            isINStim = false(size(CallStats,1),1);
            for call=1:size(CallStats,1)
                isINStim(call) = any(STIM_TABLE.StimStart<=CallStats.BeginTimes(call) & (STIM_TABLE.StimEnd+call_stim_margin_onset_figure3)>=CallStats.BeginTimes(call)) ;
            end
            prevCalldist_condition = false(size(isINStim));
            prevCalldist_condition([1;1+find(CallStats.BeginTimes(2:end) -  CallStats.EndTimes(1:end-1)>prevCalldist)]) = true;


            CallBeg_onset = CallStats.BeginTimes(ismember(CallStats.NoiseType, 'Calls'));

            PB_CallBeg  = CallStats.BeginTimes(ismember(CallStats.NoiseType, 'PB'));
            lag         = mean(PB_CallBeg(1:numel_PB_calls) - CallBeg_onset(1:numel_PB_calls));
            CallBeg_onset     = CallStats.BeginTimes(prevCalldist_condition & ~isINStim & ismember(CallStats.NoiseType, 'Calls'));
            CallEnd_onset     = CallStats.EndTimes(prevCalldist_condition & ~isINStim & ismember(CallStats.NoiseType, 'Calls'));
            PB_CallFreq       = CallStats.PrincipalFrequencykHz(prevCalldist_condition & ~isINStim & ismember(CallStats.NoiseType, 'Calls'));

            PB_CallBeg  = CallBeg_onset + lag;
            PB_CallEnd  = CallEnd_onset + lag;
            PB_LENGTHS = PB_CallEnd-PB_CallBeg;

            STIM_TABLE.StimPeriod(ismember(STIM_TABLE.StimPeriod, 'Baseline')) = {'Calls'};

            stim_list = find(ismember(STIM_TABLE.StimPeriod, baseline_conditions) & ismember(STIM_TABLE.StimeType, condition_type));


            isINStim = false(size(CallStats,1),1);
            for call=1:size(CallStats,1)
                isINStim(call) = any(STIM_TABLE.StimStart<=CallStats.BeginTimes(call) & (STIM_TABLE.StimEnd+call_stim_margin_onset_figure3)>=CallStats.BeginTimes(call)) ;
            end
            prevCalldist_condition = false(size(isINStim));
            prevCalldist_condition([1;1+find(CallStats.BeginTimes(2:end) -  CallStats.EndTimes(1:end-1)>prevCalldist)]) = true;

            onset_accepted = ismember(CallStats.NoiseType, baseline_conditions) & ismember(CallStats.Condition, condition_type) ...
                &  ~isINStim & prevCalldist_condition & CallStats.HighFreqkHz<=freq_range4corr(2) & ...
                CallStats.LowFreqkHz>=freq_range4corr(1);


            Call_onsets    = CallStats.BeginTimes(onset_accepted);
            Call_lengths   = CallStats.CallLengths(onset_accepted);
            Call_freqs     = CallStats.PrincipalFrequencykHz(onset_accepted);

            [Call_lengths, length_order] = sort(Call_lengths);
            Call_onsets                  = Call_onsets(length_order);
            Call_freqs                   = Call_freqs(length_order);


            figure('units','normalized','outerposition',[0 0 1 1]);
            subplot(4,4,(1:4:9))

            % subplot(5,2,[5 7])
            hold on
            psth_onset_call_4corr = zeros(numel(Call_onsets),numel(bin_edges)-1);
            for call_n=1:numel(Call_onsets)
                this_call_start = Call_onsets(call_n);
                this_call_length = Call_lengths(call_n);
                spikes_this_call = spikes_time_sec(spikes_time_sec>=histogram_edges(1)+this_call_start & spikes_time_sec<=histogram_edges(2)+this_call_start)-this_call_start;
                plot(spikes_this_call, spikes_this_call*0 + call_n, 'k.')
                fill([0 this_call_length this_call_length 0], [-.5 -.5 .5 .5]+call_n, 'r', 'FaceAlpha', .5, 'EdgeColor','none')
                psth_onset_call_4corr(call_n,:) = histcounts(spikes_this_call, bin_edges);
                other_calls = find(CallStats.BeginTimes-this_call_start<histogram_edges(2) & CallStats.BeginTimes-this_call_start>=histogram_edges(1));
                other_calls(CallStats.BeginTimes(other_calls)==this_call_start) = [];
                if ~isempty(other_calls)
                    for oc = other_calls'
                        other_call_start = CallStats.BeginTimes(oc);
                        other_call_end = CallStats.EndTimes(oc);

                        fill([other_call_start other_call_end other_call_end other_call_start]-this_call_start, [-.5 -.5 .5 .5]+call_n, 'k','FaceAlpha',.25, 'EdgeColor','none')
                    end
                end
            end
            xlim([-.2 .4])

            rate_before = mean(psth_onset_call_4corr(:,onset_correlation_range_index ),2);

            index2use              = abs(zscore(Call_lengths))<2 & abs(zscore(Call_freqs))<2 & rate_before>0
            try
                [c,h] = corr(rate_before(index2use),Call_lengths(index2use), 'Type','Spearman' );
                table4corr = array2table([rate_before(index2use),Call_lengths(index2use)], 'VariableNames',{'RateBeforeHz','CallLengthS'});
                linear_model = fitlm(table4corr);
                subplot(4,4,13)
                plot(rate_before(index2use)/bin_size +rand(sum(index2use),1)*2, Call_lengths(index2use), '.r')
                x = linspace(min(rate_before(index2use)), max(rate_before(index2use))+.1, 50);
                y = predict(linear_model,x');
                hold on
                plot(x/bin_size, y, 'k')
                title([num2str(linear_model.Rsquared.Ordinary),'// p = '  ,    num2str(linear_model.coefTest)])
                legend('off')

                all_z_scored_data = [all_z_scored_data;[zscore(rate_before(index2use)), zscore(Call_lengths(index2use)), Call_lengths(index2use)*0 + sign_val Call_lengths(index2use)*0+type_n Call_lengths(index2use)*0+dsN]];
            catch
            end
           



            psth_onset_call = zeros(numel(CallBeg_onset),numel(bin_edges)-1);
            psth_onset_PB   = zeros(numel(CallBeg_onset),numel(bin_edges)-1);

            [PB_LENGTHS, PB_order] = sort(PB_LENGTHS);
            CallBeg_onset   = CallBeg_onset(PB_order);
            PB_CallBeg      = PB_CallBeg(PB_order);

            for call_n=1:numel(CallBeg_onset)
                this_call_start = CallBeg_onset(call_n);
                spikes_this_call = spikes_time_sec(spikes_time_sec>=histogram_edges(1)+this_call_start & spikes_time_sec<=histogram_edges(2)+this_call_start)-this_call_start;

                psth_onset_call(call_n,:) = histcounts(spikes_this_call, bin_edges);


                this_PB_start = PB_CallBeg(call_n);

                spikes_this_call = spikes_time_sec(spikes_time_sec>=histogram_edges(1)+this_PB_start & spikes_time_sec<=histogram_edges(2)+this_PB_start)-this_PB_start;

                psth_onset_PB(call_n,:) = histcounts(spikes_this_call, bin_edges);


            end


            subplot(4,4,(1:4:5)+1)
            plot(time2plot, mean(psth_onset_call)/bin_size, 'r')

            hold on
            plot(time2plot, mean(psth_onset_PB)/bin_size, 'k')
            xlim([-.2 .4])
            title(sub_tit_neg)


            type_n=types2plot(2);
            type = neuron_types{type_n};
            index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & table_data.NeuronDepth>depth_lim);
            sub_tit_neg = 'NON RESPONSIVE';


            spikes_time_sec = double(ALL_DATA(dsN).spike_times(ismember(ALL_DATA(dsN).spike_clusters,table_data.ID(index_list_neg))))/30000;
            spikes_time_sec = predict(ALL_DATA(dsN).synch_models.AUDIO_NPX, spikes_time_sec);


            subplot(4,4,(1:4:9)+2)
            hold on
            psth_onset_call_4corr = zeros(numel(Call_onsets),numel(bin_edges)-1);
            for call_n=1:numel(Call_onsets)
                this_call_start = Call_onsets(call_n);
                this_call_length = Call_lengths(call_n);
                spikes_this_call = spikes_time_sec(spikes_time_sec>=histogram_edges(1)+this_call_start & spikes_time_sec<=histogram_edges(2)+this_call_start)-this_call_start;
                plot(spikes_this_call, spikes_this_call*0 + call_n, 'k.')
                fill([0 this_call_length this_call_length 0], [-.5 -.5 .5 .5]+call_n, 'r', 'FaceAlpha', .5, 'EdgeColor','none')
                psth_onset_call_4corr(call_n,:) = histcounts(spikes_this_call, bin_edges);


                other_calls = find(CallStats.BeginTimes-this_call_start<histogram_edges(2) & CallStats.BeginTimes-this_call_start>=histogram_edges(1));
                other_calls(CallStats.BeginTimes(other_calls)==this_call_start) = [];
                if ~isempty(other_calls)
                    for oc = other_calls'
                        other_call_start = CallStats.BeginTimes(oc);
                        other_call_end = CallStats.EndTimes(oc);

                        fill([other_call_start other_call_end other_call_end other_call_start]-this_call_start, [-.5 -.5 .5 .5]+call_n, 'k','FaceAlpha',.25, 'EdgeColor','none')
                    end
                end
            end
            xlim([-.2 .4])


            rate_before = mean(psth_onset_call_4corr(:,onset_correlation_range_index ),2);

            length2remove = abs(zscore(Call_lengths))>3;
            [c,h] = corr(rate_before(~length2remove),Call_lengths(~length2remove), 'Type','Spearman' );
            table4corr = array2table([rate_before(~length2remove),Call_lengths(~length2remove)], 'VariableNames',{'RateBeforeHz','CallLengthS'});
            linear_model = fitlm(table4corr);


            subplot(4,4,(13+2))
            plot(linear_model)
            title([num2str(linear_model.Rsquared.Ordinary),'// p = '  ,    num2str(linear_model.coefTest)])
            legend('off')

            psth_onset_call = zeros(numel(CallBeg_onset),numel(bin_edges)-1);
            psth_onset_PB   = zeros(numel(CallBeg_onset),numel(bin_edges)-1);

            [PB_LENGTHS, PB_order] = sort(PB_LENGTHS);
            CallBeg_onset   = CallBeg_onset(PB_order);
            PB_CallBeg      = PB_CallBeg(PB_order);

            for call_n=1:numel(CallBeg_onset)
                this_call_start = CallBeg_onset(call_n);
                spikes_this_call = spikes_time_sec(spikes_time_sec>=histogram_edges(1)+this_call_start & spikes_time_sec<=histogram_edges(2)+this_call_start)-this_call_start;

                psth_onset_call(call_n,:) = histcounts(spikes_this_call, bin_edges);


                this_PB_start = PB_CallBeg(call_n);

                spikes_this_call = spikes_time_sec(spikes_time_sec>=histogram_edges(1)+this_PB_start & spikes_time_sec<=histogram_edges(2)+this_PB_start)-this_PB_start;

                psth_onset_PB(call_n,:) = histcounts(spikes_this_call, bin_edges);


            end


            subplot(4,4,(1:4:5)+3)
            plot(time2plot, mean(psth_onset_call)/bin_size, 'r')

            hold on
            plot(time2plot, mean(psth_onset_PB)/bin_size, 'k')
            xlim([-.2 .4])
            title(sub_tit_neg)

            pause(.1)
        end
    end
end
%% ploting r2 between call length and population rate (obtained from table: summary length correlation) (Fig 2G)

r2_comparison = [0.22485 ,0.044395,0.1983; 0.051479,0.0027017,0.060285; 0.14076,0.079712,0.08243]';
r2_sign         = [1 1 1;1 1 -1; 1 1 1]';
p_comparison = [0.000000000015518,0.048784,0.000000025075;0.0047976,0.66231,0.0020054;0.00021103,0.0055706,0.0052659]';
labels2x = {'OnsetSuppressed','NonResponsive','RampingSuppressed'};

figure
subplot(1,2,1)
plot(repmat([1 2 3],3,1)',sqrt(r2_comparison).*r2_sign, 'k.')
hold on
plot(repmat([1 2 3],3,1)',sqrt(r2_comparison).*r2_sign, 'k:')
ylim([-.5 .5])
xticks([1 2 3])
xticklabels(labels2x)
xlim([.5 3.5])


subplot(1,2,2)
plot(repmat([1 2 3],3,1)',log10(p_comparison), 'k.')
hold on
plot(repmat([1 2 3],3,1)',log10(p_comparison), 'k:')
xticks([1 2 3])
plot([.2 3.5], log10([0.05 0.05]/4), 'g')
xlim([.5 3.5])

xticklabels(labels2x)
%% reading from table: summary length correlation

r2summary = readtable('summary length correlation.xlsx')
figure
scatter(categorical(r2summary.NeuronType), sqrt(r2summary.r2).*r2summary.CorrelationSign)
set(gca, 'FontSize', 14)
ylim([-.5 .5])
xtickangle(90)
ylabel('R')




%% creating matrix for svm (new dataset)



dsN = 4;
type_n=3;
type = neuron_types{type_n};
onset_index2use  = onset_correlation_range_index;
if type_n==3
    index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Base-PB Onset post'}<0 & table_data.NeuronDepth>-1);
    sub_tit_neg = 'ONSET INHIBITED';
elseif type_n==1
    index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Offset pre'}<0 & table_data.NeuronDepth>depth_lim);
    sub_tit_neg = 'RAMPING INHIBITED';
elseif type_n==2
    index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & table_data.NeuronDepth>depth_lim);
    sub_tit_neg= 'BEFORE';
    onset_index2use = onset_correlation_range_indexBeforeCells;
elseif type_n==4
    index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & table_data.NeuronDepth>depth_lim);
    sub_tit_neg = 'NON RESPONSIVE';
end


spikes_time_sec = double(ALL_DATA(dsN).spike_times(ismember(ALL_DATA(dsN).spike_clusters,table_data.ID(index_list_neg))))/30000;
spikes_time_sec = predict(ALL_DATA(dsN).synch_models.AUDIO_NPX, spikes_time_sec);

STIM_TABLE =  ALL_DATA(dsN).STIM_TIMES;
CallStats = ALL_DATA(dsN).CallStats;

if dsN==4

    CallStats = readtable('CorrectedBoxes_Stats.xlsx');
    CallStats.Properties.VariableNames = cellfun(@(x) strrep(x, '_', ''),CallStats.Properties.VariableNames, 'UniformOutput',false );



end


next_call_distance = zeros(size(CallStats,1),1);
curren_Stim_latency = zeros(size(CallStats,1),1);
time2nextStim = zeros(size(CallStats,1),1);

for call_n=1:size(CallStats,1)
    last_stim = max(find(STIM_TABLE.StimStart<CallStats.BeginTimes(call_n)));
    if last_stim<size(STIM_TABLE,1)
        curren_Stim_latency(call_n) = STIM_TABLE.StimStart(last_stim+1)-STIM_TABLE.StimStart(last_stim);
        time2nextStim(call_n) =STIM_TABLE.StimStart(last_stim+1)-CallStats.EndTimes(call_n);
    else
        curren_Stim_latency(call_n) = Inf;
        time2nextStim(call_n)  = Inf;
    end

end

for call_n=1:size(CallStats,1)-1
    next_call_distance(call_n) = CallStats.BeginTimes(call_n+1) -CallStats.EndTimes(call_n);
end

not_followed_calls = next_call_distance>=time2nextStim;






calls_in_stim = any( STIM_TABLE.StimStart<=CallStats.BeginTimes' & STIM_TABLE.StimEnd+call_stim_margin_onset_figure3>=CallStats.BeginTimes',1);
calls_in_stim = calls_in_stim';
call_is_valid =  ismember(CallStats.Label, {'B'}');
%
% figure
% histogram(CallStats.CallLengths(~calls_in_stim & not_followed_calls & next_call_distance>0.250), 0:0.01:1,'FaceColor','k','Normalization','percentage','EdgeColor', 'none')
% hold on
% histogram(CallStats.CallLengths(~calls_in_stim & ~not_followed_calls & next_call_distance>0.250), 0:0.01:1,'FaceColor','r','Normalization','percentage','EdgeColor', 'none')
h =true;
min_diff = 1;
index_followedbycall    = find(~calls_in_stim & ~not_followed_calls & next_call_distance>=0.25 & call_is_valid);
index_notfollowedbycall =  find(~calls_in_stim & not_followed_calls & next_call_distance>=0.25 & call_is_valid);

lengths_1 = CallStats.CallLengths(index_followedbycall);
lengths_2 = CallStats.CallLengths(index_notfollowedbycall);
[shortests] = match_lengths(lengths_1,lengths_2);

index_followedbycall = index_followedbycall(shortests(shortests(:,1)<min_diff,2));
index_notfollowedbycall = index_notfollowedbycall(shortests(shortests(:,1)<min_diff,3));

while h & ~isempty(index_followedbycall) &  ~isempty(index_notfollowedbycall)
    [h,p] = kstest2(CallStats.CallLengths(index_followedbycall), CallStats.CallLengths(index_notfollowedbycall));
    lengths_1 = CallStats.CallLengths(index_followedbycall);
    lengths_2 = CallStats.CallLengths(index_notfollowedbycall);

    [shortests] = match_lengths(lengths_1,lengths_2);
    index_followedbycall = index_followedbycall(shortests(shortests(:,1)<min_diff,2));
    index_notfollowedbycall = index_notfollowedbycall(shortests(shortests(:,1)<min_diff,3));
    min_diff=min_diff-0.001;
end


figure
plot(CallStats.CallLengths(index_followedbycall), CallStats.CallLengths(index_notfollowedbycall), '.k')

figure
histogram(CallStats.CallLengths(index_followedbycall), 0:0.01:0.4, 'FaceColor','r', 'FaceAlpha', .2,'Normalization','probability', 'EdgeColor', 'none')
hold on

histogram(CallStats.CallLengths(index_notfollowedbycall), 0:0.01:0.4, 'FaceColor','k', 'FaceAlpha',.2, 'Normalization','probability', 'EdgeColor', 'none')



figure
histogram(next_call_distance(index_followedbycall), 0:0.01:.6, 'FaceColor','r', 'FaceAlpha', .2,'Normalization','probability', 'EdgeColor', 'none')

figure
hold on

for  call_n=1:size(CallStats,1)
    call_start = CallStats.BeginTimes(call_n);
    call_end = CallStats.EndTimes(call_n);
    if not_followed_calls(call_n)
        if ismember(call_n, index_notfollowedbycall)
            fill([call_start call_end call_end call_start], [0 0 1 1], 'k', 'EdgeColor', 'none')
        else
            fill([call_start call_end call_end call_start], [0 0 1 1], 'k', 'EdgeColor', 'none', 'FaceAlpha', .25)
        end

    else
        if ismember(call_n, index_followedbycall)
            fill([call_start call_end call_end call_start], [0 0 1 1], 'r', 'EdgeColor', 'none')
        else
            fill([call_start call_end call_end call_start], [0 0 1 1], 'r', 'EdgeColor', 'none', 'FaceAlpha', .25)
        end
    end
end


psth_offset_call = nan(numel(index_followedbycall), numel(bin_edges)-1);

psth_offset_nocall = nan(numel(index_followedbycall), numel(bin_edges)-1);


for call_n = 1:numel(index_followedbycall)
    call_offset_followed = CallStats.EndTimes(index_followedbycall(call_n));

    spike_times = spikes_time_sec(spikes_time_sec>=call_offset_followed+histogram_edges(1) & spikes_time_sec<=call_offset_followed+histogram_edges(2))-call_offset_followed;
    psth_offset_call(call_n,:) = histcounts(spike_times, bin_edges);

    call_offset_NOTfollowed = CallStats.EndTimes(index_notfollowedbycall(call_n));

    spike_times = spikes_time_sec(spikes_time_sec>=call_offset_NOTfollowed+histogram_edges(1) & spikes_time_sec<=call_offset_NOTfollowed+histogram_edges(2))-call_offset_NOTfollowed;
    psth_offset_nocall(call_n,:) = histcounts(spike_times, bin_edges);
end



figure
colormap(1-gray)
subplot(5,1,1)
imagesc(time2plot,1:size(psth_offset_call,1), psth_offset_call)
axis xy
xlim([.1 .25])
subplot(5,1,2)
imagesc(time2plot,1:size(psth_offset_nocall,1), psth_offset_nocall)
axis xy
xlim([.1 .25])
subplot(5,1,3:5)
plot(time2plot, movmean(mean(psth_offset_call)/bin_size,5), 'r')


hold on
plot(time2plot, movmean(mean(psth_offset_nocall)/bin_size,5), 'k')

xlim([.1 .25])


%% creating matrix for svm prediction selecting right interval (new dataset)
predictive_window = [.1 .25];


t_index = time2plot  >=predictive_window(1) &  time2plot  <=predictive_window(2)  ;



matrix2svm = [psth_offset_call(:,t_index);psth_offset_nocall(:,t_index)];
matrix2svm = [matrix2svm,[zeros(size(psth_offset_call,1),1);ones(size(psth_offset_call,1),1)]];

%% improving rate estimate (new dataset)
matrix2svm_sm = matrix2svm;
for j=1:size(matrix2svm,1)

    matrix2svm_sm(j,1:end-1) = movmean(matrix2svm_sm(j,1:end-1),2);
end

%% ploitng svm summary 

svm_summary = [70.3	29.7	92.4	7.6;
    70.7	29.3	82.8	17.2;
    61.2	38.8	57.1	42.9];

figure
plot(repmat([1 2 3 4],3,1)', svm_summary', '.k', 'MarKerSize', 10)
hold on
plot(repmat([1 2 3 4],3,1)', svm_summary', ':k', 'MarKerSize', 10)
xlim([.5 4.5])
xticks(1:4)
xticklabels({'TP','FN','TN','FP'})

%% plot neuron examples( datasets 1-3)
prevCalldist = 0.1;
before_cells = [21,35,56,82,96,100,110,115,116];
after_Cells = [1,2,4,6,7,8,10,23,25,26,27,31,32,34,36,37,45,47,48,49,50,61,66,72,74,79,81,84,90,95,97,101,104,112,117];
during_Cells =  [5,14,15, 39,43,44,46,51,52,53,54,57,58,59,60,62,65,70,73,80,83,86,87,88,89,94,102,103,105,106,111];
poleimc_cases_OA = [14  119 159 164 ]
poleimc_cases_OS = [146 136 141 111]
for index_n=poleimc_cases



    dsN             =  table_data.DataSetIndex(index_n);

    spikes_time_sec = double(ALL_DATA(dsN).spike_times(ALL_DATA(dsN).spike_clusters==table_data.ID(index_n)))/30000;
    spikes_time_sec = predict(ALL_DATA(dsN).synch_models.AUDIO_NPX, spikes_time_sec);

    STIM_TABLE =  ALL_DATA(dsN).STIM_TIMES;
    CallStats = ALL_DATA(dsN).CallStats;

    isINStim = false(size(CallStats,1),1);
    for call=1:size(CallStats,1)
        isINStim(call) = any(STIM_TABLE.StimStart<=CallStats.BeginTimes(call) & (STIM_TABLE.StimEnd+call_stim_margin_onset)>=CallStats.BeginTimes(call)) ;
    end
    prevCalldist_condition = false(size(isINStim));
    prevCalldist_condition([1;1+find(CallStats.BeginTimes(2:end) -  CallStats.EndTimes(1:end-1)>prevCalldist)]) = true;





    CallBeg = CallStats.BeginTimes(ismember(CallStats.NoiseType, 'Calls'));
     % CallBeg = CallStats.BeginTimes(ismember(CallStats.NoiseType, baseline_conditions) & ismember(CallStats.Condition, 'Baseline'));


    PB_CallBeg = CallStats.BeginTimes(ismember(CallStats.NoiseType, 'PB'));


    lag = mean(PB_CallBeg(1:numel_PB_calls) - CallBeg(1:numel_PB_calls));

    CallBeg = CallStats.BeginTimes(prevCalldist_condition & ~isINStim & ismember(CallStats.NoiseType, 'Calls'));
        % CallBeg = CallStats.BeginTimes(prevCalldist_condition & ~isINStim );

    CallEnd = CallStats.EndTimes(prevCalldist_condition & ~isINStim & ismember(CallStats.NoiseType, 'Calls'));
    % CallEnd = CallStats.EndTimes(prevCalldist_condition & ~isINStim );
    PB_CallBeg = CallBeg + lag;
    PB_CallEnd = CallEnd + lag;

    all_call_PB = [[CallBeg;PB_CallBeg],[CallEnd;PB_CallEnd]];

    CallLengths = PB_CallEnd-PB_CallBeg;

    [CallLengths, order] = sort(CallLengths);


    CallBeg     = CallBeg(order);
    PB_CallBeg  = PB_CallBeg(order);
    CallEnd     = CallEnd(order);
    PB_CallEnd  = PB_CallEnd(order);

    figure('units','normalized','outerposition',[0 0 .25 1], 'color','white');
    %CAll onset
    subplot(3,2,1)
    psth_onset = zeros(numel(PB_CallBeg),diff(histogram_edges)/bin_size);
    call_n = 1;
    for call_beg = CallBeg'
        this_call_length = CallLengths(call_n);
        spike_this_call       = spikes_time_sec(spikes_time_sec>call_beg+histogram_edges(1) & spikes_time_sec<call_beg+histogram_edges(2)) - call_beg;
        psth_onset(call_n,:)     = histcounts(spike_this_call, bin_edges);
        call_n = call_n+1;
    end
    psth_onset_PB = zeros(numel(PB_CallBeg),diff(histogram_edges)/bin_size);
    call_n = 1;
    for call_beg = PB_CallBeg'
        this_call_length = CallLengths(call_n);
        spike_this_call       = spikes_time_sec(spikes_time_sec>call_beg+histogram_edges(1) & spikes_time_sec<call_beg+histogram_edges(2)) - call_beg;
        psth_onset_PB(call_n,:)     = histcounts(spike_this_call, bin_edges);
        call_n = call_n+1;
    end


    are_there_spikes_PB = sum(psth_onset_PB(:, time2plot>-.20 & time2plot<.2),2)>0;

    call_n = 1;
    are_there_spikes_call = sum(psth_onset(:, time2plot>=-.2 & time2plot<.2),2)>0;

    sub_call2plot = find(are_there_spikes_PB & are_there_spikes_call);;

    for call_beg = CallBeg'
        this_call_length = CallLengths(call_n);
        spike_this_call       = spikes_time_sec(spikes_time_sec>call_beg+histogram_edges(1) & spikes_time_sec<call_beg+histogram_edges(2)) - call_beg;
        psth_onset(call_n,:)     = histcounts(spike_this_call, bin_edges);
        plot(spike_this_call, spike_this_call*0 + call_n, 'k.')
        psth_onset_PB(call_n,:) = histcounts(spike_this_call,bin_edges);
        hold on
        fill([0 this_call_length this_call_length 0], [-.5 -.5 .5 .5]+call_n, 'r','FaceAlpha',.5, 'EdgeColor','none')

        other_calls = find(all_call_PB(:,1)<call_beg+histogram_edges(2) & all_call_PB(:,1)>=call_beg+histogram_edges(1));
        other_calls(all_call_PB(other_calls,1)==call_beg) = [];

        if ~isempty(other_calls)
            for oc = other_calls'
                other_call_start = all_call_PB(oc,1);
                other_call_end  = all_call_PB(oc,2);

                fill([other_call_start other_call_end other_call_end other_call_start]-call_beg, [-.5 -.5 .5 .5]+call_n, 'k','FaceAlpha',.25, 'EdgeColor','none')
            end

        end
        call_n = call_n+1;
    end
    xlim([-.2 .2])
    ylim tight
    title(index_n)

    subplot(3,2,5)
    hold on
    mean_psth_onset = mean(psth_onset)/bin_size;
    plot(time2plot,movmean(mean_psth_onset,5), 'k')




    %Call offset
    subplot(3,2,2)
    psth_offset = zeros(size(ALL_CallBeg_BASE,1),diff(histogram_edges)/bin_size);
    call_n = 1;
    for call_end = CallEnd'
        spike_this_call       = spikes_time_sec(spikes_time_sec>call_end+histogram_edges(1) & spikes_time_sec<call_end+histogram_edges(2)) - call_end;
        this_call_length = CallLengths(call_n);
        plot(spike_this_call, spike_this_call*0 + call_n, 'k.')
        psth_offset(call_n,:) = histcounts(spike_this_call,bin_edges);
        hold on
        fill([0 -this_call_length -this_call_length 0], [-.5 -.5 .5 .5]+call_n, 'r','FaceAlpha',.5, 'EdgeColor','none')

        other_calls = find(all_call_PB(:,1)<call_end+histogram_edges(2) & all_call_PB(:,1)>=call_end+histogram_edges(1));
        other_calls(all_call_PB(other_calls,2)==call_end) = [];

        if ~isempty(other_calls)
            for oc = other_calls'
                other_call_start = all_call_PB(oc,1);
                other_call_end  = all_call_PB(oc,2);

                fill([other_call_start other_call_end other_call_end other_call_start]-call_end, [-.5 -.5 .5 .5]+call_n, 'k','FaceAlpha',.25, 'EdgeColor','none')
            end

        end
        call_n = call_n+1;
    end
    xlim([-.2 .2])
    ylim tight

    subplot(3,2,6)
    hold on
    mean_psth_offset = mean(psth_offset)/bin_size;
    plot(time2plot,movmean(mean_psth_offset,5), 'k')
    xlim([-.2 .2])

    %PB onset
    subplot(3,2,3)
    % psth_onset_PB = zeros(numel(PB_CallBeg),diff(histogram_edges)/bin_size);
    call_n = 1;
    for call_beg = PB_CallBeg'
        this_call_length = CallLengths(call_n);
        spike_this_call       = spikes_time_sec(spikes_time_sec>call_beg+histogram_edges(1) & spikes_time_sec<call_beg+histogram_edges(2)) - call_beg;
        psth_onset_PB(call_n,:)     = histcounts(spike_this_call, bin_edges);
        plot(spike_this_call, spike_this_call*0 + call_n, 'k.')
        psth_offset_PB(call_n,:) = histcounts(spike_this_call,bin_edges);
        hold on
        fill([0 this_call_length this_call_length 0], [-.5 -.5 .5 .5]+call_n, 'r','FaceAlpha',.5, 'EdgeColor','none')

        other_calls = find(all_call_PB(:,1)<call_beg+histogram_edges(2) & all_call_PB(:,1)>=call_beg+histogram_edges(1));
        other_calls(all_call_PB(other_calls,1)==call_beg) = [];

        if ~isempty(other_calls)
            for oc = other_calls'
                other_call_start = all_call_PB(oc,1);
                other_call_end  = all_call_PB(oc,2);

                fill([other_call_start other_call_end other_call_end other_call_start]-call_beg, [-.5 -.5 .5 .5]+call_n, 'k','FaceAlpha',.25, 'EdgeColor','none')
            end

        end
        call_n = call_n+1;
    end
    xlim([-.2 .2])
    ylim tight

    subplot(3,2,5)
    mean_psth_onset_PB = mean(psth_onset_PB)/bin_size;
    plot(time2plot,movmean(mean_psth_onset_PB,5), 'c')
    xlim([-.2 .2])
    y_lim = ylim;
    plot([0 0], y_lim, 'r')


    %PB offset
    subplot(3,2,4)
    psth_offset_PB = zeros(size(ALL_CallBeg_BASE,1),diff(histogram_edges)/bin_size);
    call_n = 1;
    for call_end = PB_CallEnd'
        spike_this_call       = spikes_time_sec(spikes_time_sec>call_end+histogram_edges(1) & spikes_time_sec<call_end+histogram_edges(2)) - call_end;
        this_call_length = CallLengths(call_n);
        plot(spike_this_call, spike_this_call*0 + call_n, 'k.')
        psth_offset_PB(call_n,:) = histcounts(spike_this_call,bin_edges);
        hold on
        fill([0 -this_call_length -this_call_length 0], [-.5 -.5 .5 .5]+call_n, 'r','FaceAlpha',.5, 'EdgeColor','none')

        other_calls = find(all_call_PB(:,1)<call_end+histogram_edges(2) & all_call_PB(:,1)>=call_end+histogram_edges(1));
        other_calls(all_call_PB(other_calls,2)==call_end) = [];

        if ~isempty(other_calls)
            for oc = other_calls'
                other_call_start = all_call_PB(oc,1);
                other_call_end  = all_call_PB(oc,2);

                fill([other_call_start other_call_end other_call_end other_call_start]-call_end, [-.5 -.5 .5 .5]+call_n, 'k','FaceAlpha',.25, 'EdgeColor','none')
            end

        end
        call_n = call_n+1;
    end
    xlim([-.2 .2])
    ylim tight


    subplot(3,2,6)
    mean_psth_offset_PB = mean(psth_offset_PB)/bin_size;
    plot(time2plot,movmean(mean_psth_offset_PB,5), 'c')
    xlim([-.2 .2])
    pause(.1)
end
%% Finding Stims with no call
n_rand                  = 100;
n_calls_per_stim        = cell(4,3);

n_calls_per_stim_ptg    = zeros(4,3);
time_after_stim         = 1;
min_call_latency        = .1;
latency2compare         = .1;
for dsN = [1 2 3 4]


STIM_TABLE          =  ALL_DATA(dsN).STIM_TIMES;
CallStats           = ALL_DATA(dsN).CallStats;
lasts_call          = max(CallStats.BeginTimes);
% STIM_TABLE          = STIM_TABLE(STIM_TABLE.StimStart<lasts_call,:);
nCallsPerStim       = zeros(size(STIM_TABLE,1),2);
next_stim_latency   = zeros(size(STIM_TABLE,1),1);
next_call_latency   = zeros(size(STIM_TABLE,1),1);
call_end_latency    = zeros(size(STIM_TABLE,1),1);

for stim_n =1:size(STIM_TABLE,1)

    stim_start = STIM_TABLE.StimStart(stim_n);
    stim_end = STIM_TABLE.StimEnd(stim_n);
    next_stim = Inf;

    if stim_n<size(STIM_TABLE,1)
        next_stim = STIM_TABLE.StimStart(stim_n+1);
    end

    next_stim_latency(stim_n)   = next_stim-stim_end;
    this_call_latency     = min(CallStats.BeginTimes(CallStats.BeginTimes>=stim_end))-stim_end;
    this_call_end_latency = min(CallStats.EndTimes(CallStats.EndTimes>=stim_end))-stim_end;
    if ~isempty(this_call_latency)
        next_call_latency(stim_n)   = this_call_latency;
    end
    if ~isempty(this_call_end_latency)
        call_end_latency(stim_n)    = this_call_end_latency;
    end
    nCallsPerStim(stim_n,1)     = sum(CallStats.BeginTimes>=stim_end & CallStats.BeginTimes<=stim_start+time_after_stim & CallStats.BeginTimes<next_stim & (next_stim-stim_end>=time_after_stim) );
    nCallsPerStim(stim_n,2)     = sum(CallStats.BeginTimes>=stim_start & CallStats.BeginTimes<=stim_start+time_after_stim & CallStats.BeginTimes<next_stim & (next_stim-stim_end>=time_after_stim));
end

n_calls_per_stim_ptg(dsN,[1 2])= 100*sum(nCallsPerStim==0 & ismember(STIM_TABLE.StimeType, 'Baseline'))/size(nCallsPerStim,1);
n_calls_per_stim_ptg(dsN,3)= size(nCallsPerStim,1);

n_calls_per_stim{dsN,1} = nCallsPerStim;
n_calls_per_stim{dsN,2} = STIM_TABLE.StimeType;
n_calls_per_stim{dsN,3} = next_call_latency;

figure
for type_n=1:4
type = neuron_types{type_n};
onset_index2use  = onset_correlation_range_index;
if type_n==3
    index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Base-PB Onset post'}<0);
    index_list_pos = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Base-PB Onset post'}>0);

    sub_tit_neg = 'ONSET INHIBITED';
    sub_tit_pos = 'ONSET ACTIVATED';
elseif type_n==1
    index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Offset pre'}<0);
    index_list_pos = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN & Index_table{:,'Offset pre'}<0);
    sub_tit_neg = 'RAMPING INHIBITED';
    sub_tit_pos = 'RAMPING ACTIVATED';
elseif type_n==2
    index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN);
    index_list_pos = index_list_neg;
    sub_tit_neg= 'BEFORE';
    onset_index2use = onset_correlation_range_indexBeforeCells;
elseif type_n==4
    index_list_neg = find(ismember(neuron_type_table.Type,type) & table_data.DataSetN==dsN);
    index_list_pos = index_list_neg;
    sub_tit_neg = 'NON RESPONSIVE';
end

spikes_time_sec_neg = double(ALL_DATA(dsN).spike_times(ismember(ALL_DATA(dsN).spike_clusters,table_data.ID(index_list_neg))))/30000;
spikes_time_sec_neg = predict(ALL_DATA(dsN).synch_models.AUDIO_NPX, spikes_time_sec_neg);


spikes_time_sec_pos = double(ALL_DATA(dsN).spike_times(ismember(ALL_DATA(dsN).spike_clusters,table_data.ID(index_list_pos))))/30000;
spikes_time_sec_pos = predict(ALL_DATA(dsN).synch_models.AUDIO_NPX, spikes_time_sec_pos);

if dsN<4

stim_list_no_call_after     = find(nCallsPerStim(:,2)==0 & ismember(STIM_TABLE.StimeType, 'Baseline') & next_stim_latency>=time_after_stim);
stim_list_with_call_after = find(nCallsPerStim(:,2)>0 & ismember(STIM_TABLE.StimeType, 'Baseline') & next_stim_latency>=time_after_stim & next_call_latency>=min_call_latency & call_end_latency>=min_call_latency);
else

stim_list_no_call_after     = find(nCallsPerStim(:,2)==0  & next_stim_latency>=time_after_stim);
stim_list_with_call_after = find(nCallsPerStim(:,2)>0  & next_stim_latency>=time_after_stim & next_call_latency>=min_call_latency & call_end_latency>=min_call_latency);
end

psth_no_Call_pos = zeros(numel(stim_list_no_call_after),numel(bin_edges)-1 );
psth_no_Call_neg = zeros(numel(stim_list_no_call_after),numel(bin_edges)-1 );


for stim_n = 1:numel(stim_list_no_call_after)
    stim_index = stim_list_no_call_after(stim_n);

    stim_end = STIM_TABLE.StimEnd(stim_index);

    this_stim_spikes = spikes_time_sec_pos(spikes_time_sec_pos>=stim_end+histogram_edges(1) & spikes_time_sec_pos<=stim_end++histogram_edges(2))-stim_end;
    psth_no_Call_pos(stim_n,:)     = histcounts(this_stim_spikes, bin_edges);


    this_stim_spikes = spikes_time_sec_neg(spikes_time_sec_neg>=stim_end+histogram_edges(1) & spikes_time_sec_neg<=stim_end++histogram_edges(2))-stim_end;
    psth_no_Call_neg(stim_n,:)     = histcounts(this_stim_spikes, bin_edges);   

end



psth_with_Call_pos = zeros(numel(stim_list_with_call_after),numel(bin_edges)-1 );
psth_with_Call_neg = zeros(numel(stim_list_with_call_after),numel(bin_edges)-1 );


for stim_n = 1:numel(stim_list_with_call_after)
    stim_index = stim_list_with_call_after(stim_n);

    stim_end = STIM_TABLE.StimEnd(stim_index);

    this_stim_spikes = spikes_time_sec_pos(spikes_time_sec_pos>=stim_end+histogram_edges(1) & spikes_time_sec_pos<=stim_end++histogram_edges(2))-stim_end;
    psth_with_Call_pos(stim_n,:)     = histcounts(this_stim_spikes, bin_edges);


    this_stim_spikes = spikes_time_sec_neg(spikes_time_sec_neg>=stim_end+histogram_edges(1) & spikes_time_sec_neg<=stim_end++histogram_edges(2))-stim_end;
    psth_with_Call_neg(stim_n,:)     = histcounts(this_stim_spikes, bin_edges);   

end


mean_distribution_pos = zeros(n_rand,numel(bin_edges)-1 );

for j=1:n_rand
    samples = randsample(size(psth_with_Call_pos,1),size(psth_no_Call_pos,1));
    mean2plot = movmean(mean(psth_with_Call_pos(samples,:)),5);
    mean_distribution_pos(j,:) = mean2plot;
end


mean_distribution_neg = zeros(n_rand,numel(bin_edges)-1 );

for j=1:n_rand
    samples = randsample(size(psth_with_Call_neg,1),size(psth_no_Call_neg,1));
    mean2plot = movmean(mean(psth_with_Call_neg(samples,:)),5);
    mean_distribution_neg(j,:) = mean2plot;
end


subplot(4,2,1 + (type_n-1)*2)
mean2plot = mean(mean_distribution_pos);
% std2plot196 = 1.96*std(mean_distribution);
ci = prctile(mean_distribution_pos, [2.5 97.5]);
hold on
fill([time2plot fliplr(time2plot)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
ci = prctile(mean_distribution_pos, [10 90]);
hold on
fill([time2plot fliplr(time2plot)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
plot(time2plot, mean2plot, 'r')

hold on
plot(time2plot, movmean(mean(psth_no_Call_pos),5), 'k')
xlim([-.1 latency2compare])
ylim  tight
y_lim = ylim;
fill([-.1 0 0 -.1], [0 0 y_lim(2) y_lim(2)], 'k')

    title(sub_tit_pos)



subplot(4,2,2 + (type_n-1)*2)
mean2plot = mean(mean_distribution_neg);
% std2plot196 = 1.96*std(mean_distribution);
ci = prctile(mean_distribution_pos, [2.5 97.5]);
hold on
fill([time2plot fliplr(time2plot)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
ci = prctile(mean_distribution_neg, [10 90]);
hold on
fill([time2plot fliplr(time2plot)], [ci(1,:) fliplr(ci(2,:))], 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')
plot(time2plot, mean2plot, 'r')

hold on
plot(time2plot, movmean(mean(psth_no_Call_neg),5), 'k')
xlim([-.1 latency2compare])
ylim  tight
y_lim = ylim;
fill([-.1 0 0 -.1], [0 0 y_lim(2) y_lim(2)], 'k')

    title(sub_tit_neg)
end

end