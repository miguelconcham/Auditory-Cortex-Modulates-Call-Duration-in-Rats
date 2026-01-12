%% LOAD DATASET
disp('############################')
disp('############################')
disp('SECTION # 1')
data_Set_list = dir('*DataSet.mat');


load(data_Set_list(1).name)
DataSet.Experiment =str2double(data_Set_list(1).name(3:10));
ALL_DATA(1) = DataSet;

for file_n=1:numel(data_Set_list)

    load(data_Set_list(file_n).name)
    DataSet.Experiment = str2double(data_Set_list(file_n).name(3:10));

    list_of_fields = fieldnames(ALL_DATA);
    new_fields = fieldnames(DataSet);

    for j = find(~ismember(list_of_fields, new_fields))'

        DataSet.(list_of_fields{j}) = [];
    end

    for j = find(~ismember(new_fields,list_of_fields))'

       ALL_DATA(1).(new_fields{j}) = [];
    end
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
    if ~isempty(NeuropixelsDepth) 
        [CortexChann, ~]    = assignBrainArea(NeuropixelsDepth);
        ChannDepth         = assign_depth(NeuropixelsDepth,ResponseTypes.Channel);
        ResponseTypes.BrainArea(ResponseTypes.Channel<CortexChann) = {'SubCortical'};
        ResponseTypes.NeuronDepth = ChannDepth;
    else
        disp(['NO NeuropixelsDepth for ', num2str(data_set_n)])
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

%% SET PARAMETERS


next_stim =1;
onset_correlation_range             = [-.1 0];
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



%%  ESTIMATE of ONSET AND OFFSET to call
ONSET_OFFSET_PSTH = cell(1,5);

index_list =  1:size(table_data,1);
% index_list = find(ismember(neuron_type_table.SecondType,type))';
psth_onset_all_neuron      = zeros(numel(index_list),diff(histogram_edges)/bin_size);
psth_offset_all_neuron     = zeros(numel(index_list),diff(histogram_edges)/bin_size);
offset_index               = zeros(numel(index_list),2);
onset_index                = zeros(numel(index_list),2);
rate_index                 = zeros(numel(index_list),1);

neuron_n = 1;
for index_n =index_list

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
    psth_onset_all_neuron(neuron_n,:)     = psth2smooth;


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
    psth_offset_all_neuron(neuron_n,:)    = smoothed_psth;




    offset_index(neuron_n,[1 2])    = [mean(psth_offset_all_neuron(neuron_n,pre_index))  mean(psth_offset_all_neuron(neuron_n,post_index))];
    onset_index(neuron_n, [1 2])    = [mean(psth_onset_all_neuron(neuron_n,pre_index))  mean(psth_onset_all_neuron(neuron_n,post_index))];

    neuron_n = neuron_n+1;
end

ONSET_OFFSET_PSTH{1} = psth_onset_all_neuron;
ONSET_OFFSET_PSTH{2} = psth_offset_all_neuron;
ONSET_OFFSET_PSTH{3} = onset_index;
ONSET_OFFSET_PSTH{4} = offset_index;
ONSET_OFFSET_PSTH{5}  = rate_index;






%% ESTIMATE PB (and plot)


figure('units','normalized','outerposition',[0.33 0 .33 1]);
ONSET_OFFSET_PSTH_PB = cell(1,2);
colormap(color_map)
index_list =  1:size(table_data,1);
% index_list = find(ismember(neuron_type_table.SecondType,type))';
psth_onset_PB_all_neurons   = zeros(numel(index_list),diff(histogram_edges)/bin_size);
psth_offset_PB_all_neurons  = zeros(numel(index_list),diff(histogram_edges)/bin_size);

neuron_n = 1;
for index_n = index_list

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
    psth_onset_PB_all_neurons(neuron_n,:)     = psth2smooth;


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
    psth_offset_PB_all_neurons(neuron_n,:)    = smoothed_psth;


    neuron_n = neuron_n+1;
end



offset_pre          = (ONSET_OFFSET_PSTH{4}(:,1) - mean(ONSET_OFFSET_PSTH{4}(:,1),'omitmissing'))/std(ONSET_OFFSET_PSTH{4}(:,1),'omitmissing');
offset_post         = (ONSET_OFFSET_PSTH{4}(:,2) - mean(ONSET_OFFSET_PSTH{4}(:,2),'omitmissing'))/std(ONSET_OFFSET_PSTH{4}(:,1),'omitmissing');
offset_RI           = -offset_post+offset_pre; %offset post - offset  pre %check sign

ONSET_OFFSET_PSTH_PB{1} = psth_onset_PB_all_neurons;
ONSET_OFFSET_PSTH_PB{2} = psth_offset_PB_all_neurons;

%% estiamte response indexes and classify

response_indexes = nan(size(psth_onset_all_neuron,1),4);

for cn = 1:size(psth_onset_all_neuron,1)
    data2zscore_baseline = psth_onset_all_neuron( cn,:);
   

    response_indexes(cn,1) = mean(data2zscore_baseline(pre_index));
    response_indexes(cn,3) = mean(data2zscore_baseline(post_index));


    data2zscore_PB = psth_onset_PB_all_neurons( cn,:);



    response_indexes(cn,2) = mean(data2zscore_baseline(pre_index)-data2zscore_PB(pre_index));
    response_indexes(cn,4) = mean(data2zscore_baseline(post_index)-data2zscore_PB(post_index));

end

%% select only cortical neurons

neurons2classify = ismember(table_data.BrainArea, 'Cortex') &  ismember(table_data.ClusterQuality,{'Mua','Good'});
response_indexes = response_indexes(neurons2classify,:);

offset_indexes = [offset_RI offset_pre offset_post];
offset_indexes = offset_indexes(neurons2classify,:);

%% loading neuron types

neuron_types = {'A', 'B', 'D', {'N', 'T'}};
neuron_types_table = readtable('NeuronTypesAfterRevision.xlsx');

non_responsive_neurons = ismember(neuron_types_table.Type, neuron_types{4});

%% plot response index (Iteration =0)

figure

[order_pb_response, order] = sort(response_indexes(:,end));


std_tresh = 2;
axis_lim_onset  = [-12 12 -12 30];
axis_lim_offset = [-8 8 -8 14];

responsive_cells_onset        = false(size(response_indexes,1),1);
non_responsive_onset          = true(size(response_indexes,1),1);

responsive_cells_offset        = false(size(response_indexes,1),1);
non_responsive_offset          = true(size(response_indexes,1),1);

onset_pb_index          = (response_indexes(:,end) -mean(response_indexes(non_responsive_onset,end), 'omitmissing'))/std(response_indexes(non_responsive_onset,end), 'omitmissing');
onset_index             = (response_indexes(:,3) -mean(response_indexes(non_responsive_onset,3), 'omitmissing'))/std(response_indexes(non_responsive_onset,3), 'omitmissing');
remaining_responsive_onset   = abs(onset_index)>std_tresh | abs(onset_pb_index)>std_tresh & non_responsive_onset;
responsive_cells_onset        = responsive_cells_onset | remaining_responsive_onset;
non_responsive_onset          = ~responsive_cells_onset;



offset_ri           = (offset_indexes(:,1) - mean(offset_indexes(responsive_cells_offset,1), 'omitmissing'))/std(offset_indexes(responsive_cells_offset,1), 'omitmissing');
offset_rate_pre     = (offset_indexes(:,2) - mean(offset_indexes(responsive_cells_offset,2), 'omitmissing'))/std(offset_indexes(responsive_cells_offset,2), 'omitmissing');
remaining_responsive_offset = (abs(offset_ri)>std_tresh | abs(offset_rate_pre)>std_tresh) & non_responsive_offset;
responsive_cells_offset = responsive_cells_offset | remaining_responsive_offset;
non_responsive_offset   = ~responsive_cells_offset;


  subplot(2,1,1)
plot(onset_pb_index, onset_index, 'k.')
hold on
plot(onset_pb_index(responsive_cells_onset), onset_index(responsive_cells_onset), 'g.', 'MarkerSize',16)
axis(axis_lim_onset);
plot(axis_lim_onset([1 2]), [std_tresh std_tresh], ':k')
plot(axis_lim_onset([1 2]), -[std_tresh std_tresh], ':k')
plot( [std_tresh std_tresh], axis_lim_onset([1 2]+2),':k')
plot( -[std_tresh std_tresh],axis_lim_onset([1 2]+2), ':k')



subplot(2,1,2)
hold off
plot(offset_ri, offset_rate_pre, 'k.')
hold on
plot(offset_ri(responsive_cells_onset), offset_rate_pre(responsive_cells_onset), 'g.', 'MarkerSize',16)
axis(axis_lim_offset);
plot(axis_lim_offset([1 2]), [std_tresh std_tresh], ':k')
plot(axis_lim_offset([1 2]), -[std_tresh std_tresh], ':k')
plot( [std_tresh std_tresh], axis_lim_offset([1 2]+2),':k')
plot( -[std_tresh std_tresh],axis_lim_offset([1 2]+2), ':k')


%% iterate classification until convergence



while any(remaining_responsive_onset) || any(remaining_responsive_offset)
    
    onset_pb_index      = (response_indexes(:,end) -mean(response_indexes(non_responsive_onset,end), 'omitmissing'))/std(response_indexes(non_responsive_onset,end), 'omitmissing');
    onset_index         = (response_indexes(:,3) -mean(response_indexes(non_responsive_onset,3), 'omitmissing'))/std(response_indexes(non_responsive_onset,3), 'omitmissing');
    max_value = max(max(abs(onset_pb_index)), max(abs(onset_index)));
    % axis_lim = [-max_value max_value -max_value max_value];

           % = (offset_indexes(:,1) - mean(offset_indexes(non_responsive_offset,1), 'omitmissing'))/std(offset_indexes(non_responsive_offset,1), 'omitmissing');
    % try the former line agian but redifn
    offset_rate_pre     = (offset_indexes(:,2) - mean(offset_indexes(non_responsive_offset,2), 'omitmissing'))/std(offset_indexes(non_responsive_offset,2), 'omitmissing');
    offset_rate_post     = (offset_indexes(:,3) - mean(offset_indexes(non_responsive_offset,3), 'omitmissing'))/std(offset_indexes(non_responsive_offset,3), 'omitmissing');
    offset_ri           = offset_rate_post-offset_rate_pre; %check sign
    remaining_responsive_onset = (abs(onset_index)>std_tresh | abs(onset_pb_index)>std_tresh) & non_responsive_onset;
    responsive_cells_onset = responsive_cells_onset | remaining_responsive_onset;
    non_responsive_onset   = ~responsive_cells_onset;



    remaining_responsive_offset = (abs(offset_ri)>std_tresh | abs(offset_rate_pre)>std_tresh) & non_responsive_offset ;
    responsive_cells_offset = (responsive_cells_offset | remaining_responsive_offset)  ;
    non_responsive_offset   = ~responsive_cells_offset;

    responsive_cells_offset = responsive_cells_offset;
    subplot(2,1,1)
       hold off
    plot(onset_pb_index, onset_index, 'k.')
    hold on
    plot(onset_pb_index(responsive_cells_onset), onset_index(responsive_cells_onset), 'g.', 'MarkerSize',16)
    axis(axis_lim_onset);
    plot(axis_lim_onset([1 2]), [std_tresh std_tresh], ':k')
    plot(axis_lim_onset([1 2]), -[std_tresh std_tresh], ':k')
    plot( [std_tresh std_tresh], axis_lim_onset([1 2]+2),':k')
    plot( -[std_tresh std_tresh],axis_lim_onset([1 2]+2), ':k')


    subplot(2,1,2)
    hold off
    plot(offset_ri, offset_rate_pre, 'k.')
    hold on
    plot(offset_ri(responsive_cells_offset), offset_rate_pre(responsive_cells_offset), 'g.', 'MarkerSize',16)
    axis(axis_lim_offset);
    plot(axis_lim_offset([1 2]), [std_tresh std_tresh], ':k')
    plot(axis_lim_offset([1 2]), -[std_tresh std_tresh], ':k')
    plot( [std_tresh std_tresh], axis_lim_offset([1 2]+2),':k')
    plot( -[std_tresh std_tresh],axis_lim_offset([1 2]+2), ':k')
    

    pause(.1)
end

    responsive_cells_offset = responsive_cells_offset & ~responsive_cells_onset;

    subplot(2,1,2)
    hold off
    plot(offset_ri, offset_rate_pre, 'k.')
    hold on
    plot(offset_ri(responsive_cells_offset), offset_rate_pre(responsive_cells_offset), 'g.', 'MarkerSize',16)
    axis(axis_lim_offset);
    plot(axis_lim_offset([1 2]), [std_tresh std_tresh], ':k')
    plot(axis_lim_offset([1 2]), -[std_tresh std_tresh], ':k')
    plot( [std_tresh std_tresh], axis_lim_offset([1 2]+2),':k')
    plot( -[std_tresh std_tresh],axis_lim_offset([1 2]+2), ':k')
    
