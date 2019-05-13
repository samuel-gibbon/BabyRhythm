% PREPROCESS (filter, downsample, insert events, add chanlocs, interpolate
% bad chans, rereference)

clc
clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

re_sample=1;
detrend=0;
pathIn='D:\drum_ta\data\raw\';
cd 'D:\drum_ta';
FileList= uipickfiles('FilterSpec', '*.raw','Prompt','Select the raw files');
nID=size(FileList,2); % number of files loaded
pathoutset='D:\drum_ta\data\preprocessed_v2\';

%%
for i =1:length(FileList)
    i
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % opens new eeglab
    [filepath,name,ext] = fileparts(FileList{1,i});
    subjectName = [filepath '\' name]; % [pathin  ssPath(t).name '/' ssList(i).name]
    subjectEvent= strcat( name, '.evt'); % download the evt file (csv file)
    setNamex = strcat(name,'.set'); % new name to save the preprocessed data
    setNamex2 = strcat(name,'2.set'); % new name to save the preprocessed data
    filt_setNamex = strcat(name,'_Filt.set'); % new name to save the preprocessed data
    matNamex = strcat(subjectName,'.mat'); % if needed
    
    %% load data
    EEG = pop_readegi([filepath,'\',name,ext]);
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    
    %% add details to the EEG structure
    EEG.filename=subjectName;
    EEG.filepath=filepath;
    EEG.setname=setNamex;
    
    %% filter
    EEG = pop_eegfiltnew(EEG, 0.5,45,6600,0,[],1);   
    
    %% downsample 
    EEG = pop_resample( EEG, 100); % consider downsamling after epoching, and rejecting
    
    %% detrend, consider doing it, due to pink noise (1/f, or aperiodic)
    if detrend==1
        y=nt_detrend(x,1); 
    else
    end

    %% insert events
    subjectName
    EEG=Insert_Event(EEG, subjectEvent, [filepath,'\']);
    EEG = pop_saveset( EEG, setNamex, pathoutset);
    
    %% if 65 channels this removes chan 65
    EEG = pop_select( EEG,'nochannel',65);
    
     %% add channel locations
    chanlocs='D:\hackathon\matlab\eeglab_current\channel64.xyz';
    EEG=pop_chanedit(EEG,  'load',{chanlocs, 'filetype', 'autodetect'});
    EEG = pop_select( EEG,'nochannel',[61 62 63 64]);
    
    %% Regect + Interpolate Via Probability and Kertosis (consider, epoch, reject bad epochs liberally
    % then interpolate epoch by epoch, then reref
    [EEG2,indelec] = pop_rejchan(EEG, 'elec',[1:60] ,'threshold',2,'norm','on','measure','prob');
    [EEG3,indelec2] = pop_rejchan(EEG, 'elec',[1:60] ,'threshold',2,'norm','on','measure','kurt');
    % this code collates the channels id-ed by the 2 rejection criteria
    rej=[indelec,indelec2];
    rej=sort(rej);
    rej=unique(rej);
    % intopolates the channels id-ed by the 2 rejection criteria
    EEG = eeg_interp(EEG, rej);
    
    %% ReRef
    EEG = pop_reref( EEG, []);
    
    %% save
    EEG = eeg_checkset( EEG );
    
    %% saveset
    EEG = pop_saveset( EEG, setNamex, pathoutset);
    all_rej(1,i)=size(rej,2);
    clear EEG2 EEG3
    close all
end
save([pathoutset,'\rejection'],'all_rej', 'FileList')