
clc
clear
pathIn='D:\drum_ta\data\preprocessed_v2';
pathOutTa='D:\drum_ta\data\epoched_v2\ta\';
pathOutDrum='D:\drum_ta\data\epoched_v2\drum\';

cd ('D:\drum_ta\data\preprocessed_v2');
EEG_EGI = uipickfiles('FilterSpec', '*.set','Prompt','Select the .set EGI data files');
nID=size(EEG_EGI,2); %number of files loaded

eeglab
%% loop through files
for pp=1:nID
    % load data
    fname=EEG_EGI{1,pp};
    [filepath,name,ext] = fileparts(EEG_EGI{1,pp});
    subjectName =EEG_EGI{1,pp};
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    
    EEG = pop_loadset(subjectName);
    
    %     try
    for ev=1:length(EEG.event)
        if strcmp(EEG.event(ev).type, 'TRSP')%find the 'TRSP'
            tempev=1;
            try
                while ~strcmp(EEG.event(ev+tempev).type, 'TRSP') & ev+tempev<length(EEG.event)
                    if strcmp(EEG.event(ev+tempev).type, 'DIN4')
                        EEG.event(ev+tempev).test=EEG.event(ev).test
                        EEG.event(ev+tempev).cell=EEG.event(ev).cell
                        EEG.event(ev+tempev).obss=EEG.event(ev).obss
                        
                    end
                    tempev=tempev+1;
                end
            end
            switch EEG.event(ev).test
                
                case 0
                    EEG.event(ev).type='TRS0';
                case 1
                    EEG.event(ev).type='TRS1';
                case 2
                    EEG.event(ev).type='TRS2';
                    
                case 3
                    EEG.event(ev).type='TRS3';
                    
                    
            end
            
        else
            switch EEG.event(ev).test %Change the DIN to a DIN that repreets what it is
                
                case 0
                    EEG.event(ev).type='temp';
                    
                case 1
                    EEG.event(ev).type='DINV';
                    
                case 2
                    EEG.event(ev).type='DIND';
                    
                case 3
                    EEG.event(ev).type='DINR';
                    
                otherwise
                    continue
            end
        end
        
        
    end
    %% find the Ta's
    A_cell = struct2cell(EEG.event);%make structure a cell array
    A_cell=(A_cell(1,1,:));%make structure a cell array
    A_cell=squeeze(A_cell);%remove a dimension
    
    idx=all(ismember(A_cell,'DINV'),2);%logical index all the ta DIN's in the eeg.event
    ind2=find(idx);%find the location of the Ta DIN's
    ind3=ind2(1:4:length(ind2));%take every 4th DIN (this will be ~2seconds)
    %this loop renames every 4th DIN (so that we can epoch it in  EEGlab)
    for i=1:length(ind3)
        b=ind3(i);
        EEG.event(b).type = 'TTTT';
    end
    
    %% epoch the Ta's
    EEGT = pop_epoch( EEG, { 'TTTT' }, [0 2], 'newname', 'EGI file epochs', 'epochinfo', 'yes');
    eeglab redraw
    %% save the Ta's
    all_data=EEGT.data;
    save([pathOutTa,name, '_Ta.mat'], 'all_data', 'fname');
    
    clear A_cell idx ind2 ind3 all_data
    %% find the Drums (see the Ta loop to explain what each line does)
    A_cell = struct2cell(EEG.event);
    A_cell=(A_cell(1,1,:));
    A_cell=squeeze(A_cell);
    
    idx=all(ismember(A_cell,'DIND'),2);
    ind2=find(idx);
    ind3=ind2(1:4:length(ind2));
    
    for i=1:length(ind3)
        b=ind3(i);
        EEG.event(b).type = 'DDDD';
    end
    
    
    %% epoch the Drum
    EEGD = pop_epoch( EEG, { 'DDDD' }, [0 2], 'newname', 'EGI file epochs', 'epochinfo', 'yes');
    eeglab redraw
    %% save the drum
    all_data=EEGD.data;
    save([pathOutDrum,name, '_Drum.mat'], 'all_data', 'fname');
    
    clear A_cell idx ind2 ind3 EEGD EEGT all_data
end