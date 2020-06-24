% remove 4 strange channels

clc
clear
cd 'D:\drum_ta\data\epoched_v2';
FileList= uipickfiles('FilterSpec', '*.mat','Prompt','Select the preprossesed .mat files');
nID=size(FileList,2); % number of files loaded
PathIn='D:\drum_ta\data\epoched_v2\drum\'
pathoutset='D:\drum_ta\data\epoched_v2\bad_epochs_included\ta\';
eeglab
rejected_trials = []

for pp=1:nID;
 name=(FileList{1,pp}); 
 load(name);
 EEG.data=all_data;
 EEG.srate=100;
 trials_before = EEG.trials;

 [filepath,name,ext] = fileparts(FileList{1,pp});
  EEG.setname=name

EEG = eeg_checkset( EEG );
EEG = pop_select( EEG,'nochannel',[61 62 63 64]);
trials_before = EEG.trials;
%EEG2 = pop_autorej(EEG, 'nogui','on','startprob',3,'eegplot','off','maxrej',5);
eeglab redraw

all_data=EEG.data;
trials_after = EEG.trials;
decrease = (trials_before - trials_after);
percentage_trials_removed = (decrease / trials_before * 100);
%append(rejected_trials, percentage_trials_removed);
rejected_trials=[rejected_trials;percentage_trials_removed];
save([pathoutset,name,'4chan_rem.mat'], 'all_data');

clear all_data
end