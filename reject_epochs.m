clc
clear
cd 'D:\drum_ta\data\epoched_v2';
FileList= uipickfiles('FilterSpec', '*.mat','Prompt','Select the preprossesed .mat files');
nID=size(FileList,2); % number of files loaded
PathIn='D:\drum_ta\data\epoched_v2\ta\'
pathoutset='D:\drum_ta\data\epoched_v2\bad_epochs_rejected\ta\';
eeglab

for pp=1:nID;
 name=(FileList{1,pp}); 
 load(name);
 EEG.data=all_data;
 EEG.srate=100;

 [filepath,name,ext] = fileparts(FileList{1,pp});
  EEG.setname=name

EEG = eeg_checkset( EEG );
EEG2 = pop_autorej(EEG, 'nogui','on','startprob',3,'eegplot','on','maxrej',15);
eeglab redraw

all_data=EEG2.data;

save([pathoutset,name,'_epoch_rej.mat'], 'all_data');

clear all_data
end