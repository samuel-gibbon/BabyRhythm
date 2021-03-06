# BabyRhythm
These are the scripts associated with the paper

Gibbon, S., Attaheri, A., Choisdealbha, Á.N., Rocha, S., Brusini, P., Mead, N., Boutris, P., Olawole-Scott, H., Ahmed, H., Flanagan, S. and Mandke, K., 2021. Machine learning accurately classifies neural responses to rhythmic speech vs. non-speech from 8-week-old infant EEG. Brain and Language, 220, p.104968.

The data is available on request from Prof Usha Goswami (Centre for Neuroscience in Education, University of Cambridge). Due to the sensitive nature of the data, all requests will go through a data access committee, and will in most cases only be granted to researchers affiliated with a higher education institution.

EEG preprocessing is done in MATLAB, and Deep Learning (DL) is done in Python. For researchers interested in reproducing the DL results, you should request the "preprocessed EEG data", these files are relatively small. For researchers interested in reproducing the EEG preprocessing, you should request the "raw EEG", these files are much larger. NB: Data requests will only be considered upon completion of the BabyRhythm project - sometime in 2021.

Some of the MATLAB scripts may be tricky for people outside the project to follow, and the team can offer guidance upon request for those wishing to replicate the results. However, the scripts "preprocess.m" and "reject_epochs.m" detail the parameters we used for interpolating bad channels and rejecting bad epochs, which may be useful for EEG researchers who wish to know the specifics, but do not wish to actually run the scripts.

Any issues/questions please get in touch!
