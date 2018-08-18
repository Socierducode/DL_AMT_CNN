# DL_AMT_CNN
Deep learning for Piano Automatic Music Transcription task (CNN demo)

A LeNet-like code for AMT is implemented on Pytorch+Anaconda 2+Ubuntu plateform

Some dependencies are required to install to run this code:  librosa,pretty_midi


--Ask and download MAPS dataset from the following site:

<MAPS dataset> http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/
  
--concatenate the MAPS downloaded pieces using 'cat name block1 block2> MAPS.zip'

--unzip MAPS.zip with p7zip (MAPS volume>4G)

-- prepare the MAPS dataset: excute the MAPS_prepare.py
MAPS_prepare.py includes following function
-unzip the piano model list (must be defined) to a temporary directory
-make a train/val/test dataset
-convert the wav/midi files into CQT spectrum(X) and pianoroll(Y)　respectively generated by librosa CQT module and pretty midi get_piano_roll module
  
To understand more about the data preparation, see the librosa_example.py
 
--run Train.py for training the network 
