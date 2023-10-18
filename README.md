###                  ###
DEMONSTRATION REPOSITORY
###                  ###

Demonstration repo of codes used to train an anomaly detector.

Person responsible: sonja.gronroos@cern.ch // smgronroos@gmail.com

The repository contains the Python scripts used for developing the anomaly detector for 8inch HGCAL silicon sensor and partials.
By running the algorithm with new scan mappings, also other sensor geometries could be inspected.
The anomaly detector is an ensemble of deep CNNs (referred to as the pre-selection algorithm, or PS). 
The images produced by the scan program are analysed by the PS, and those images that are flagged to potentially 
contain anomalies are indicated to the inspector so that the anomalous area can be inspected and potentially cleaned. 

The accuracy of the developed model was shown to be ~0.9. The accuracy is expected to increase if the model is trained further with new data. 
Images determined to be normal by the PS can be removed and only those containing anomalies are saved for future use in re-training the model. 
Only the anomalous images need to be saved for re-training purposes as data is heavily imbalanced and lacks in anomalous areas.  

### Additional resources
Video demonstration of a scan of a (very dirty) sensor:  
[Link to Youtube](https://youtu.be/5Pm26gaBaMA)

Publication documenting the results at HGCAL lab at CERN:  
[Link to paper](https://iopscience.iop.org/article/10.1088/2632-2153/aced7e)

