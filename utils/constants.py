
def max_seq_len(dataset):
    if dataset == "50words":
        return 270
    if dataset == "Adiac":
        return 176
    if dataset == "ArrowHead":
        return 251
    if dataset == "Beef":
        return 470
    if dataset == "BeetleFly":
        return 512
    if dataset == "BirdChicken":
        return 512
    if dataset == "Car":
        return 577
    if dataset == "CBF":
        return 128
    if dataset == "ClorineConcentration":
        return 166
    if dataset == "CinC_ECG_torso":
        return 1639
    if dataset == "Coffee":
        return 286
    if dataset == "Computers":
        return 720
    if dataset == "Cricket_X":
        return 300
    if dataset == "Cricket_Y":
        return 300
    if dataset == "Cricket_Z":
        return 300
    if dataset == "DiatomSizeReduction":
        return 345
    if dataset == "DistalPhalanxOutlineAgeGroup":
        return 80
    if dataset == "DistalPhalanxOutlineCorrect":
        return 80
    if dataset == "DistalPhalanxTW":
        return 80
    if dataset == "Earthquakes":
        return 512
    if dataset == "ECG200":
        return 96
    if dataset == "ECG5000":
        return 140
    if dataset == "ECGFiveDays":
        return 136
    if dataset == "ElectricDevices":
        return 96
    if dataset == "FaceAll":
        return 131
    if dataset == "FaceFour":
        return 350
    if dataset == "FacesUCR":
        return 131
    if dataset == "FISH":
        return 463
    if dataset == "FordA":
        return 500
    if dataset == "FordB":
        return 500
    if dataset == "Gun_Point":
        return 150
    if dataset == "Ham":
        return 431
    if dataset == "HandOutlines":
        return 2709
    if dataset == "Haptics":
        return 1092
    if dataset == "Herring":
        return 512
    if dataset == "InlineSkate":
        return 1882
    if dataset == "InsectWingbeatSound":
        return 256
    if dataset == "ItalyPowerDemand":
        return 24
    if dataset == "LargeKitchenAppliances":
        return 720
    if dataset == "Lightning2":
        return 637
    if dataset == "Lightning7":
        return 319
    if dataset == "MALLAT":
        return 1024
    if dataset == "Meat":
        return 448
    if dataset == "MedicalImages":
        return 99
    if dataset == "MiddlePhalanxOutlineAgeGroup":
        return 80
    if dataset == "MiddlePhalanxOutlineCorrect":
        return 80
    if dataset == "MiddlePhalanxTW":
        return 80
    if dataset == "MoteStrain":
        return 84
    if dataset == "NonInvasiveFatalECG_Thorax1":
        return 750
    if dataset == "NonInvasiveFatalECG_Thorax2":
        return 750
    if dataset == "OliveOil":
        return 570
    if dataset == "OSULeaf":
        return 427
    if dataset == "PhalangesOutlinesCorrect":
        return 80
    if dataset == "Phoneme":
        return 1024
    if dataset == "Plane":
        return 144
    if dataset == "ProximalPhalanxOutlineAgeGroup":
        return 80
    if dataset == "ProximalPhalanxOutlineCorrect":
        return 80
    if dataset == "ProximalPhalanxTW":
        return 80
    if dataset == "RefrigerationDevices":
        return 720
    if dataset == "ScreenType":
        return 720
    if dataset == "ShapeletSim":
        return 500
    if dataset == "ShapesAll":
        return 512
    if dataset == "SmallKitchenAppliances":
        return 720
    if dataset == "SonyAIBORobotSurfaceII":
        return 65
    if dataset == "SonyAIBORobotSurface":
        return 70
    if dataset == "StarLightCurves":
        return 1024
    if dataset == "Strawberry":
        return 235
    if dataset == "SwedishLeaf":
        return 128
    if dataset == "Symbols":
        return 398
    if dataset == "synthetic_control":
        return 60
    if dataset == "ToeSegmentation1":
        return 277
    if dataset == "ToeSegmentation2":
        return 343
    if dataset == "Trace":
        return 275
    if dataset == "TwoLeadECG":
        return 82
    if dataset == "Two_Patterns":
        return 128
    if dataset == "uWaveGestureLibrary_X":
        return 315
    if dataset == "uWaveGestureLibrary_Y":
        return 315
    if dataset == "uWaveGestureLibrary_Z":
        return 315
    if dataset == "UWaveGestureLibraryAll":
        return 945
    if dataset == "wafer":
        return 152
    if dataset == "Wine":
        return 234
    if dataset == "WordSynonyms":
        return 270
    if dataset == "Worms":
        return 900
    if dataset == "WormsTwoClass":
        return 900
    if dataset == "yoga":
        return 426
    exit('missing dataset')

def nb_classes(dataset):
    if dataset == "50words":
        return 50 #270
    if dataset == "Adiac":
        return 37 #176
    if dataset == "ArrowHead":
        return 3 #251
    if dataset == "Beef":
        return 5 #470
    if dataset == "BeetleFly":
        return 2 #512
    if dataset == "BirdChicken":
        return 2 #512
    if dataset == "Car":
        return 4 #577
    if dataset == "CBF":
        return 3 #128
    if dataset == "ClorineConcentration":
        return 3 #166
    if dataset == "CinC_ECG_torso":
        return 4 #1639
    if dataset == "Coffee":
        return 2 #286
    if dataset == "Computers":
        return 2 #720
    if dataset == "Cricket_X":
        return 12 #300
    if dataset == "Cricket_Y":
        return 12 #300
    if dataset == "Cricket_Z":
        return 12 #300
    if dataset == "DiatomSizeReduction":
        return 4 #345
    if dataset == "DistalPhalanxOutlineAgeGroup":
        return 3 #80
    if dataset == "DistalPhalanxOutlineCorrect":
        return 2 #80
    if dataset == "DistalPhalanxTW":
        return 6 #80
    if dataset == "Earthquakes":
        return 2 #512
    if dataset == "ECG200":
        return 2 #96
    if dataset == "ECG5000":
        return 5 #140
    if dataset == "ECGFiveDays":
        return 2 #136
    if dataset == "ElectricDevices":
        return 7 #96
    if dataset == "FaceAll":
        return 14 # 131
    if dataset == "FaceFour":
        return 4 # 350
    if dataset == "FacesUCR":
        return 14 # 131
    if dataset == "FISH":
        return 7 # 463
    if dataset == "FordA":
        return 2 #500
    if dataset == "FordB":
        return 2 # 500
    if dataset == "Gun_Point":
        return 2 # 150
    if dataset == "Ham":
        return 2 # 431
    if dataset == "HandOutlines":
        return 2 # 2709
    if dataset == "Haptics":
        return 5 # 1092
    if dataset == "Herring":
        return 2 # 512
    if dataset == "InlineSkate":
        return 7 # 1882
    if dataset == "InsectWingbeatSound":
        return 11 # 256
    if dataset == "ItalyPowerDemand":
        return 2 # 24
    if dataset == "LargeKitchenAppliances":
        return 3 # 720
    if dataset == "Lightning2":
        return 2 # 637
    if dataset == "Lightning7":
        return 7 # 319
    if dataset == "MALLAT":
        return 8 # 1024
    if dataset == "Meat":
        return 3 # 448
    if dataset == "MedicalImages":
        return 10 # 99
    if dataset == "MiddlePhalanxOutlineAgeGroup":
        return 3 #80
    if dataset == "MiddlePhalanxOutlineCorrect":
        return 2 #80
    if dataset == "MiddlePhalanxTW":
        return 6 #80
    if dataset == "MoteStrain":
        return 2 #84
    if dataset == "NonInvasiveFatalECG_Thorax1":
        return 42 #750
    if dataset == "NonInvasiveFatalECG_Thorax2":
        return 42 #750
    if dataset == "OliveOil":
        return 4 #570
    if dataset == "OSULeaf":
        return 6 #427
    if dataset == "PhalangesOutlinesCorrect":
        return 2 #80
    if dataset == "Phoneme":
        return 39 #1024
    if dataset == "Plane":
        return 7 #144
    if dataset == "ProximalPhalanxOutlineAgeGroup":
        return 3 #80
    if dataset == "ProximalPhalanxOutlineCorrect":
        return 2 #80
    if dataset == "ProximalPhalanxTW":
        return 6 #80
    if dataset == "RefrigerationDevices":
        return 3 #720
    if dataset == "ScreenType":
        return 3 #720
    if dataset == "ShapeletSim":
        return 2 #500
    if dataset == "ShapesAll":
        return 60 # 512
    if dataset == "SmallKitchenAppliances":
        return 3 #720
    if dataset == "SonyAIBORobotSurfaceII":
        return 2 #65
    if dataset == "SonyAIBORobotSurface":
        return 2 #70
    if dataset == "StarLightCurves":
        return 3 #1024
    if dataset == "Strawberry":
        return 2 #235
    if dataset == "SwedishLeaf":
        return 15 # 128
    if dataset == "Symbols":
        return 6 #398
    if dataset == "synthetic_control":
        return 6 #60
    if dataset == "ToeSegmentation1":
        return 2 #277
    if dataset == "ToeSegmentation2":
        return 2 #343
    if dataset == "Trace":
        return 4 #275
    if dataset == "TwoLeadECG":
        return 2 #82
    if dataset == "Two_Patterns":
        return 4 #128
    if dataset == "uWaveGestureLibrary_X":
        return 8 # 315
    if dataset == "uWaveGestureLibrary_Y":
        return 8 # 315
    if dataset == "uWaveGestureLibrary_Z":
        return 8 # 315
    if dataset == "UWaveGestureLibraryAll":
        return 8 # 945
    if dataset == "wafer":
        return 2 #152
    if dataset == "Wine":
        return 2 #234
    if dataset == "WordSynonyms":
        return 25 #270
    if dataset == "Worms":
        return 5 #900
    if dataset == "WormsTwoClass":
        return 2 #900
    if dataset == "yoga":
        return 2 #426
    exit('missing dataset')
