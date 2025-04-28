# README  
The CNN architecture lives in llr_cnn_model.py. The eval_llr_model.py file can be ran to evaluate 15 specified test samples from the dataset using the optimized model parameters. The LLRDataset class is defined in llr_dataset.py , which reads the Excel file that was compiled from MATLAB scripts and contains the ground truth labels for every sample. The training script is stored in train_llr_model.py. 

The training script can be ran locally to obtain a .pth file with all the optimized model parameters. Once obtained, the eval_llr_model.py file can be ran to evaluate the model. Due to the size of the .pth file being approximately 120 MB, we have not uploaded a version onto Github.

Raw sample data, annotated sample data, and the custom MATLAB scripts used to annotate the samples and capture ground truth coordinate points for each landmark are captured in the data_acquisition folder.
