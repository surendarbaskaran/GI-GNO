# GI-GNO  Model


The current version in main branch is updated working file set, next update will be setup a continuous pipeline where once the model is predicted,  
predicted vtk file generated along with cl and cd value, based on inference result 


I have added setup.sh to install required libraries, 
make changes in model.py to add FNO layers, 
config.py contains the parameters input nodes, hidden nodes, output nodes, and 

in training.py , need to change the path for preprocessed files, 
inference.py requires ->  test case file in pt file format, and model.pt file
pt_to_vtk.py file converts the predicted value from inference.py to  vtk file for visualization.
