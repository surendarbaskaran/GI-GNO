pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.6.1 -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
pip install pyvista
#for windows, use below script : 
#pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
