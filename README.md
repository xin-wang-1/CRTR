# CRTR（Cross-Resolution Target Recognition）
### ENVIRONMENT（04/20/2024）
conda create -n CRTR python=3.8.12  
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch  
pip install tqdm==4.50.2  
pip install tensorboard==2.8.0  
conda install -c conda-forge nvidia-apex  
pip install scipy==1.5.2  
pip install ml-collections==0.1.0  
pip install scikit-learn==0.23.2  

### DATASETS: 
data/dataset_name/domain/class/fig
**.txt corresponds to the domain

## TRAINING
python main.py --dataset *** --name *** --source_list data/source.txt --target_list data/target.txt --test_list data/test.txt --num_steps 5000 --img_size 256 --use_CFAM --use_RSCM 

