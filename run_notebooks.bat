@echo off
echo Running part3_rnn_medical.ipynb...
py -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 part3_rnn_medical.ipynb

echo Running part4_agents_xai.ipynb...
py -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 part4_agents_xai.ipynb

echo Running part5_hybrid_models.ipynb...
py -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 part5_hybrid_models.ipynb

echo Running part6_ablation.ipynb...
py -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 part6_ablation.ipynb

echo All notebooks executed successfully.
del "%~f0"
