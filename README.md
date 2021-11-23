# Offline Meta Reinforcment Learning - Identifiability Challenges and Effective Data Collection Strategies

### Requirements ### 
All requirements are specified in ``requirements.txt``. \
We also provide a yaml file: ``omrl.yml``. Run: ``conda env create -f omrl.yml`` and activate the env with ``conda activate omrl``.  
For the MuJoCo-based experiments (``Half-Cheetah-Vel`` and ``Ant-Semi-circle`` you will need a MuJoCo license).  

## Offline Setting ##
### Data collection ###
The main script for data collecion is ``train_single_agent.py``.  
Configuration files are in ``data_collection_config``. All training parameters can be set from within the files, or by passing command line arguments.  

Run:  
``python train_single_agent.py --env-type X --seed Y``  

where X is a domain (e.g., ``gridworld``, ``point_robot_sparse``, ``cheetah_vel``, ``ant_semicircle_sparse``, ``point_robot_wind``, ``escape_room``) and Y is some integer (e.g. ``73``).  
This will train a standard RL agent (implemented in ``learner.py``) to solve a single task. Different seeds correspond to different tasks (from the same task distribution).  




### VAE training ###
The main script for the VAE training is ``train_vae_offline.py``.  
Configuration files are in ``vae_config``. All training parameters can be set from within the files, or by passing command line arguments.  

Run (for example):  
``python train_vae_offline.py --env-type ant_semicircle_sparse``  

This will train the VAE (implemented in ``models\vae.py``).  

Reward Relabelling (RR) is used when the argument ``--hindsight-relabelling`` is set to ``True``. 



### Offline Meta-RL Training ###
The main script for the offline meta-RL training is ``train_agent_offline.py``.  
Configuration files are in ``offline_config``. All training parameters can be set from within the files, or by passing command line arguments.  

Run (for example):  
``python train_offline_agent.py --env-type ant_semicircle_sparse``  

Note the ``--transform-data-bamdp`` argument. When training the meta-RL agent for the first time, this argument should be set to ``True`` in order to perform State Relabelling. 
That is, after loading the datasets and the trained vae, the datasets will be passed through the encoder to produce the approximate belief. This belief is then concatenated 
to the states in our data to form the hyper-states on which our meta-RL agent is trained. This new dataset (with hyper-states) is also saved locally. If this dataset is available
(e.g., after already running the script), you can change the argument to ``False`` in order to save time.  



## Online Setting ##
For the online training, run: ``python online_training.py --env-type X`` where X is a domain (see Data Collection part above).
Configuration files are in ``online_config``. All training parameters can be set from within the files, or by passing command line arguments.  


## Citation ##
```
@inproceedings{dorfman2021offline,
  title={Offline Meta Reinforcement Learning--Identifiability Challenges and Effective Data Collection Strategies},
  author={Dorfman, Ron and Shenfeld, Idan and Tamar, Aviv},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

## Communication ##
For any questions, please contact Ron Dorfman: ``rondorfman2@gmail.com``





