#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`


# config_path='adv.adaptseg.2urban'
# python AdaptSegNet_train.py ${config_path}

# config_path='adv.clan.2urban'
# python CLAN_train.py ${config_path}

# config_path='adv.fada.2urban'
# python FADA_train.py ${config_path}

# config_path='adv.tn.2urban'
# python TN_train.py ${config_path}

# config_path='st.cbst.2urban'
# python CBST_train.py ${config_path}

config_path='st.iast.2urban'
python IAST_train.py ${config_path}

# config_path='st.pycda.2urban'
# python PyCDA_train.py ${config_path}

