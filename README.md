# iNaturalist 2021 - UFAM Team

6th Place Solution to [iNat Challenge 2021](https://www.kaggle.com/c/inaturalist-2021/): 10,000 Species Recognition Challenge with iNaturalist Data

### Requirements

Prepare an environment with python=3.8, tensorflow=2.3.1

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

### Data

Please refer to the [iNaturalist 2021 Competition Github page](https://github.com/visipedia/inat_comp/tree/master/2021) for additional dataset details and download links.

Use the script `dataset_tools/create_inat2021_tf_records.py` to generate the TFRecords files.

### Training

To train a classifier use the script `main.py`. As long as our final submission has two training stages, you can use the script `multi_stage_train.py`:
```bash
python multi_stage_train.py --training_files=PATH_TO_BE_CONFIGURED/inat_train.record-?????-of-02240 \
    --num_training_instances=2686843 \
    --validation_files=PATH_TO_BE_CONFIGURED/inat_val.record-?????-of-00084 \
    --num_validation_instances=100000 \
    --num_classes=10000 \
    --model_name=efficientnet-b3 \
    --input_size=300 \
    --input_size_stage3=432 \
    --input_scale_mode=uint8 \
    --batch_size=32 \
    --lr_stage1=0.1 \
    --lr_stage2=0.1 \
    --lr_stage3=0.008 \
    --momentum=0.9 \
    --epochs_stage1=0 \
    --epochs_stage2=20 \
    --epochs_stage3=2 \
    --unfreeze_layers=18 \
    --label_smoothing=0.1 \
    --randaug_num_layers=6 \
    --randaug_magnitude=4 \
    --model_dir=PATH_TO_BE_CONFIGURED \
    --random_seed=42
```

The parameters can also be passed using a config file:
```bash
python multi_stage_train.py --flagfile=configs/efficientnet_b3_final_submission_training.config \
    --model_dir=PATH_TO_BE_CONFIGURED
```

For more parameter information, please refer to `multi_stage_train.py` or `main.py`. See `configs` folder for some training configs examples.

#### Training Geo Prior Model

To train geo prior model used on our final submission please see our [TF implementation](https://github.com/alcunha/geo_prior_tf/).

### Prediction

To create a submission for the competition use script `predict_main.py`:
```bash
python predict_main.py --test_files=PATH_TO_BE_CONFIGURED/inat_public_test.record-?????-of-00417 \
    --num_classes=10000 \
    --model_name=efficientnet-b3 \
    --input_size=432 \
    --input_scale_mode=uint8 \
    --batch_size=8 \
    --ckpt_dir=PATH_TO_BE_CONFIGURED/ \
    --geo_prior_ckpt_dir=PATH_TO_BE_CONFIGURED/ \
    --submission_file_path=PATH_TO_BE_CONFIGURED/final_submission.csv \
    --use_tta
```

### Results

[Efficientnet-B3](https://drive.google.com/file/d/1SDx5P-ovb1NQPyPu4ubgsOhttUEdzs4A/view?usp=sharing) was trained on iNat2021 train set, inference using input of 432x432.
[Geo Prior](https://drive.google.com/file/d/1xzYaouGOZQrbibHbTMUs4d8PZZYDnZXT/view?usp=sharing) model was trained using coordinates and date info from iNat2021 train set.

| Model name                        | Private Score |
|-----------------------------------|---------------|
| Efficientnet-B3                   | 0.16756       |
| Efficientnet-B3 + Geo Prior       | 0.10752       |
| Efficientnet-B3 + Geo Prior + TTA | 0.09894       |

### Contact

If you have any questions, feel free to contact Fagner Cunha (e-mail: fagner.cunha@icomp.ufam.edu.br) or Github issues. 

### License

[Apache License 2.0](LICENSE)