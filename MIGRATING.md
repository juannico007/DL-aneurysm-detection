## Migration (from v0.0.3 → v0.0.6)

This release introduces major internal optimizations and changes to the data pipeline, model definition, and training setup.  
Follow these steps to migrate from previous versions.


1.  Install the new dependencies:
```bash
pip install accelerate[sagemaker] boto3 python-dotenv h5py
```

2. Run the `patch_accelerate_config.py` script to fix the Accelerate bug:

```bash
python3 scripts/patch_accelerate_config.py
```

3. Create a `.env` file in the `src` folder and add the secret values

4. If you run locally, make sure to have h5-aneurysm.h5 file in the root folder and the train.csv file. 

5. You can launch training either locally or on the cloud (SageMaker):
```bash
python src/run_training.py --mode local
```

or

```bash
python src/run_training.py --mode cloud
```

