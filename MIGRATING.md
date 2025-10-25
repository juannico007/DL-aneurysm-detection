## Migration (from v0.0.2 → v0.0.3)

This release introduces major internal optimizations and changes to the data pipeline, model definition, and training setup.  
Follow these steps to migrate from previous versions.


- Preprocess now with the new outputs `.npz`. Last version returned `.nii.qz`.
Run:
```bash
python3 -m preprocessing.cli
```

- Run new depndencies and make sure to be in the right environment: 
```bash
pip install -r requirements.txt
```

- Setup the accelerate
```bash
accelerate config
```

Make sure to be similar as those variables: 

```text
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: model/ds_config.json
  zero3_init_flag: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: INDUCTOR
  dynamo_mode: reduce-overhead
  dynamo_use_dynamic: true
  dynamo_use_fullgraph: true
  dynamo_use_regional_compilation: true
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

- Run the model:
```bash
accelerate launch -m model.cli
```

