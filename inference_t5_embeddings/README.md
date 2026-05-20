## How to load and inject inference-time embedding

Each embedding is a single numpy.float32 array shape (1, 512, 1024) (~2 MB), ready to slot into obs/language_embedding at inference:
```
  import pickle, torch
  with open("the_robot_arm_gently_nudges_the_white_object_towards_the_target_circle.pickle", "rb") as f:
      emb = pickle.load(f)
  batch["obs/language_embedding"] = torch.from_numpy(emb).to(device)
```

More can be generated in the mimic-video repository using

```
model/.venv/bin/python scripts/embed_t5_prompt_inference \
    --config configs/so101/idm_large_all_tasks.yaml \
    --prompt "the robot arm gently nudges the colored cube towards the target circle"
```

which will save the respective pickle file in `cwd`. 
