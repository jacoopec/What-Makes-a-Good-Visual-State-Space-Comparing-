# PushCube environment for DINO-WM

This extension adds support for ManiSkill's `PushCube-v1` task to DINO-WM. It provides an adapter for image-goal planning and a loader for recorded demonstrations. The adapter converts simulator observations into DINO-WM's visual/proprioceptive format, restores recorded simulator states, executes action sequences, and evaluates whether the cube reaches a goal position.

The robot is a Panda arm. The default controller is `pd_ee_delta_pos`, observations use RGB images from `base_camera`, and the planning wrapper runs one simulator environment with the `physx_cpu` backend.

## Files and integration

| File | Purpose |
| --- | --- |
| [pushcube_env.py](pushcube_env.py) | Minimal `PushCubeDinoEnv` adapter exposing native ManiSkill `reset`, `step`, action/observation spaces, and `close`. |
| [pushcube_wrapper.py](pushcube_wrapper.py) | `PushCubeWrapper`, implementing the DINO-WM planning methods and observation/state conversion. |
| [__init__.py](__init__.py) | Package marker; currently empty. |
| [../../conf/env/pushcube.yaml](../../conf/env/pushcube.yaml) | Hydra environment configuration selected with `env=pushcube`. |
| [../../datasets/pushcube_dset.py](../../datasets/pushcube_dset.py) | HDF5 trajectory loading, normalization statistics, and train/validation splitting. |

`PushCubeDinoEnv` returns native ManiSkill observations and Gymnasium step results. Use `PushCubeWrapper` when working with DINO-WM's `prepare`, `rollout`, and `eval_state` interfaces.

## Prerequisites

Use the project's DINO-WM dependencies, plus a ManiSkill installation providing `PushCube-v1` and Gymnasium. The environment code also imports NumPy and PyTorch; dataset loading additionally requires `h5py` and `einops`.

The repository's [environment.yaml](../../environment.yaml) lists the original DINO-WM dependencies, including legacy `gym`, but does not list ManiSkill or Gymnasium. for generating the dataset another environment  has been  created because of conflicts between  dependencies and/or python versionand Gymnasium. This extension does not pin a tested ManiSkill version.

Run the examples below from the `dino_wm` project root, which contains `train.py`, `plan.py`, `env/`, and `conf/`.

## Configuration and training

The defaults in [pushcube.yaml](../../conf/env/pushcube.yaml) are:

```yaml
name: pushcube
args: []
kwargs: {}

dataset:
  _target_: datasets.pushcube_dset.load_pushcube_slice_train_val
  data_path: /work/cvcs2026/DINO_WM/data/pushcube/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
  n_rollout: null
  normalize_action: true
  split_ratio: 0.9
  filter_success: true
  transform:
    _target_: datasets.img_transforms.default_transform
    img_size: ${img_size}

decoder_path: null
num_workers: 4
```

Override `env.dataset.data_path` with the absolute path to your own demonstration file. The supplied path is specific to the original machine; this loader does not resolve it through `DATASET_DIR`.

```bash
python train.py --config-name train.yaml env=pushcube env.dataset.data_path=/absolute/path/to/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 frameskip=1 num_hist=3
```

Quote the entire `env.dataset.data_path=...` argument if the path contains spaces. This example explicitly uses `frameskip=1`; the base training configuration defaults to `5`.

| Setting | Behavior |
| --- | --- |
| `n_rollout: null` | Load all eligible trajectories; an integer limits the number retained. |
| `filter_success: true` | Keep a trajectory if any entry in its `success` array is true. |
| `normalize_action: true` | Compute action and proprioception normalization statistics. State statistics are also computed, but returned simulator states remain unnormalized. |
| `split_ratio: 0.9` | Split trajectories into training and validation sets with a fixed random seed of `42`. |
| `transform.img_size: ${img_size}` | Resize/center-crop images and normalize RGB values to `[-1, 1]`. |
| `num_workers: 4` | Number of training DataLoader workers; HDF5 handles are opened lazily per worker. |
| `decoder_path: null` | No pretrained decoder path is supplied by this environment configuration. |

The loader requires at least two retained trajectories. Trajectories must also be long enough for the requested `num_hist`, `num_pred`, and `frameskip`. Normalization statistics are currently calculated on the retained dataset before the train/validation split.

## Demonstration format

The loader expects an HDF5 file containing numerically named trajectory groups (`traj_0`, `traj_1`, etc.) with these entries:

```text
traj_<id>/
  actions
  success
  obs/
    sensor_data/base_camera/rgb
    agent/qpos
    agent/qvel
  env_states/
    articulations/panda
    actors/cube
    actors/goal_region
    actors/table-workspace
```

`success` is required when success filtering is enabled. For a trajectory with `T` actions, the loader uses `T` aligned observation/state frames; any extra terminal frame is not included in the dataset sequence length. Recorded actions must match the controller used for replay.

`PushCubeDataset.get_frames(...)` returns `(obs, actions, state, info)`. Images are converted from HWC uint8 to CHW floating-point values in `[0, 1]` before the configured transform. Actions and proprioception are standardized when normalization is enabled. `info` contains `traj_name`.

## Observation, action, and state formats

The planning wrapper removes ManiSkill's leading singleton environment dimension and returns NumPy arrays:

| Value | Shape | Meaning |
| --- | --- | --- |
| `obs["visual"]` | `[H, W, 3]` | Base-camera RGB image, `uint8`. |
| `obs["proprio"]` | `[18]` | Nine joint positions followed by nine joint velocities, `float32`. |
| `state` | `[70]` | Raw simulator state, `float32`, using the layout below. |
| Action | `[action_dim]` | Controller action; dimension and bounds are read from `single_action_space`. |

The wrapper clips actions to the controller's bounds and adds a batch dimension before calling ManiSkill. It expects controller-scale actions, not dataset-standardized actions. To replay actions returned by the dataset with normalization enabled, first apply `actions * dataset.action_std + dataset.action_mean`.

The 70-dimensional simulator state is shared by the dataset and wrapper:

| Slice | Size | Simulator entry |
| --- | --- | --- |
| `0:31` | 31 | Panda articulation |
| `31:44` | 13 | Cube actor |
| `44:57` | 13 | Goal-region actor |
| `57:70` | 13 | Table actor |

`prepare` reconstructs ManiSkill's state dictionary and resets through `reset_to_env_states`. It restores the robot, cube, goal region, and table, preserving the tensor type/device expected by the simulator.

## Wrapper API

```python
PushCubeWrapper(
    control_mode="pd_ee_delta_pos",
    obs_mode="rgb",
    sim_backend="physx_cpu",
    goal_tolerance=0.03,
)
```

| Method | Behavior |
| --- | --- |
| `prepare(seed, init_state)` | Reset to a raw 70D recorded state; return `(obs, state)`. |
| `rollout(seed, init_state, actions)` | Reset and execute `[T, action_dim]` actions; return observations and states including the initial frame. Shapes are `[T+1, H, W, 3]`, `[T+1, 18]`, and `[T+1, 70]`. |
| `step_multiple(actions)` | Execute from the current state; return `(obses, rewards, dones, infos)`, with `T` post-step frames and `infos["state"]` shaped `[T, 70]`. Requires a nonempty action sequence. |
| `eval_state(goal_state, cur_state)` | Return image-goal and physical-task metrics for two individual raw 70D states. |
| `update_env(env_info)` | No-op: scene geometry is fixed. |
| `sample_random_init_goal_states(seed)` | Raises `NotImplementedError`; use dataset goals. |
| `close()` | Release the simulator. |

`step_multiple` combines termination and truncation into `dones`. Neither execution method stops early on a done flag; both execute the supplied action sequence. Observation conversion requires the RGB camera and Panda joint fields, even though `obs_mode` is configurable.

Example: replay the first ten recorded actions from a demonstration:

```python
from datasets.pushcube_dset import PushCubeDataset
from env.pushcube.pushcube_wrapper import PushCubeWrapper

dataset = PushCubeDataset(
    data_path="/absolute/path/to/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5",
    normalize_action=False,
)
_, actions, states, _ = dataset[0]

env = PushCubeWrapper()
try:
    observations, replay_states = env.rollout(
        seed=0,
        init_state=states[0].numpy(),
        actions=actions[:10].numpy(),
    )
    print(observations["visual"].shape)
    print(replay_states.shape)
finally:
    env.close()
```

## Evaluation and planning integration

`eval_state` reports two different goals:

| Metric | Definition in this wrapper |
| --- | --- |
| `cube_goal_dist` | XY distance between the current cube and the cube in the desired dataset goal state. |
| `success` | `cube_goal_dist < goal_tolerance` (default `0.03`). |
| `task_goal_dist` | XY distance between the current cube and the current physical goal marker. |
| `task_success` | `task_goal_dist < 0.1` and current cube Z position `< 0.025`. |

Image-goal success can differ from physical-task success because the dataset goal image may show the cube away from the physical goal marker.

**Command-line planning is not yet connected in this checkout.** [plan.py](../../plan.py) constructs environments with legacy `gym.make(model_cfg.env.name, ...)`, while [env/__init__.py](../__init__.py) does not register a `pushcube` environment. Importing `mani_skill.envs` registers `PushCube-v1` with Gymnasium; it does not register DINO-WM's lowercase `pushcube` adapter with legacy Gym.

Before using `plan.py`, its environment-construction path needs to instantiate `PushCubeWrapper` and supply it through a compatible DINO-WM vector environment. The wrapper API can already be called directly as in the example above. The empty `args`/`kwargs` fields in `pushcube.yaml` alone do not establish this connection.

Once that integration is in place, use `goal_source=dset`: random initial/goal state sampling is not implemented. Planning reloads the dataset configuration from the model's saved `hydra.yaml`, so its dataset path must remain accessible. If enabling planning evaluations during training, also restrict `plan_settings.goal_source` to `[dset]`; the base training configuration includes `random_state`, although training-time planning is disabled by default.
