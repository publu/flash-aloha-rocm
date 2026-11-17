# INVENTORY — JohnYao93/flash-aloha

Step label: `download`. Written 2026-11-13 (PST) from a read-only inspection of the clone at
`~/research/flash-aloha/repo`. Nothing from the repository was executed, imported,
built or installed. Paths below are relative to the repository root unless they start with `/`.

Conventions used in this file:

- **[VERIFIED]** — read directly from the checkout, from `git`, or from a content hash.
- **[README CLAIM]** / **[DOC CLAIM]** — what the repository's own text says; not checked by running anything.
- **[PROJECT PAGE CLAIM]** — text of the linked project web page (outside the repository).
- **[NOT ESTABLISHED]** — could not be determined from what was read.

---

## 0. Summary

- **What it is.** A Python code release (first commit 2026-11-11, 5 commits) for teaching ARX L5 robot
  arms by hand, recording camera + joint data, training two imitation policies (a decoder-only ACT and a
  flow-matching policy built on LeRobot's diffusion U-Net), and running them on the arms either locally or
  through a WebSocket policy server. It vendors a modified copy of the ARX5 arm SDK (C++/pybind, with
  prebuilt closed vendor libraries) and a copy of ROS `kdl_parser`.
- **Size.** 399 files, 64.2 MiB work tree + 11.2 MiB `.git` = 75.4 MiB on disk. 95 % of the bytes are the
  two vendored directories (meshes, URDFs, shared libraries); the project's own material is 250 files / 3.3 MiB.
- **Licence.** MIT (root), plus MIT (vendored arx5-sdk) and BSD-3-Clause (vendored kdl_parser).
- **No datasets, no model weights, no notebooks, no LFS, no submodules, no releases, no tags.**
- **Main things to know before touching it** (details in §12):
  1. Ten prebuilt ELF libraries with no source in the repo (byte-identical to upstream arx5-sdk).
  2. `robologger`, imported by the default camera-server path, is neither declared nor locked nor vendored.
  3. The policy server and the camera recording endpoints bind `0.0.0.0`; the repository configures no
     authentication for either.
  4. All four shipped training recipes set `wandb: true` (metrics and config go to Weights & Biases).
  5. Checkpoint loading unpickles `dataset_stats.pkl`; three calibration `.npy` files are pickles.
  6. Almost every entry point moves real arms on `can0`/`can1`; helper scripts call `sudo`.

---

## 1. Clone verification

The clone was made by an earlier run of this step (stopped by a usage limit) with:

```
cd ~/research/flash-aloha && GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 nice -n 19 \
  git clone --no-recurse-submodules https://github.com/JohnYao93/flash-aloha repo
```

It was not cloned or fetched again. Checks made in this run, all read-only (`GIT_OPTIONAL_LOCKS=0`):

| Check | Result |
| --- | --- |
| `git status --short` | empty (clean); `status --ignored` also empty |
| `git fsck --no-dangling` | no output, exit 0 |
| `git rev-parse HEAD` | `aec35bb7d29bf16fe360fd5abe71f976356b5a24` (= `origin/main`) |
| `git rev-parse --is-shallow-repository` | `false` (full history) |
| pack | 1 pack, 514 objects, 11,631,960 bytes (`size-pack: 11.11 MiB`) |
| `.git/config` | only `core`, `remote "origin"` (the GitHub URL) and `branch "main"`; no filters, hooks path or aliases |
| `.git/hooks` | only the stock `*.sample` files |
| tracked vs on disk | 399 tracked files, 399 files on disk (excluding `.git`), 74 directories, no symlinks, no special files |
| `git diff --stat HEAD` | empty |
| HEAD tree | `1a3f2ada41acc84f8ba537265921adffeb83ee6f` |

The public GitHub API (unauthenticated, fetched 2026-11-14 03:51 UTC) agrees: default branch `main` at
`aec35bb…`, 1 branch, 0 tags, 0 releases, `size: 11382` (KB), `license: MIT`, `language: Python`,
`created_at: 2026-11-12T01:16:30Z`, `pushed_at: 2026-11-12T22:31:39Z`, 70 stars, 1 fork, not a fork,
no description, no homepage field, `has_pages: false`, `has_wiki: false`.

---

## 2. Git facts

| Item | Value |
| --- | --- |
| HEAD | `aec35bb7d29bf16fe360fd5abe71f976356b5a24` |
| HEAD subject | "Add --graph to replay practice: commanded vs achieved position after each replay" |
| HEAD date | author 2026-11-12T14:31:27-08:00, commit 2026-11-12T14:31:28-08:00 |
| Commits | 5 on `main` (5 reachable from all refs); 0 merges; linear history |
| Date range | 2026-11-11T17:11:40-08:00 → 2026-11-12T14:31:27-08:00 (about 21 hours) |
| Authors | 2 names: **John Yao** (4 commits), **FarSky** (1 commit). 2 distinct identities. Emails not recorded here. |
| Committers | John Yao (4), GitHub (1 — the `FarSky` commit was made through the GitHub web interface and carries a GitHub signature that could not be checked locally) |
| Co-author trailers | 4 of 5 commits end with `Co-Authored-By: Claude Fable 5.2` (all except the web edit) |
| Branches | local `main`; remote `origin/main`, `origin/HEAD -> origin/main`. No other branch. |
| Tags | none |
| Submodules / gitlinks | none (no `.gitmodules`, no mode-160000 entries) |
| `.gitattributes` | none |
| File modes | 367 × `100644`, 32 × `100755` |
| Deletions / renames in history | none (`git log --diff-filter=DR` is empty) |

### How the repository evolved (`git log --stat`)

| # | Commit | Date (-08:00) | Author | Subject | Change |
| --- | --- | --- | --- | --- | --- |
| 1 | `cb50318` | 2026-11-11 17:11 | John Yao | Initial public release | 398 files, 723,473 insertions |
| 2 | `a88394d` | 2026-11-11 17:22 | John Yao | Scale the channels-last gradient test tolerance to the gradient magnitude | 1 file, +4 −1 (`scripts/tests/test_flow_precision.py`) |
| 3 | `7bb658d` | 2026-11-11 18:07 | John Yao | Remove tests for things outside this repository | 15 test files, +23 −125 |
| 4 | `4172a8e` | 2026-11-11 18:29 | FarSky | Update README.md | 1 file, +3 −1 (title changed from `# flash_aloha`, project-website line added) |
| 5 | `aec35bb` | 2026-11-12 14:31 | John Yao | Add --graph to replay practice… | 7 files, +250 −8; adds `src/visualization/replay_tracking_graph.py` (the only file added after commit 1) |

Composition of the first commit (from `git show --numstat cb50318`):

| Group | Files | Added lines |
| --- | --- | --- |
| vendored `src/hardware/arx5-sdk` (text) | 104 | 370,657 (mostly 10 `.obj` meshes, 347,654 lines) |
| vendored `src/kdl_parser` (text) | 26 | 302,515 (one file is 300,391 lines, see §12.1) |
| `uv.lock` | 1 | 5,922 |
| own Python, non-test (`src/`, `scripts/`) | 121 | 29,821 |
| own tests (`scripts/tests/`) | 62 | 10,446 |
| own Markdown | 31 | 3,109 |
| own other (YAML, shell, TOML, …) | 25 | 1,003 |
| binary files | 28 | — |

So about 93 % of the inserted lines of "Initial public release" are vendored models/meshes/code; the
project's own text is roughly 44,000 lines. Commit 3's message and diff show the release is an **export
of a larger private codebase**: it drops "guards against subsystems that were never part of the release,
the test that needed an OpenPI checkout, and tests of pre-release data and checkpoint formats", and the
removed test code named excluded policies `Diffusion`, `CNNMLP`, `TransformerPolicy` and import prefixes
`inpainting`, `policy.detr` (`git show 7bb658d -- scripts/tests/test_export_scope.py`).

---

## 3. Size on disk and file counts

| Part | Files | Bytes | MiB |
| --- | --- | --- | --- |
| Whole folder `repo/` (apparent) | — | 79,026,556 | 75.37 |
| `.git/` | — | 11,721,635 | 11.18 |
| Work tree | 399 | 67,304,921 | 64.19 |
| — vendored `src/hardware/arx5-sdk/` | 123 | 40,186,646 | 38.33 |
| — vendored `src/kdl_parser/` | 26 | 23,782,300 | 22.68 |
| — everything else (own) | 250 | 3,335,975 | 3.18 (of which `uv.lock` 1.24) |

Top-level directories: `src/` 274 files / 65.2 MB, `scripts/` 91 / 0.65 MB, `docs/` 13 / 71 KB,
`calibration/` 12 / 26 KB, `.github/` 1 / 0.7 KB, 8 root files / 1.34 MB.

### Files by extension (`tools/inventory.py ext`)

| Extension | Files | Bytes | Share of bytes |
| --- | --- | --- | --- |
| `.py` | 213 | 2,234,031 | 3.3 % |
| `.md` | 34 | 205,883 | 0.3 % |
| `.yaml` | 25 | 20,084 | 0.0 % |
| `.cpp` | 16 | 230,794 | 0.3 % |
| `.urdf` | 14 | 23,832,988 | 35.4 % |
| `.h` | 13 | 76,647 | 0.1 % |
| `.obj` | 10 | 11,342,120 | 16.9 % |
| `.sh` | 10 | 16,168 | 0.0 % |
| `.so` | 10 | 2,650,320 | 3.9 % |
| `.stl` / `.STL` | 9 | 25,303,556 | 37.6 % |
| `.npy` | 5 | 2,069 | 0.0 % |
| `.txt` (all `CMakeLists.txt`) | 5 | 12,289 | 0.0 % |
| `.npz` | 4 | 13,830 | 0.0 % |
| `LICENSE` (no extension) | 3 | 3,731 | 0.0 % |
| `.gitignore` | 3 | 314 | 0.0 % |
| `.toml` | 3 | 2,136 | 0.0 % |
| `.xml` | 3 | 11,791 | 0.0 % |
| `.hpp` | 2 | 5,896 | 0.0 % |
| `.rst` | 2 | 10,089 | 0.0 % |
| `.yml` | 2 | 2,055 | 0.0 % |
| `.lock` (`uv.lock`) | 1 | 1,301,007 | 1.9 % |
| `.pyi` | 1 | 15,365 | 0.0 % |
| one each: `AMENT_IGNORE`, `activate_arm`, `.cfg`, `.cmake`, `.json`, `.launch`, `.python-version`, `Dockerfile.{x86_64,aarch64,single_ver_x86_64,single_ver_aarch64}` | 11 | 11,758 | 0.0 % |
| **Total** | **399** | **67,304,921** | |

No `.ipynb`, no `Makefile`, no `requirements*.txt`, no root `setup.py`, no `docker-compose`.

### Code size (lines, counted from tracked files)

| Area | Python files | Lines |
| --- | --- | --- |
| `src/inference` | 11 | 5,625 (one file, `inference_direct.py`, is 3,505 lines / 177 KB) |
| `src/camera_utils` | 9 | 3,944 |
| `src/recording` | 11 | 3,885 |
| `src/policy` | 11 | 3,626 |
| `src/visualization` | 11 | 2,590 |
| `src/config` | 15 | 2,005 |
| `src/utils` | 9 | 1,316 |
| `src/data_processing` | 7 | 1,166 |
| `src/contracts` | 9 | 741 |
| `src/hardware` (own: top level + `backends/`) | 8 | 795 |
| `scripts/` non-test | 21 | 4,290 |
| `scripts/tests` | 55 (52 `test_*.py`, 468 `def test_…`) | 10,218 |
| **own Python total** | **177** | **≈ 40,200** |
| vendored `src/hardware/arx5-sdk` Python | 32 | 10,961 |
| vendored `src/kdl_parser` Python | 4 | 175 |
| vendored C/C++ (`.cpp/.h/.hpp`) | 31 | 7,227 (arx5-sdk 6,217; kdl_parser 1,010) |

### The 40 largest files (`tools/inventory.py largest 40`)

| Bytes | Size | Path |
| --- | --- | --- |
| 23,714,538 | 22.6 MiB | `src/kdl_parser/kdl_parser_py/test/kdl_parser_py.urdf` |
| 6,634,584 | 6.3 MiB | `src/hardware/arx5-sdk/models/meshes/arx_finray_gripper.stl` |
| 5,208,084 | 5.0 MiB | `src/hardware/arx5-sdk/models/meshes/arx_gopro_finray_gripper.stl` |
| 4,509,584 | 4.3 MiB | `src/hardware/arx5-sdk/models/meshes/X5/link2.STL` |
| 2,533,384 | 2.4 MiB | `src/hardware/arx5-sdk/models/meshes/X5/link6.STL` |
| 2,301,371 | 2.2 MiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/link5.obj` |
| 2,270,184 | 2.2 MiB | `src/hardware/arx5-sdk/models/meshes/X5/link5.STL` |
| 2,144,530 | 2.0 MiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/link3.obj` |
| 1,956,784 | 1.9 MiB | `src/hardware/arx5-sdk/models/meshes/X5/link3.STL` |
| 1,944,757 | 1.9 MiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/link6.obj` |
| 1,933,493 | 1.8 MiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/link2.obj` |
| 1,804,484 | 1.7 MiB | `src/hardware/arx5-sdk/models/meshes/X5/base_link.STL` |
| 1,301,007 | 1.2 MiB | `uv.lock` |
| 835,733 | 816.1 KiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/camera.obj` |
| 831,334 | 811.8 KiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/base_link.obj` |
| 722,584 | 705.6 KiB | `src/hardware/arx5-sdk/wheels/lib/aarch64/libsolver.so` |
| 660,208 | 644.7 KiB | `src/hardware/arx5-sdk/wheels/lib/x86_64/libsolver.so` |
| 406,200 | 396.7 KiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/link4.obj` |
| 379,622 | 370.7 KiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/link1.obj` |
| 322,888 | 315.3 KiB | `src/hardware/arx5-sdk/wheels/lib/aarch64/libhardware.so` |
| 285,359 | 278.7 KiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/link8.obj` |
| 279,721 | 273.2 KiB | `src/hardware/arx5-sdk/models/mujoco/ARX_L5/meshes/assets/link7.obj` |
| 270,376 | 264.0 KiB | `src/hardware/arx5-sdk/wheels/lib/x86_64/libhardware.so` |
| 240,084 | 234.5 KiB | `src/hardware/arx5-sdk/models/meshes/X5/link1.STL` |
| 226,720 | 221.4 KiB | `src/hardware/arx5-sdk/lib/aarch64/libhardware.so` |
| 181,298 | 177.0 KiB | `src/inference/inference_direct.py` |
| 159,496 | 155.8 KiB | `src/hardware/arx5-sdk/lib/aarch64/libsolver.so` |
| 146,384 | 143.0 KiB | `src/hardware/arx5-sdk/models/meshes/X5/link4.STL` |
| 92,608 | 90.4 KiB | `src/hardware/arx5-sdk/lib/x86_64/libsolver.so` |
| 92,592 | 90.4 KiB | `src/hardware/arx5-sdk/lib/x86_64/libsolver_20_04.so` |
| 77,549 | 75.7 KiB | `src/camera_utils/camera_server.py` |
| 74,495 | 72.7 KiB | `src/hardware/arx5-sdk/python/examples/record_torque_latency.py` |
| 66,418 | 64.9 KiB | `src/recording/record_episode_single.py` |
| 62,011 | 60.6 KiB | `src/hardware/arx5-sdk/python/examples/record_replay_hybrid_cartesian.py` |
| 61,503 | 60.1 KiB | `src/policy/imitate_episodes_train.py` |
| 61,389 | 60.0 KiB | `src/hardware/arx5-sdk/python/examples/record_cartesian_pd_motion.py` |
| 53,865 | 52.6 KiB | `src/recording/distributed_recording.py` |
| 51,424 | 50.2 KiB | `src/hardware/arx5-sdk/lib/x86_64/libhardware_20_04.so` |
| 51,424 | 50.2 KiB | `src/hardware/arx5-sdk/lib/x86_64/libhardware.so` |
| 51,100 | 49.9 KiB | `src/visualization/visualize_episode.py` |

---

## 4. Tree (depth 3)

Condensed from the output of `tools/inventory.py tree 3` (file counts and sizes are recursive; several
files are put on one line and per-file sizes are dropped below depth 2):

```
./  (399 files, 64.2 MiB)
    .github/  (1 files, 707 B)
        workflows/  (1 files, 707 B)
            tests.yml  [707 B]
    calibration/  (12 files, 25.2 KiB)
        masking/  (9 files, 13.3 KiB)
            head-to-robot_video_left_hand/  (1 files, 1.3 KiB)
            head-to-robot_video_right_hand/  (1 files, 1.3 KiB)
            robot_video_left_hand/  (1 files, 256 B)
            robot_video_right_hand/  (1 files, 256 B)
            video_left_intri_imgs/  (1 files, 519 B)
            video_overhead_intri_imgs/  (1 files, 519 B)
            video_right_intri_imgs/  (1 files, 519 B)
            README.md  [4.7 KiB]
            masking.yaml  [3.9 KiB]
        README.md  [1.0 KiB]
        camera_params.npz  [8.8 KiB]
        start_position.npz  [2.1 KiB]
    docs/  (13 files, 69.2 KiB)
        arm_bringup.md  camera_settings_contract.md  collection_inference_contracts.md
        episode_storage.md  hand_guided_recording.md  image_preprocessing.md
        inference_layout.md  raw_training.md  recording_rate_metadata.md
        recording_review_replay.md  recording_timestamps.md  replay_timestamps.md  safe_exit.md
    scripts/  (91 files, 633.6 KiB)
        cameras/  (6 files, 69.3 KiB)
            README.md  camera_int.py  camera_servers_tmux.sh  configure_camera_controls.sh
            livestream_camera.py  tune_cameras.py
        config/  (4 files, 1.8 KiB)
            check_inference_config.py  check_task_config.py  check_training_config.py  create_config.py
        diagnostics/  (2 files, 1.5 KiB)
            README.md  write_recording_rate_metadata.py
        policy/  (1 files, 458 B)
            imitate_episodes_train.py
        setup/  (9 files, 86.3 KiB)
            activate_arm  arx_arms.sh  calibrate_arx_arm.py  calibrate_masking.py
            hold_start_position.py  packages.sh  paths.sh  set_up_recording.py  tune_masking.py
        tests/  (62 files, 461.8 KiB)
            task_fixtures/  (7 files, 2.8 KiB)
            conftest.py  episode_fixtures.py  fake_streams.py  test_*.py (52 files)
        _launcher.py  infer.py  practice.py  record.py  run.py  tune.py  visualize.py
    src/  (274 files, 62.2 MiB)
        camera_utils/  (10 files, 152.8 KiB)
            README.md  __init__.py  camera_int.py  camera_server.py  camera_settings_contract.py
            camera_timing.py  capture.py  cpu_latency.py  robotmq_camera_cache.py  sources.py
        config/  (31 files, 126.5 KiB)
            inference/  (6 files, 5.8 KiB)    README.md + 5 profiles
            tasks/  (3 files, 5.3 KiB)        README.md, single_arm.yaml, bimanual.yaml
            training/  (5 files, 10.3 KiB)    README.md + 4 recipes
            README.md  __init__.py  check_inference_config.py  check_task_config.py
            check_training_config.py  constants.py  create_config.py  inference_config.py
            masking_config.py  paths.py  rtc_config.py  run.py  supported_policy.py
            task_config.py  task_defaults.yaml  training_args.py  training_config.py
        contracts/  (10 files, 48.9 KiB)
            README.md  __init__.py  action_contract.py  action_rate_contract.py  capture_gc.py
            robot_contract.py  sequence_contract.py  standard_capture.py
            standard_image_processing.py  standard_recording.py
        data_processing/  (8 files, 53.5 KiB)
            README.md  __init__.py  dagger_review.py  episode_data.py  episode_inputs.py
            episode_store.py  recording_rate_metadata.py  video_processing.py
        hardware/  (131 files, 38.4 MiB)
            arx5-sdk/  (123 files, 38.3 MiB)   VENDORED (see §6)
            backends/  (2 files, 5.1 KiB)      __init__.py, arx.py
            __init__.py  arm_factory.py  arm_interface.py  check_gain.py  runtime.py  safe_exit.py
        inference/  (12 files, 273.5 KiB)
            README.md  __init__.py  dagger_controls.py  inference_direct.py  inference_remote.py
            inference_utils.py  lookahead_tuning.py  policy_chunks.py  policy_transport.py
            policy_warmup.py  serve_policy.py  shared_hardware.py
        kdl_parser/  (26 files, 22.7 MiB)      VENDORED (see §6)
            kdl_parser/  (9 files, 44.5 KiB)
            kdl_parser_py/  (13 files, 22.6 MiB)
            .gitignore  .travis.yml  LICENSE  README.md
        policy/  (12 files, 173.9 KiB)
            README.md  __init__.py  act.py  fast_flow_inference.py  flow_rtc.py  frame_store.py
            imitate_episodes_train.py  mixed_precision.py  model_util.py  policy.py
            replay_data.py  training_utils.py
        recording/  (12 files, 181.0 KiB)
            README.md  __init__.py  distributed_recording.py  episode_buffer.py  pedal_keys.py
            record_episode_single.py  record_hardware.py  recording_alignment.py
            recording_buffer.py  recording_replay.py  recording_utils.py  replay_practice.py
        utils/  (10 files, 62.2 KiB)
            README.md  __init__.py  hardware_utils.py  masking.py  masking_calibration.py
            masking_tuner.py  random_utils.py  replay_utils.py  run_log.py  window_utils.py
        visualization/  (12 files, 115.6 KiB)
            visualization_utils/  (5 files, 51.3 KiB)
            README.md  __init__.py  episode_visualization.py  replay_tracking_graph.py
            rerun_utils.py  training_preview.py  visualize_episode.py
    .gitignore  [228 B]
    .python-version  [5 B]          "3.10"
    AGENTS.md  [5.8 KiB]
    LICENSE  [1.0 KiB]
    README.md  [30.7 KiB]
    arx_setup.sh  [59 B]
    pyproject.toml  [1.3 KiB]
    uv.lock  [1.2 MiB]
```

Inside `src/hardware/arx5-sdk/`: `models/` 33 files / 36.8 MB, `wheels/` 15 / 2.0 MB, `lib/` 6 / 0.67 MB,
`python/` 28 / 0.41 MB, `src/` 8 / 135 KB, `tests/` 6 / 84 KB, `include/` 12 / 76 KB, `examples/` 3,
`conda_environments/` 5, `cmake/` 1, plus `CMakeLists.txt`, `README.md`, `CHANGES_FROM_UPSTREAM.md`,
`LICENSE`, `env.sh`, `.gitignore`.

---

## 5. Licences

| File | Identity | Evidence |
| --- | --- | --- |
| `LICENSE` (root, 21 lines) | **MIT License**, "Copyright (c) 2026 John Yao" | `LICENSE:1,3`. Lines 4–21 are byte-identical to the vendored arx5-sdk MIT text (`diff` shows only the copyright line differs). sha256 `21c95940…f718124`. `pyproject.toml:6` says `license = { text = "MIT" }`. GitHub API reports SPDX `MIT`. |
| `src/hardware/arx5-sdk/LICENSE` | **MIT License**, "Copyright (c) 2024 Yihuai Gao" | byte-identical to upstream real-stanford/arx5-sdk@79240efc (same git blob id) |
| `src/kdl_parser/LICENSE` | **BSD 3-Clause License**, "Copyright (c) 2008, Willow Garage Inc." / "Copyright (c) 2015-2017, Open Source Robotics Foundation" | byte-identical to upstream ros/kdl_parser tag 2.13.0 |

**[README CLAIM]** `README.md:604-606`: "This repository is released under the MIT License ([LICENSE](LICENSE)). Two
components are included with local modifications under their own licences, which continue to apply to
those directories."

**[NOT ESTABLISHED]** The licence of the prebuilt `libhardware.so` / `libsolver.so` binaries themselves
(they ship inside the MIT-licensed upstream SDK tree, without source). No licence is stated for the mesh
and URDF model files beyond the directory licence.

---

## 6. Vendored code and its provenance (checked by content hash)

Local git blob ids were recomputed from the work-tree bytes and compared with the upstream trees
returned by the public GitHub API (`tools/upstream_compare.py`; saved answers in
`~/research/flash-aloha/web/download-step/github-api/`).

### `src/hardware/arx5-sdk` vs `real-stanford/arx5-sdk` @ `79240efc2e0d75fdc1e50a8b883c5e928fbfc681`

The base commit is the one named in `src/hardware/arx5-sdk/CHANGES_FROM_UPSTREAM.md:4-5`.

| Category | Files | Notes |
| --- | --- | --- |
| identical to upstream | 60 (28,102,518 bytes) | **all 10 `.so` libraries**, all 9 STL meshes, 10 of the URDFs, `LICENSE`, `.gitignore`, the 5 conda environment files, the 4 Dockerfiles, `wheels/{setup.py,pyproject.toml,__init__.py,build_*.sh,upload_to_pypi.sh}`, the CAN headers (`include/hardware/*.h`, `include/libcan/*.h`), `include/app/solver.h`, `include/utils.h`, 2 C++ examples, 5 Python examples and `python/peripherals/keystroke_counter.py` |
| modified | 24 | `CMakeLists.txt`, `README.md` (23 KB vs 12 KB upstream), `include/app/{cartesian_controller,common,config,controller_base,joint_controller}.h`, `models/L5.urdf`, `python/{CMakeLists.txt,arx5_interface.pyi,arx5_pybind.cpp}`, 8 Python examples, `src/app/{cartesian_controller,controller_base,joint_controller}.cpp`, `src/utils.cpp`, `wheels/CMakeLists.txt` |
| only local (added) | 39 | `CHANGES_FROM_UPSTREAM.md`, `cmake/SelectSdkLibDir.cmake`, `env.sh`, `include/app/cartesian_pd.h`, `src/app/{cartesian_ik.cpp,cartesian_ik.h,cartesian_pd.cpp,friction_config.cpp}`, `models/Trossen_L5.urdf`, `models/Trossen_L5_cartesian_pd.json`, `models/mujoco/ARX_L5/` (XML + 10 `.obj`), 11 Python examples (tuners, recorders, simulator), 6 tests, 1 C++ example |
| only upstream (not carried) | 12 | `.vscode/*`, `python/communication/zmq_{client,server}.py`, `python/examples/{arx5_zmq,spacemouse_teleop}.py`, `python/peripherals/spacemouse_shared_memory.py`, `python/shared_memory/*` (4), `update_logs.md` |

This matches **[DOC CLAIM]** `CHANGES_FROM_UPSTREAM.md:7-8` "The vendor libraries in `lib/` and the low-level
CAN headers are unchanged" and `:38-40` (the list of removed upstream files) — both **[VERIFIED]** by hash.
The same file lists ten groups of local changes (`:14-23`), among them: a cubic-interpolation fix,
start-up/shutdown locking, "Per-model velocity scaling … Upstream would need to confirm the motor firmware
ranges this assumes" (`:16`), friction feed-forward loaded from a URDF block (`:17`), gravity/friction tuners
(`:18`), "Cartesian torque control … Largest change" (`:19`), host joint damping and telemetry (`:20`), build
changes (`:21`), a wider gripper sanity tolerance (`:22`) and a changed default in `calibrate.py` (`:23`).
"Local values, not for upstreaming" (`:25-34`) include a `Trossen_L5` profile and a changed `L5` gripper
calibration ("Open width 0.100 m (upstream 0.088 m)").

### `src/kdl_parser` vs `ros/kdl_parser` tag `2.13.0`

The README gives no base version; `kdl_parser/package.xml` says `<version>2.13.0</version>` and the tag matches.

| Category | Files | Notes |
| --- | --- | --- |
| identical to upstream | 21 (23,777,632 bytes) | everything under `kdl_parser/`, the tests, `LICENSE`, `.travis.yml`, **and the 23.7 MB `kdl_parser_py/test/kdl_parser_py.urdf`** |
| modified | 2 | `README.md` (one note line added at the top); `kdl_parser_py/kdl_parser_py/urdf.py` (one line: `kdl.Joint.None` → `getattr(kdl.Joint, 'Fixed') if hasattr(kdl.Joint, 'Fixed') else getattr(kdl.Joint, 'None')`, a Python 3 syntax fix) |
| only local | 3 | `.gitignore`, `kdl_parser_py/pyproject.toml`, `kdl_parser_py/setup.cfg` |
| only upstream | 0 | |

The Python part is put on `PYTHONPATH` by `scripts/setup/paths.sh:29`; `pyproject.toml:46` excludes both
vendored trees from the installable package (`exclude = ["hardware.arx5-sdk*", "kdl_parser*"]`).

---

## 7. Git LFS, submodules, large files skipped

- **LFS pointers: 0** (`tools/inventory.py lfs` — no file starts with `version https://git-lfs`). No `.gitattributes`.
- **Submodules: none** (no `.gitmodules`, no gitlinks).
- **Nothing was skipped or left out.** The clone is complete; the largest file is 22.6 MiB.
- `.gitignore:9-16` excludes `ckpts/`, `data/`, `logs/`, `wandb/`, `*.rrd` — datasets, checkpoints and logs are
  not part of the repository (**[README CLAIM]** `README.md:68-70`: "Datasets, checkpoints and logs default to
  `data/`, `ckpts/` and `logs/` in the repository root and are not part of the repository.").

---

## 8. Outside links

### 8.1 In the project's own README and docs (complete list, `tools/links.py --docs-only`)

| Link | Where | Kind |
| --- | --- | --- |
| `johnyao93.github.io/flash-aloha` (bare text, not a Markdown link) | `README.md:3` | project page |
| `https://github.com/real-stanford/arx5-sdk` | `README.md:610` | vendored dependency's repository |
| `https://github.com/ros/kdl_parser` | `README.md:611` | vendored dependency's repository |
| `https://docs.kernel.org/networking/can.html` | `docs/arm_bringup.md:46` | Linux SocketCAN documentation |

That is all. The 13 files in `docs/`, `AGENTS.md`, `calibration/**/README.md`, `scripts/**/README.md` and the
`src/*/README.md` files of the project's own packages contain **no other outside link**. All 147 relative
Markdown links in the repository resolve to existing files and headings (`tools/mdlinks.py`: 0 broken).

**Not linked anywhere in the README/docs:** a paper, a dataset, model weights/checkpoints, videos, or the
repositories of the Python dependencies. OpenPI / Pi0.5 is named (`README.md:11-12`) without a link.
Dependencies are named only as PyPI requirements in `pyproject.toml:9-31`:
`av`, `evdev`, `getkey`, `ipython`, `lerobot==0.4.4`, `matplotlib`, `msgpack`, `opencv-python`, `pyyaml`,
`robotmq==0.1.15`, `scipy`, `torch==2.7.1`, `torchvision==0.22.1`, `tqdm`, `transformers>=4.50.3,<4.52.0`,
`urdf-parser-py`, `wandb`, `websockets`, `zarr>=2.18.3,<3`, `numpy`, `rerun-sdk==0.26.2`; dev extra `pytest`.

### 8.2 In the vendored READMEs and docs

`src/hardware/arx5-sdk/CHANGES_FROM_UPSTREAM.md`:
- `https://github.com/real-stanford/arx5-sdk` (`:4`)
- `https://github.com/real-stanford/arx5-sdk/tree/79240efc2e0d75fdc1e50a8b883c5e928fbfc681` (`:5`)

`src/hardware/arx5-sdk/README.md` (upstream README with local sections):
- `https://github.com/real-stanford/arx5-sdk` (`:1`)
- `https://github.com/real-stanford/arx5-sdk/blob/709f7ab7429f97c83e18687e650f3ee77d14719a/python/examples/test_joint_control.py#L31` (`:10`)
- `https://jimmyyhwu.github.io/` (`:15`), `https://cheng-chi.github.io/` (`:22`, `:25`) — acknowledgements
- `https://github.com/conda-forge/miniforge?tab=readme-ov-file#install` (`:268`)
- "Projects Using this Repository" (`:420-427`): `https://umi-on-legs.github.io/`, `https://umi-gripper.github.io/`,
  `https://github.com/real-stanford/umi-arx`, `https://unified-video-action-model.github.io/`,
  `https://github.com/real-stanford/umi-arx/tree/uva`, `https://vision-in-action.github.io/`,
  `https://latentpolicybarrier.github.io/`, `https://dynaguide.github.io/`,
  `https://minimalist-compliance-control.github.io/`, `http://gated-memory-policy.github.io/`
- citation target `https://umi-on-legs.github.io/` (`:432`)

`src/kdl_parser/README.md`: `https://github.com/ros/kdl_parser` (`:1`), `http://wiki.ros.org/kdl_parser` (`:8`),
`http://wiki.ros.org/kdl_parser_py` (`:9`), `https://github.com/ros/robot_model` (`:11`),
`https://github.com/ros/robot_model/issues/195` (`:12`).

`src/kdl_parser/kdl_parser{,_py}/CHANGELOG.rst`: 26 issue links, all under `https://github.com/ros/kdl_parser`
(13), `https://github.com/ros/robot_model` (7) or `https://github.com/ros2/kdl_parser` (6).

### 8.3 In code and configuration (`tools/links.py --skip-lock`)

The project's **own** code contains no hard-coded outside URL at all; the only match is the template
`ws://{host}` at `src/inference/policy_transport.py:147`. Every other URL in non-documentation files is in the
vendored trees:

- `https://github.com/OpenEtherCATsociety/SOEM.git` — the 4 `wheels/Dockerfile.*:14` and the 5 conda environment files (comment)
- `https://github.com/ros/kdl_parser.git` — the 4 `wheels/Dockerfile.*:19`
- `https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz` — pip requirement in the 5 `conda_environments/py3*_environment.yaml` (e.g. `py310_environment.yaml:30`)
- `quay.io/pypa/manylinux_2_28_{x86_64,aarch64}:2025.09.01-1` — Docker base images, `wheels/Dockerfile.*:1`
- `https://github.com/yihuai-gao/arx5-sdk` — `wheels/pyproject.toml:23`
- `https://opensource.org/licenses/MIT` — `wheels/setup.py:5`
- `http://wiki.ros.org/sw_urdf_exporter` — comment in 7 URDF files
- `https://github.com/ros2/kdl_parser`, `https://github.com/ros2/kdl_parser/issues`, `http://ros.org/wiki/kdl_parser[_py]` — `src/kdl_parser/*/package.xml`
- `https://gcc.gnu.org/wiki/Visibility` — `visibility_control.hpp:40`
- `https://github.com/ros/urdf_tutorial/` — `kdl_parser_py/test/test.urdf:2`
- `http://packages.ros.org/ros/ubuntu`, `http://packages.ros.org/ros.key`, `https://github.com/catkin/catkin_tools/issues/245` — `src/kdl_parser/.travis.yml:15,16,38`

`uv.lock`: 175 package entries, 174 from `registry = "https://pypi.org/simple"` and 1 editable (the project). The only
hosts are `files.pythonhosted.org` (3,844 URLs) and `pypi.org` (251). No git, URL or path sources.

### 8.4 On the project page (outside the repository; fetched once, 2026-11-14 03:53 UTC)

`https://johnyao93.github.io/flash-aloha/` answered HTTP 200 (16,192 bytes, `last-modified: Thu, 12 Nov 2026
02:32:06 GMT`). The GitHub API says this repository has `has_pages: false`, so the page is served from
somewhere else under the same account (**[NOT ESTABLISHED]** which repository). Links found on it:

| Label on the page | Link |
| --- | --- |
| Paper | `/static/files/flash-aloha.pdf?v=2` → `https://johnyao93.github.io/static/files/flash-aloha.pdf` (not fetched in this step) |
| Hardware | `https://docs.google.com/document/d/1GAxYQq6ppRNMVVDRZRWVP7z5HXIXlONxjdPNFYmqPC0/edit?usp=sharing` (not fetched) |
| Code | `https://github.com/JohnYao93/flash-aloha` |
| arXiv | text only: "arXiv (coming soon)" — no link |
| "ALOHA Lightning" | `https://ieeexplore.ieee.org/abstract/document/11696199` ("First presented as ALOHA Lightning at ICRA in Vienna on July, 2026") |
| Authors | John Hua Yao (`/`), Qi Wu (`https://scholar.google.com/citations?user=3wFmCXAAAAAJ&hl=zh-CN`), Yihuai Gao (`https://yihuai-gao.github.io/`), Chelsea Finn (`https://ai.stanford.edu/~cbfinn/`), Zipeng Fu (`https://zipengfu.github.io/`) |
| Videos (16 distinct `.mp4`) | `/static/videos/flash-aloha/{shorts,cups,battery,table,skill-2491,skill-6270,skill-6349,skill-6296,skill-6361,toss-catch,puppeteer-normal,puppeteer-double,puppeteer-skipping,handheld-real,handheld-sim,catching}.mp4` plus poster images under `/static/images/flash-aloha/` |
| Other | Google Fonts stylesheet; local `style.css`, `sections.js`, `cad.js` |

**No dataset link and no model-weights link on the project page either.**

---

## 9. What the README says the project is (its claims, quoted)

**[README CLAIM]** `README.md:1-12`:

> # Flash ALOHA: Learning Fast and Precise Manipulation
>
> Project website: johnyao93.github.io/flash-aloha
>
> Data collection, training and deployment for bimanual and single-arm manipulation
> with ARX arms (L5 and Trossen_L5): camera servers, hand-guided recording, ACT
> and Flow policy training, direct and remote inference, and visualization.
>
> Two policies are included: decoder-only ACT (the default) and Flow, which uses the
> LeRobot U-Net/visual backbone. Both support training-time RTC prefix conditioning
> and optional image/proprioceptive masking. The remote client can also execute an
> OpenPI server's policy (for example Pi0.5); see `docs/image_preprocessing.md`.

Further claims, each quoted from the README:

- `README.md:16-17`: "Commands that move a robot are marked operator-run: read the section before running them."
- `README.md:41-42`: "`single_arm` (one ARX arm, two cameras) and `bimanual` (two ARX arms, three cameras) are the example tasks in `src/config/tasks/`; copy one to describe your own setup."
- `README.md:55-56`: "The tests never start a camera or a robot controller. `.python-version` selects Python 3.10."
- `README.md:58-60`: "Build the ARX SDK's Python extension on the robot computer following [src/hardware/arx5-sdk/README.md] (its Conda environment supplies the native dependencies)."
- `README.md:64-65`: "The files under `calibration/` were measured on one physical setup and are examples only: recalibrate for your own arms and cameras before use"
- `README.md:278-281`: "Task YAML owns cameras, arms/CAN interfaces, action rate and prediction horizon. Training YAML owns model/training settings. Inference YAML owns execution windows, blending, RTC and recording choices. The same shared contracts enforce consistency between collection, training and deployment."
- `README.md:298-299`: "Inference loads that file and refuses to run if the task or the profile disagrees with it"
- `README.md:356-361`: "Both ACT and Flow support training-time real-time chunking (RTC). … at inference each new chunk is conditioned on the previous plan's next actions, so chunks join without a jump while the policy query runs in the background."
- `README.md:411-418`: "The transport uses a NumPy/MessagePack WebSocket protocol; no OpenPI checkout is required for this repository's own server. … This repository does not export data for or train OpenPI"
- `README.md:553-554`: "Recording review/replay, generic DAgger collection, raw-episode training, training-resolution previews and Rerun visualization are available."
- `README.md:563-566`: "A policy can be trained with parts of its inputs hidden: image regions that follow the arms, fixed image regions, whole cameras, or the proprioception. … Strategies are defined as data in `calibration/masking/masking.yaml`, not code"
- `README.md:322-323`: "[AGENTS.md](AGENTS.md) makes these contracts binding on coding agents (and is a good checklist for human contributors)."

The README's quick start (`README.md:19-39`) is: `uv sync --extra dev && source .venv/bin/activate && source
scripts/setup/paths.sh`, `python -m pytest`, then `uv run scripts/tune.py gravity L5 can0` / `friction`,
`uv run scripts/practice.py --arm right`, `uv run src/recording/replay_practice.py --arms right`,
`uv run scripts/setup/set_up_recording.py --task_name single_arm`, `python scripts/run.py collect single_arm`,
`python scripts/run.py train single_arm_act`, `python scripts/run.py infer single_arm_act`.

### The project page's description (outside the repository)

**[PROJECT PAGE CLAIM]** meta description: "A system for learning near-human-speed bimanual manipulation from
demonstrations." Abstract, quoted in full:

> Learning from human demonstrations has enabled robots to acquire a wide range of manipulation skills, but
> learned policies typically execute far slower than ordinary humans. This speed gap is mainly due to lack of an
> interface for precisely collecting demonstration data at high speed, and the difficulty in training policies
> that can robustly execute high-speed motions. In this paper, we present Flash ALOHA, a practical system for
> learning fast and precise robotic manipulation. Our system uses kinesthetic teaching to intuitively collect
> accurate and near-human-speed demonstrations on a backdrivable bimanual platform, yielding natural and fast
> trajectories. Our learning pipeline consists of two components: a lightweight smoothing algorithm to address
> tracking instabilities produced by action chunking, and for visual alignment we mask out the operator visible
> in the camera views. Given 50 demonstrations for each task, Flash ALOHA autonomously completes bimanual tasks
> such as ball tossing and catching, folding shorts, battery insertion, and bussing tables at over 80% success
> rates while close to human speed.

None of the numbers in that abstract (50 demonstrations, 80 % success, "close to human speed") appears in the
repository, and the repository contains no data, weights or evaluation results with which to check them.

---

## 10. What the checkout shows for the main claims (static reading only)

| README/doc claim | What the files show |
| --- | --- |
| Two policies, ACT the default | **[VERIFIED]** `class ACTPolicy` (`src/policy/act.py:159`), `class LeRobotFlowPolicy` (`src/policy/policy.py:295`); factory `POLICY_CLASSES = ('Flow', 'ACT')` and "Only Flow and ACT are supported" (`src/policy/model_util.py:2,12`); `--policy_class` default `"ACT"`, choices `["ACT", "Flow"]` (`src/config/training_args.py:13`) |
| Flow "uses the LeRobot U-Net/visual backbone" | **[VERIFIED]** `src/policy/policy.py:26-58` imports `DiffusionPolicy` / `DiffusionConfig` from `lerobot.common.policies.diffusion…` or `lerobot.policies.diffusion…`; `lerobot==0.4.4` pinned (`pyproject.toml:14`) |
| ARX arms L5 and Trossen_L5 | **[VERIFIED]** model files `src/hardware/arx5-sdk/models/L5.urdf`, `Trossen_L5.urdf`; both example tasks use `model: L5` (`src/config/tasks/single_arm.yaml:14`, `bimanual.yaml:14,17`) |
| `single_arm`: one arm, two cameras; `bimanual`: two arms, three cameras, 100 Hz | **[VERIFIED]** `single_arm.yaml:7,13-19,22` (`control_freq: 100`, left/`can0`, `[video_overhead, video_left]`, 15 s); `bimanual.yaml:7,13-20,23` (`can0`, `can1`, three cameras, 30 s) |
| `.python-version` selects Python 3.10 | **[VERIFIED]** the file contains `3.10`; `pyproject.toml:8` `requires-python = ">=3.10"` |
| "The tests never start a camera or a robot controller" | **[VERIFIED in part]** autouse fixtures in `scripts/tests/conftest.py:30-35` make `DistributedRecorder.__init__` raise ("Tests must inject synthetic streams…") and `:77-89` replace the `arx5_interface` module with a stub whose controllers raise ("Tests must inject simulated hardware"). **Not established** that no test reaches hardware by another path; the suite does open loopback sockets, spawn subprocesses and use `/dev/shm` (§12.9). Tests were not run. |
| Direct profiles blend with sigmoid, RTC off | **[VERIFIED]** `src/config/inference/{single_arm_act,single_arm_flow,bimanual_act}.yaml:8-13` (`method: sigmoid`, `enabled: false`) |
| `*_flow` recipes enable training-time RTC, `*_act` leave it off; prefix default 10 | **[VERIFIED]** `training/*_flow.yaml:29` `training_time_rtc: true`; `*_act.yaml:28` `false`; `--training_time_rtc_max_delay` default 10 (`src/config/training_args.py:74`) |
| `--action_lookahead` default 10, `--gripper_lookahead` default 5 | **[VERIFIED]** argparse default is `None` (`src/inference/inference_direct.py:3443,3446`), resolved to 10 and 5 at `:2267-2268` |
| NumPy/MessagePack WebSocket transport; no OpenPI checkout needed | **[VERIFIED]** `src/inference/policy_transport.py:7-9,19-48` (msgpack + numpy + `websockets`); no module in the repository imports `openpi*` (`tools/imports.py`) |
| Remote client can talk to an OpenPI server | **[VERIFIED that the client code exists]** `class WebsocketOpenPIClient` (`policy_transport.py:142`), `OPENPI_PROFILE = "openpi_letterbox_448"`, `openpi_image_contract`, `openpi_request_images` (`src/contracts/standard_image_processing.py:135,160,201`), profile `src/config/inference/single_arm_openpi_remote.yaml`. No OpenPI server, trainer or checkpoint is in the repository. |
| Masking strategies are data in `masking.yaml` | **[VERIFIED]** `calibration/masking/masking.yaml:75-84` defines 9 strategies; `class Masker` (`src/utils/masking.py:72`) |
| Failure exit "implemented once in `src/hardware/safe_exit.py`" (`docs/safe_exit.md:3-4`) | **[VERIFIED that the module exists]** `damp_until_home`, `damping_exit` (`src/hardware/safe_exit.py:93,130`). Behaviour not exercised. |
| Documented gains and limits | **[VERIFIED]** `ACTIVE_KP = (200,200,200,100,100,100,10)`, `ACTIVE_KD = (7,7,7,1.5,1.5,1.5,1)` (`src/utils/hardware_utils.py:10-11`); camera limits 50/50/5 ms (`src/camera_utils/camera_timing.py:72-73`); practice loop 200 Hz (`scripts/practice.py:114`) |
| One-command launcher | **[VERIFIED]** `scripts/run.py` re-executes `src/config/run.py`, whose sub-commands are `collect`, `train`, `infer`, `serve`, `view`, `replay` (`src/config/run.py:172-185`) |
| CI | **[VERIFIED]** `.github/workflows/tests.yml:12-24`: `ubuntu-22.04`, `uv sync --locked --extra dev`, `uv run python -m pytest -q`; its comment says "tests that need PyKDL or an OpenPI checkout skip" |

Everything that requires running code — whether the tests pass, whether training converges, the timing
figures in the docs (for example `src/config/inference/README.md:47` "a Flow query takes ~17 ms instead of
~70 ms"), whether the safety behaviour works on a real arm — is **[NOT ESTABLISHED]**.

---

## 11. Entry points

69 Python files have an `if __name__ == "__main__"` guard (`tools/imports.py --mains`); the ones a user is
pointed at are listed here. "Moves arms" = the docs mark it operator-run.

### 11a. Top-level launchers (`scripts/`)

| Command | What it starts | Evidence | Moves arms |
| --- | --- | --- | --- |
| `scripts/run.py {collect,train,infer,serve,view,replay} NAME [--dry-run]` | re-executes `src/config/run.py`, which resolves names and execs one of the launchers below | `scripts/run.py:7-11`; `src/config/run.py:1-12,172-185` | collect / infer / replay: yes |
| `scripts/record.py single` | `src/recording/record_episode_single.py` (hand-guided collection + review replay) | `scripts/record.py:11-15` | yes |
| `scripts/record.py publish` | `src/recording/distributed_recording.py` (offline publication) | same | no (per docs) |
| `scripts/infer.py direct` | `src/inference/inference_direct.py` (load checkpoint, run robot) | `scripts/infer.py:12-16` | yes |
| `scripts/infer.py remote` | `src/inference/inference_remote.py` (robot-side WebSocket client) | same | yes |
| `scripts/infer.py serve` | `src/inference/serve_policy.py` (model server, default `0.0.0.0:8000`) | same; `serve_policy.py:445-446` | no |
| `scripts/visualize.py episode` | `src/visualization/visualize_episode.py` (Rerun view; `--action_replay` / `--review_replay` move arms) | `scripts/visualize.py:11-14` | optional |
| `scripts/tune.py {gravity,friction} MODEL INTERFACE` | vendored SDK tuners `…/python/examples/tune_gravity_compensation.py`, `tune_gripper_friction_compensation.py`; they write into the model URDF | `scripts/tune.py:18-23`; `docs/arm_bringup.md:133-134` | yes (motors enabled, zero position gain) |
| `scripts/practice.py --arm {left,right,both}` | gravity-compensated manual practice, no cameras | `scripts/practice.py`; `README.md:139-153` | yes |
| `scripts/policy/imitate_episodes_train.py --training-config NAME` | `src/policy/imitate_episodes_train.py` (training) | `scripts/policy/imitate_episodes_train.py:7-11` | no |
| `scripts/config/{check_task_config,check_training_config,check_inference_config,create_config}.py` | same-named files in `src/config/` (offline validation / scaffolding) | each `:7-11` | no |
| `scripts/diagnostics/write_recording_rate_metadata.py` | writes `meta/recording_rate.json` into a LeRobot export | file `:12-20` | no |

All of these launchers pass control with `os.execv(sys.executable, …)` (for example `scripts/_launcher.py:39`).

### 11b. Setup and camera tools

| File | Purpose (from the file / docs) | Notes |
| --- | --- | --- |
| `arx_setup.sh` | `source .venv/bin/activate; source ./scripts/setup/paths.sh` | 2 lines |
| `scripts/setup/paths.sh` | sets `PYTHONPATH` and `LD_LIBRARY_PATH` (SDK libs, `$HOME/miniconda3/envs/arx-py310/lib`, PyKDL build dir, vendored `kdl_parser_py`) | to be sourced |
| `scripts/setup/packages.sh` | `conda env create -f …/py310_environment.yaml`, `conda activate arx-py310`, `cmake ..`, `make -j` in the SDK | no shebang, no `set -e` |
| `scripts/setup/activate_arm {0|1}` | `sudo ip link set canN down / type can bitrate 1000000 / up` | `:11-13` |
| `scripts/setup/arx_arms.sh` | `sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up` (and `can1`) | `:1-2` |
| `scripts/setup/set_up_recording.py` | camera preview and saving start poses (`calibration/start_positions/<task>.npz`) | moves/holds arms |
| `scripts/setup/hold_start_position.py`, `calibrate_arx_arm.py`, `calibrate_masking.py`, `tune_masking.py` | hold pose, arm/camera calibration, mask tuning | arm-moving except `--solve-only` / `--episode` modes (per docs) |
| `scripts/cameras/camera_servers_tmux.sh {start,tail,status,attach,stop}` | starts one `uv run src/camera_utils/camera_server.py --camera NAME --write-shared-memory` per camera in tmux session `camera_servers` | `:15,156,158,315` |
| `scripts/cameras/configure_camera_controls.sh` | `v4l2-ctl -d /dev/video_{left,right,overhead} --set-ctrl=exposure_dynamic_framerate=0` | `:2-4` |
| `scripts/cameras/tune_cameras.py`, `livestream_camera.py`, `camera_int.py` | tune controls and save `camera_params.npz`; preview; intrinsics | use `v4l2-ctl` via `subprocess.run([...])` |
| `src/camera_utils/camera_server.py` | the camera server itself (V4L2 capture, shared memory, video logging) | 2,062 lines |
| `src/recording/replay_practice.py [--graph]`, `src/recording/record_hardware.py --target-provider module:factory`, `src/visualization/replay_tracking_graph.py LOGDIR` | record/replay practice; recording of externally generated targets; tracking graph | first two move arms |
| `src/hardware/check_gain.py` | prints SDK default gains ("does not connect to an arm", `docs/arm_bringup.md:83`) | |

### 11c. Configuration files

| File | Content |
| --- | --- |
| `src/config/task_defaults.yaml` | `schema_version: 1`, `camera_settings: {}`, observations `[qpos]`, outputs `[action]` |
| `src/config/tasks/single_arm.yaml` | left arm `L5` on `can0`, cameras `[video_overhead, video_left]`, 100 Hz, 15 s takes, 7-dim state/action, data in `../../../data/single_arm` |
| `src/config/tasks/bimanual.yaml` | `L5` on `can0` + `can1`, three cameras, 100 Hz, 30 s takes, 14-dim |
| `src/config/training/single_arm_act.yaml`, `bimanual_act.yaml` | `type: ACT`, `resnet18`, `hidden_dim: 512`, `dec_layers: 7`, `chunk_size: 100`, batch 128, lr 1.2e-4, 100k steps, **`wandb: true`**, `training_time_rtc: false`, `compile: true`, 16 loader workers |
| `src/config/training/single_arm_flow.yaml`, `bimanual_flow.yaml` | `type: Flow`, `pretrained_backbone_weights: IMAGENET1K_V1`, `unet_depth: 3`, `chunk_size: 128`, batch 128, 100k steps, **`wandb: true`**, `training_time_rtc: true` |
| `src/config/inference/single_arm_act.yaml`, `bimanual_act.yaml`, `single_arm_flow.yaml` | `backend: direct`, checkpoint `../../../ckpts/<name>/all_configs.json`, 50-step chunks, sigmoid blending, RTC off |
| `src/config/inference/single_arm_remote.yaml` | `backend: remote`, `localhost:8000`, async, `overlap_steps: 10`, RTC on with `prefix_steps: 10`, blending none |
| `src/config/inference/single_arm_openpi_remote.yaml` | `backend: remote` against an OpenPI server, `overlap_steps: 30`, `prefix_steps: 20`, placeholder prompt |
| `calibration/masking/masking.yaml` | calibration of one bimanual setup + 9 masking strategies |
| `calibration/camera_params.npz`, `start_position.npz`, `masking/**/*.npy|npz` | example calibration values (see §12.4) |
| `scripts/tests/task_fixtures/*.yaml` (7) | test-only task definitions |
| `pyproject.toml`, `uv.lock`, `.python-version` | packaging (`name = "flash_aloha"`, version `0.1.0`), 175 lock entries, Python 3.10 |
| `.github/workflows/tests.yml` | CI offline tests |
| vendored: `src/hardware/arx5-sdk/conda_environments/py3{8,9,10,11,12}_environment.yaml`, `CMakeLists.txt` (3), `wheels/Dockerfile.*` (4), `wheels/{pyproject.toml,setup.py}`, `models/Trossen_L5_cartesian_pd.json` | SDK build/packaging |

Environment variables read by the project's own code: `FLASH_ALOHA_DATA_DIR`, `FLASH_ALOHA_CKPT_DIR`,
`FLASH_ALOHA_LOG_DIR` (`src/config/paths.py:25-27`), `FLASH_ALOHA_TASK_DIRS`, `FLASH_ALOHA_CAMERA_PARAMS`,
`FLASH_ALOHA_CAMERA_CACHE`, `FLASH_ALOHA_MASKING_CONFIG`, `FLASH_ALOHA_VIDEO_INDEX_CACHE`,
`FLASH_ALOHA_EPISODE_CACHE_SIZE`, `FLASH_ALOHA_FRAME_CACHE_MB`, `FLASH_ALOHA_FRAME_STORE_DIR`,
`FLASH_ALOHA_MIN_FREE_MEMORY_GB`, `CAMERA_MAX_AGE_MS`, `CAMERA_MAX_SKEW_MS`, `CAMERA_MAX_FUTURE_MS`,
`XDG_CACHE_HOME`, `DISPLAY`, `CUDA_VISIBLE_DEVICES`, `FFREPORT`; shell: `ARX_CONDA_PREFIX`, `OROCOS_KDL_PYTHON`,
`CAMERA_TMUX_SESSION`, `CAMERA_STARTUP_TIMEOUT`.

### 11d. Notebooks

None (no `.ipynb` file).

### 11e. Agent instruction file

`AGENTS.md` (94 lines, 5 sections) is addressed to coding agents working inside that repository: "Give the
user commands to run for robot operation. Never start a robot controller, collector, replay, or robot motion
yourself." (`AGENTS.md:3-4`), then mandatory reuse of `standard_capture.capture_observation`, the
garbage-collection pause, shared image preprocessing, and a "Protected recording and hardware code" rule
(`AGENTS.md:82-94`). For this research it is repository content, not instructions. Note for later steps: the
agent harness on this machine surfaced this file automatically as context when it was read; it was treated as
data.

---

## 12. Unusual or risky items

### 12.1 Binaries

**[VERIFIED]** `tools/inventory.py binaries` finds 28 files with NUL bytes:

1. **Ten prebuilt ELF shared libraries, 2,650,320 bytes, with no source in the repository.**
   `src/hardware/arx5-sdk/lib/x86_64/{libhardware,libhardware_20_04,libsolver,libsolver_20_04}.so`,
   `lib/aarch64/{libhardware,libsolver}.so`, `wheels/lib/{x86_64,aarch64}/{libhardware,libsolver}.so`.
   - Only headers exist for them (`include/hardware/arx_can.h`, `include/app/solver.h`); embedded source names
     are `arx_can.cpp`, `SocketCAN.cpp`, `math_ops.cpp`, `solver.cpp`.
   - All ten are byte-identical to upstream real-stanford/arx5-sdk@79240efc (§6). That establishes provenance,
     not safety: they are closed binaries.
   - They are linked into the `arx5_interface` Python extension (`src/hardware/arx5-sdk/CMakeLists.txt:52-54`,
     `python/CMakeLists.txt:45-47`) and put on `LD_LIBRARY_PATH` by `scripts/setup/paths.sh:8-14`, so every
     arm-controlling entry point loads them.
   - Read as bytes (`tools/elfinfo.py`, nothing loaded): x86-64 and AArch64 shared objects, not stripped;
     `libhardware` imports `socket`, `bind`, `ioctl`, `read`, `write` (plus `send`, `recv`, `setsockopt` in the
     aarch64 and wheel builds) and, in `lib/x86_64`, needs `libsoem.so` (EtherCAT); `libsolver` needs
     `liborocos-kdl` and `libkdl_parser`. Built with conda-forge GCC 14.3.0 / 12.4.0 and Red Hat GCC 14.2.1.
   - Six of them carry a RUNPATH into the upstream builder's directories
     (`/data/yihuai/miniforge3/envs/arx-src/lib`, `/home/yihuai/miniforge3/envs/arx-src/lib`,
     `/home/orangepi/miniforge3/envs/arx-src/lib`): the dynamic loader will look for dependencies there if such
     a directory exists on the host.
   - sha256 (first 16 hex): `lib/x86_64/libhardware.so` 34875c929a572efd, `libhardware_20_04.so` 2947f638078ad2ba,
     `libsolver.so` 5ca9c0f342918231, `libsolver_20_04.so` 31de329afb3c102a; `lib/aarch64/libhardware.so`
     a388d67559d91950, `libsolver.so` d0bb5dd3d061df4a; `wheels/lib/x86_64/libhardware.so` 68e4bc3cdc5b1343,
     `libsolver.so` f37e1ad9a73e2d15; `wheels/lib/aarch64/libhardware.so` ce6c00c7fdf16edd, `libsolver.so`
     11a54505cf5af70a.
2. Nine binary STL meshes (25.3 MB, identical to upstream) and ten text `.obj` meshes (11.3 MB, added locally for
   the MuJoCo model).
3. Nine NumPy calibration files (§12.4).

**The largest file is not what its name says.** `src/kdl_parser/kdl_parser_py/test/kdl_parser_py.urdf`
(23,714,538 bytes, 300,391 lines, 35 % of the work tree) begins `%!PS-Adobe-3.0`, `%%Creator: (ImageMagick)`,
`%%Title: (kdl_parser_py.urdf)`, `%%CreationDate: (2015-10-12T17:22:19-07:00)`, `%%BoundingBox: -0 -0 2560 1524`:
it is a PostScript image of a 2560×1524 screen, not a robot description. It is byte-identical to the file in
upstream ros/kdl_parser 2.13.0, so it was inherited, not added here. Its picture content was not rendered.

### 12.2 Install and build steps that download and execute

- `uv sync --extra dev` (`README.md:21,49`) installs from the 175 package entries locked in `uv.lock` (all PyPI;
  the entries cover several Python versions and platforms, so one install uses a subset), including
  `torch==2.7.1` with 14 `nvidia-*-cu12` wheels, `lerobot==0.4.4`, `wandb==0.24.2`, `sentry-sdk==2.71.0`,
  `huggingface-hub==0.35.3`, `hf-transfer`, `rerun-sdk==0.26.2`, `robotmq==0.1.15`, and builds the project itself
  as an editable setuptools package (`pyproject.toml:39-46`). Third-party package build/installation code runs as
  usual for a Python install.
- `scripts/setup/packages.sh:1-5` creates a Conda environment from channels `robostack-staging` and
  `conda-forge` (`conda_environments/py310_environment.yaml:2-4`), pip-installs
  `spnav @ https://github.com/cheng-chi/spnav/archive/<commit>.tar.gz` (`:30`), then runs `cmake ..` and
  `make -j` on the vendored SDK. The CMake files call `execute_process` (`cmake/SelectSdkLibDir.cmake:16-21`:
  `getconf GNU_LIBC_VERSION`).
- Vendored wheel tooling, not referenced by the project's README: four Dockerfiles that `git clone` SOEM v1.4.0
  and ros/kdl_parser 1.14.2 and `pip install` inside a manylinux image (`wheels/Dockerfile.x86_64:1,14,19,44-46`);
  `wheels/build_wheels.sh` / `build_wheel_single_ver.sh` run `docker build`; **`wheels/upload_to_pypi.sh:5-6`
  runs `pip install twine` and `twine upload wheels/wheelhouse/*`**, which would publish to PyPI with whatever
  credentials the caller has.
- Vendored `src/kdl_parser/.travis.yml:15-16` contains `wget http://packages.ros.org/ros.key -O - | sudo apt-key add -`
  (a pipe-to-root over plain http). It is an old upstream CI file and nothing in this repository invokes it.
- Training initialises the vision backbone from torchvision's hosted ImageNet weights:
  `getattr(torchvision.models, …)(weights=self.pretrained_backbone_weights)` (`src/policy/act.py:214`,
  `src/policy/policy.py:200`), default `IMAGENET1K_V1` (`src/config/training_args.py:97`). **[DOC CLAIM]**
  `src/config/training/README.md:59-61`: "The weights download once on the training host; inference builds the
  same architecture without downloading and loads everything from the checkpoint."
- CI uses third-party actions pinned by tag, not by commit: `actions/checkout@v4`, `astral-sh/setup-uv@v5`
  (`.github/workflows/tests.yml:15,18`), and `sudo apt-get install -y build-essential python3-dev` (`:17`).
- **Not found:** no `curl … | sh`, no `shell=True`, no `os.system`, no `eval(` in the project's own code. The
  four `exec(compile(…))` calls are in tests and compile functions extracted by AST from the repository's own
  files (`scripts/tests/test_export_scope.py:37`, `test_standard_capture.py:170`,
  `test_standard_image_processing.py:85`, `test_camera_settings_contract.py:182`). `subprocess` call sites use
  argument lists and start `v4l2-ctl`, the Rerun viewer (`src/visualization/visualization_utils/memory_guard.py:56-58`)
  or the repository's own helper scripts (`src/camera_utils/robotmq_camera_cache.py:285`,
  `src/recording/replay_practice.py:110`).

### 12.3 An undeclared dependency on the default recording path

**[VERIFIED]** `src/camera_utils/camera_server.py:772` does `from robologger.loggers.video_logger import VideoLogger`
and, if that fails, raises "Robologger was requested but is not installed. Install the local Robologger package
or start this camera with --no-robologger." (`:774-777`). The option defaults to on (`--use-robologger`,
`default=True`, `:1922-1925`; function default `use_robologger=True`, `:1501`), and
`scripts/cameras/camera_servers_tmux.sh:156,158` starts the server without `--no-robologger`. `robologger` is
**not** in `pyproject.toml`, **not** in `uv.lock`, and not vendored. **[DOC CLAIM]** `docs/recording_timestamps.md:35-36`:
"Task collectors and inference continue to use the same independent per-camera RoboLogger backend."
**[NOT ESTABLISHED]** where that package comes from, which version is expected, and whether video recording
works after the documented `uv sync` alone.

Other imports that the lockfile does not provide by design: `arx5_interface` (built from the vendored SDK, or
`pip install arx5-interface` per the SDK README), `PyKDL` (Conda environment or a source build,
`scripts/setup/paths.sh:23-28`); SDK examples also import `mujoco` and `click`.

### 12.4 Unpickling and other deserialisation of files a user may receive from others

- **Checkpoint statistics are a pickle.** `stats = pickle.load(f)` on `<ckpt_dir>/dataset_stats.pkl`
  (`src/inference/inference_direct.py:847-849`, `src/inference/serve_policy.py:404-406`); written by
  `pickle.dump` (`src/policy/imitate_episodes_train.py:1313-1315`). **[README CLAIM]** `README.md:350-352`: "A
  checkpoint directory holds `all_configs.json`, `dataset_stats.pkl` and the weights; copy the whole directory when
  moving a checkpoint to another machine." Loading a checkpoint directory from an untrusted source therefore runs
  arbitrary code.
- `torch.load(ckpt_path, map_location=…)` without an explicit `weights_only` argument
  (`inference_direct.py:838`, `serve_policy.py:418`, `imitate_episodes_train.py:611,619,622,650`). Whether that is
  restricted to tensors depends on the installed PyTorch's default; the pin is `torch==2.7.1`.
- **Three shipped calibration files are pickled object arrays**:
  `calibration/masking/video_{left,overhead,right}_intri_imgs/*_intrinsics.npy` (`descr='|O'`). Disassembled
  without loading (`tools/npinspect.py`): each references only `numpy._core.multiarray._reconstruct`,
  `numpy.ndarray`, `numpy.dtype` and the keys `camera_matrix`, `dist_coeffs` — benign as shipped. They are read
  with `np.load(path, allow_pickle=True).item()` (`src/utils/masking.py:125`,
  `src/visualization/visualization_utils/rerun_utils.py:531`, `scripts/setup/calibrate_masking.py:36`), and
  start-pose files are also opened with `allow_pickle=True` (`inference_direct.py:512`,
  `src/recording/record_episode_single.py:531`, `scripts/setup/set_up_recording.py:360`) although the shipped
  `start_position.npz` holds only plain arrays.
- The other six `.npy`/`.npz` files hold plain numeric/string arrays: camera controls for `left`, `overhead`,
  `right` at 1280×720, 100 fps with device names `/dev/video_left|overhead|right` (`calibration/camera_params.npz`);
  a 14-value start pose for arms `left`/`right` on `can0`/`can1` with a creation time of 1789894371 (2026-09-20)
  (`calibration/start_position.npz`); 4×4 transforms.
- YAML is loaded safely: `class ConfigLoader(yaml.SafeLoader)` (`src/config/task_config.py:23`) and
  `yaml.safe_load` elsewhere.
- `src/recording/record_hardware.py:86-89` imports an arbitrary `module:factory` named on the command line
  (documented plug-in mechanism, `docs/inference_layout.md:79-85`).

### 12.5 Network exposure

- **Policy server.** `--host` defaults to `"0.0.0.0"`, `--port` to `8000` (`src/inference/serve_policy.py:445-446`).
  `websocket_server.serve(self._handler, self.host, self.port, compression=None, max_size=None,
  process_request=_health_check)` (`:165-171`): no authentication, no TLS, no message-size limit. On connect it
  sends metadata that includes `ckpt_path` and `ckpt_dir` (`:144-145`); on an exception it sends the Python
  traceback to the peer (`:198`). Arrays are rebuilt from peer-supplied `dtype`/`shape` fields
  (`unpack_array`, `serve_policy.py:63-68`, `policy_transport.py:40-45`); the packing side rejects object, void
  and complex dtypes (`policy_transport.py:20-21`) but the unpacking side has no such check (not analysed further).
  Only the client can send a key (`Authorization: Api-Key …`, `policy_transport.py:159`, option `--api_key`,
  `inference_remote.py:489`); the repository's own server never checks one.
- **Camera servers.** Each camera's recording endpoint is created with `endpoint=f"tcp://0.0.0.0:{endpoint_port}"`
  (`src/camera_utils/camera_server.py:786`), base port 55551 (`:1502`, `--robologger-endpoint-base`, `:1934-1936`)
  plus an offset 0–6 per camera (`:703-711`), i.e. TCP 55551–55557 on all interfaces. Commands arriving there are
  decoded with `robotmq.deserialize` and acted on (`:990-998`). **[NOT ESTABLISHED]** the wire format of
  `robotmq.serialize/deserialize` (external package); if it is pickle-based this endpoint would let any host on
  the network run code. The recorder side connects to `tcp://127.0.0.1:…` (`src/recording/distributed_recording.py:146`),
  and the camera cache uses a local `ipc://` endpoint (`src/camera_utils/robotmq_camera_cache.py:276`).
- The remote client connects to `ws://{host}` (unencrypted WebSocket, `policy_transport.py:147`).

### 12.6 Telemetry and outside services

- **Weights & Biases is switched on in every shipped training recipe**: `wandb: true` at
  `src/config/training/single_arm_act.yaml:26`, `bimanual_act.yaml:26`, `single_arm_flow.yaml:27`,
  `bimanual_flow.yaml:27`. With it on, training calls `wandb.init(project="flash-aloha", name=expr_name)` and
  `wandb.config.update(config)` (`src/policy/imitate_episodes_train.py:1164-1166`) and logs metrics
  (`:689-690,738,758`). The command-line default is off (`--wandb`, `default=False`,
  `src/config/training_args.py:115`), but the documented `python scripts/run.py train single_arm_act` uses the
  recipe. `wandb` is imported unconditionally at module import (`:39`). The lockfile also carries `sentry-sdk`.
- No other reporting code: the project's own modules import no HTTP client (`requests`, `urllib`, `httpx`, …);
  `socket` is used only for `gethostname`/`gethostbyname` in the server banner (`serve_policy.py:152-153`).
- **[NOT ESTABLISHED]** what the third-party packages send on their own (Rerun viewer, Hugging Face libraries
  pulled in by `lerobot`, `wandb`'s own diagnostics).

### 12.7 Credentials

**[VERIFIED]** A pattern scan (`tools/riskscan.py --category credentials`) found no token, password, private key,
`.env` or `.netrc`. The 15 matching lines are the optional `--api_key` option described above, the word "token"
in model code, and the vendored `twine upload` script. There are no personal home-directory paths in text files.
E-mail addresses occur only in vendored package metadata (`src/hardware/arx5-sdk/wheels/pyproject.toml:13`,
`src/kdl_parser/*/package.xml`); they are not reproduced here.

### 12.8 Privileged operations and changes to the host

- `sudo` in project scripts: `scripts/setup/activate_arm:11-13`, `scripts/setup/arx_arms.sh:1-2`.
- The docs ask the operator to install udev rules, add a group and reload udev
  (`scripts/cameras/README.md:51-58`), to bring CAN interfaces up with `sudo ip link …`
  (`README.md:88-93`), and — in the vendored SDK README — to give the Python interpreter raw-network
  capabilities with `sudo setcap "cap_net_admin,cap_net_raw=eip" …/python3.10`
  (`src/hardware/arx5-sdk/README.md:293`) and to create a systemd unit (`:344-368`).
- Each camera server opens `/dev/cpu_dma_latency` to hold a 20 µs CPU wake-up limit
  (`src/camera_utils/cpu_latency.py:18-19,34`).
- Camera controls are changed with `v4l2-ctl` (`scripts/cameras/configure_camera_controls.sh:2-4`,
  `src/camera_utils/capture.py:637-645`, `scripts/setup/set_up_recording.py:493-499`).
- Shared memory segments `/dev/shm/flash_aloha_cam_<name>` with a `.clock.json` side file
  (`src/camera_utils/camera_server.py:699,1440-1455`, `camera_timing.py:124`) and
  `/dev/shm/flash_aloha_overhead.npy` (`capture.py:446`).
- `camera_servers_tmux.sh:302-304` does `rm -rf` on `${TMPDIR:-/tmp}/flash_aloha_camera_servers.<session>` and
  creates a tmux session named `camera_servers` (`:15`).
- The tuners write into tracked files: "`s` writes the link masses, or the friction terms, into the selected
  model's URDF in the SDK" (`docs/arm_bringup.md:133-134`).
- Caches under `~/.cache/flash-aloha/` (`docs/raw_training.md:97`, `src/visualization/README.md:43`).
- Python garbage collection is disabled for the life of any process that captures (`AGENTS.md:34-44`,
  `src/contracts/capture_gc.py`).

**Resources this code would claim if started on a machine** (relevant because this machine runs a real robot):
CAN interfaces `can0` / `can1`; video devices `/dev/video_{left,overhead,right,torso,test,extra,extra_overhead}`
(`camera_server.py:682-690`); TCP 8000 and 55551–55557 on all interfaces; `/dev/shm/flash_aloha_*`; tmux session
`camera_servers`; `/dev/cpu_dma_latency`; `data/`, `ckpts/`, `logs/` in the repository root.

### 12.9 Robot-motion hazards and limits the docs themselves state

- Most entry points drive motors; defaults are `can0` (left) and `can1` (right) (`README.md:146`).
- **[DOC CLAIM]** "The watchdog is host software, not a hardware emergency stop." (`docs/inference_layout.md:45-46`);
  "**Ctrl+C while damped disables the arm in place.**" (`docs/safe_exit.md:35`); "SIGKILL, a native crash, or power
  loss" cannot damp (`docs/safe_exit.md:42-43`); "The replay code is validated with fake controllers and clocks;
  that does not constitute a physical robot validation run." (`docs/recording_review_replay.md:109-110`).
- **[DOC CLAIM, vendored SDK]** "the default values are insufficient to guarantee safety when the robot is close to
  singularity or the input control signal includes a lot of noise" (`src/hardware/arx5-sdk/README.md:8`).
- The vendored SDK differs from upstream in motor-level behaviour (velocity scaling, gripper calibration
  constants, friction feed-forward; §6), tuned for the authors' arms.
- **[DOC CLAIM]** `calibration/README.md:4-5`: "**None of it is valid for another robot, camera or mounting:
  recalibrate before use.**"
- Side effects of the test suite, should anyone run it: loopback WebSocket servers
  (`scripts/tests/test_policy_transport.py:40`), subprocesses including `bash` on a rewritten copy of the tmux
  script (`test_camera_script_layout.py:43-54`) and a RobotMQ publisher (`test_robotmq_camera_cache.py:55`), and
  one case that places its image ring in `/dev/shm` (`test_robotmq_camera_cache.py:185`).

### 12.10 Smaller oddities

- The repository is two days old, has 5 commits and no tags or releases; the first commit is a one-shot export.
- Several docs describe components that are **not** in the checkout: "The box-data exporter is not present in
  this checkout." (`docs/recording_rate_metadata.md:34`); OpenPI training/serving hooks
  (`docs/recording_rate_metadata.md:63-78`, `AGENTS.md:55-66`); "specialized route recipes"
  (`docs/raw_training.md:8`); removed scripts `record_episode_robologger.py`, `zip_robologger_episode.py`
  (`docs/recording_timestamps.md:35,41`); a dataset path `record_data/red_and_blue_v2` (`:96`).
- `calibration/masking/masking.yaml:10` points forward kinematics at `…/models/X5.urdf` while both example tasks
  declare `model: L5`. Whether that is intended was not established.
- Leftovers: `DEPLOY = True #if subprocess.getoutput('whoami')=="unitree" else False`
  (`src/inference/inference_direct.py:81`); `import IPython` / `e = IPython.embed` (`src/policy/policy.py:10-11`).
- `src/hardware/arx5-sdk/env.sh:14` hard-codes `$HOME/miniconda3/envs/arx-py310/lib`
  (noted by the authors, `CHANGES_FROM_UPSTREAM.md:21`).
- The README's title and project-page line were added by a different author through the GitHub web UI (commit 4).

---

## 13. Not established

- Whether anything runs: no test, script, build or import was executed.
- Whether the prebuilt `.so` files are safe (only their identity with upstream was shown).
- The origin and version of `robologger`; the wire format of `robotmq`.
- The content of the paper (`flash-aloha.pdf` on the project site), of the hardware Google Doc and of the IEEE
  page: none was fetched in this step.
- Any dataset or trained weights: none is linked from the README, the docs or the project page, and GitHub has
  no release assets.
- The performance and success figures stated on the project page and in the docs.
- Which repository serves the project page; the relationship of the commit author `FarSky` to the project.
- Whether the shipped calibration, gains and URDF edits suit any arm other than the authors'.
- The picture inside the 23.7 MB PostScript file.

---

## 14. How this was produced

Read-only tools written for this step, in `~/research/flash-aloha/tools/`, each run as
`nice -n 19 python3 -I <tool> ~/research/flash-aloha/repo …` (Python 3.13.5):

| Tool | Use |
| --- | --- |
| `inventory.py` | tree, counts by extension, largest files, LFS pointers, binaries, executable bits |
| `links.py` | outside links in docs and in code |
| `mdlinks.py` | relative Markdown links and anchors |
| `riskscan.py` | pattern scan (download, exec, deserialise, credentials, telemetry, network, privileged, delete, machine-specific) |
| `imports.py` | third-party imports and `__main__` guards via `ast` (no import) |
| `npinspect.py` | `.npy` / `.npz` headers and pickle disassembly (no unpickling) |
| `elfinfo.py` | ELF headers, NEEDED, RUNPATH, symbols (pure Python) |
| `upstream_compare.py` | blob-hash comparison with saved upstream tree listings; repository metadata |
| `htmlinfo.py` | links and text of the saved project page |

Git was used only for read commands (`status`, `fsck`, `log`, `show`, `ls-files`, `rev-parse`, `for-each-ref`,
`shortlog`, `count-objects`, `diff --stat`).

Outside requests (all public, unauthenticated, no local detail sent), 2026-11-14 03:51–03:57 UTC; answers saved
under `~/research/flash-aloha/web/download-step/`:

- `api.github.com/repos/JohnYao93/flash-aloha` and its `/releases`, `/branches`, `/tags`
- `api.github.com/repos/real-stanford/arx5-sdk/git/trees/79240efc…?recursive=1`
- `api.github.com/repos/ros/kdl_parser/git/trees/2.13.0?recursive=1`
- `raw.githubusercontent.com/ros/kdl_parser/2.13.0/{README.md,kdl_parser_py/kdl_parser_py/urdf.py}`
- `johnyao93.github.io/flash-aloha/` (the project page, once)

All 34 Markdown files in the repository were read in full (root `README.md`, `AGENTS.md`, 13 files in `docs/`,
2 in `calibration/`, 2 in `scripts/`, 12 own package READMEs in `src/`, and the 3 vendored ones), together with
`LICENSE` ×3, `pyproject.toml`, `.gitignore`, `.python-version`, the CI workflow, all shell scripts, all 12
YAML configs, `masking.yaml`, the launcher scripts, the vendored build files, and targeted parts of the code
cited above.
