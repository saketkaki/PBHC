# Running PBHC on an RTX 5090

How we got PBHC (`INSTALL.md`) working on an NVIDIA RTX 5090, what broke, and how each problem was fixed.

**Machine:** Ubuntu 22.04, RTX 5090 (Blackwell, compute capability `sm_120`), driver 590.48 (CUDA 13.1), 24 CPU cores, 125 GB RAM. CUDA toolkits 12.9 and 13.0 installed under `/usr/local`.

**Result:** everything runs inside one conda env, `pbhc`: Isaac Gym GPU physics, training, and the evaluation scripts (`eval_agent.py`, `sample_eps.py`, `ratio_eps.py`).

## TL;DR

```bash
conda activate pbhc
cd ~/ws/PBHC
python humanoidverse/train_agent.py ...      # commands from humanoidverse/README.md work as-is
```

The one big difference from `INSTALL.md`: **PyTorch 2.4.1 was built from source** for the 5090. No prebuilt PyTorch for Python 3.8 supports this GPU.

> **Never run `pip install torch` (or `torchvision`) inside `pbhc`.** pip would replace our build with a stock wheel that cannot run on the 5090. To repair, reinstall our wheel (Section 6).

## 1. The core problem: Python 3.8 vs. the RTX 5090

| Constraint | Detail |
|---|---|
| Isaac Gym Preview 4 needs **Python ≤ 3.8** | It ships compiled bindings only for 3.6/3.7/3.8 (`gym_36.so`, `gym_37.so`, `gym_38.so`). It is NVIDIA's final, deprecated release; there is no newer Isaac Gym. |
| The RTX 5090 needs **CUDA ≥ 12.8 kernels for `sm_120`** | PyTorch has `sm_120` kernels only from 2.7 (CUDA 12.8 builds) onward. |
| PyTorch dropped Python 3.8 after **2.4.1** | The newest torch for Python 3.8 is `2.4.1+cu124`, which supports up to `sm_90`. |

We tested the stock wheel; it fails on the first GPU operation:

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Isaac Gym's own GPU physics (PhysX) **does** run on the 5090; only PyTorch was the blocker. We confirmed this with a headless 64-ball drop test (balls settled at z ≈ 0.102 m, radius 0.1 m) before building PyTorch.

**Solution:** build PyTorch 2.4.1 from source with CUDA 12.9, targeting `sm_120`.

**Alternatives we considered and rejected:**
- **Isaac Sim / Isaac Lab instead of Isaac Gym.** PBHC has an `isaacsim` backend, but it targets Isaac Lab 1.x (`omni.isaac.lab`), whose bundled PyTorch also lacks `sm_120`. It is also missing features PBHC's motion tracking needs (domain-randomization hooks, external pushes, terrain creation), and some PBHC code paths assert `simulator == isaacgym`. Porting would take days.
- **Another paper (e.g. HoST).** HoST also uses Isaac Gym and pins an even older torch (1.10+cu113), so it has the same problem.

## 2. Step-by-step: what we did

### 2.1 Create the conda env (as in `INSTALL.md`)

```bash
conda create -n pbhc python=3.8      # Python 3.8.20
```

Everything below is installed into this env. System Python and its torch were not touched.

### 2.2 Download and install Isaac Gym (as in `INSTALL.md`)

```bash
cd ~/ws/PBHC
wget https://developer.nvidia.com/isaac-gym-preview-4 -O isaac-gym-preview-4   # redirects to a signed tarball, no login needed
tar -xzf isaac-gym-preview-4
pip install -e isaacgym/python
```

`isaacgym/` and the tarball are already in `.gitignore`.

### 2.3 Problem: `ImportError: libpython3.8.so.1.0: cannot open shared object file`

A well-known Isaac Gym issue: its binding needs the conda env's `libpython3.8.so` on the library path.

**Fix:** a conda activation hook, so the env sets this up itself and undoes it on deactivate.

`~/miniconda3/envs/pbhc/etc/conda/activate.d/isaacgym_ld.sh` (final version, including later additions for torch):

```bash
export _PBHC_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
export _PBHC_OLD_CUDA_HOME="${CUDA_HOME}"
# Isaac Gym needs libpython3.8; source-built torch (sm_120) needs CUDA 12.9 runtime + pip cuDNN
export CUDA_HOME=/usr/local/cuda-12.9
_PBHC_CUDNN="${CONDA_PREFIX}/lib/python3.8/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${_PBHC_CUDNN}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _PBHC_CUDNN
```

`~/miniconda3/envs/pbhc/etc/conda/deactivate.d/isaacgym_ld.sh`:

```bash
export LD_LIBRARY_PATH="${_PBHC_OLD_LD_LIBRARY_PATH}"
export CUDA_HOME="${_PBHC_OLD_CUDA_HOME}"
unset _PBHC_OLD_LD_LIBRARY_PATH _PBHC_OLD_CUDA_HOME
[ -z "$LD_LIBRARY_PATH" ] && unset LD_LIBRARY_PATH
[ -z "$CUDA_HOME" ] && unset CUDA_HOME
```

### 2.4 Remove the stock torch

`pip install -e isaacgym/python` pulls in stock `torch 2.4.1+cu124`, `torchvision`, `triton` and the CUDA 12.4 `nvidia-*` pip packages. None of these work on the 5090, so we removed them (keeping cuDNN, see below) and freed about 4 GB:

```bash
pip uninstall -y torch torchvision triton $(pip list | awk '/^nvidia-/{print $1}' | grep -v cudnn)
```

### 2.5 Install the build dependencies (into the env)

```bash
pip install "cmake<4" ninja pyyaml typing_extensions "setuptools<70" wheel \
            mkl-static==2024.2.0 mkl-include==2024.2.0 \
            "nvidia-cudnn-cu12>=9.8,<9.11"      # got cuDNN 9.10.2
```

- `cmake<4`: CMake 4 rejects the old `cmake_minimum_required` values in torch 2.4's dependencies.
- `setuptools<70`: torch 2.4's `setup.py` predates newer setuptools changes.
- MKL: a fast CPU BLAS, statically linked into torch.
- cuDNN 9.8+: earlier 9.x releases lack Blackwell support.

### 2.6 Get the PyTorch source

```bash
git clone --depth 1 --branch v2.4.1 --recursive --shallow-submodules -j8 \
    https://github.com/pytorch/pytorch.git      # ~1.9 GB
```

### 2.7 Problem: CUDA 12.9 no longer ships `libnvToolsExt`

torch 2.4's CMake hard-requires the `nvToolsExt` library (`FATAL_ERROR "Failed to find nvToolsExt"`). CUDA 12.9 provides only the header-only **NVTX3**. Newer PyTorch versions switched to NVTX3; we back-ported that change:

- `cmake/public/cuda.cmake`: remove the fatal check, and make `torch::nvtoolsext` link nothing (`${CMAKE_DL_LIBS}`) instead of `CUDA::nvToolsExt`.
- `torch/csrc/cuda/shared/nvtx.cpp` and `torch/csrc/profiler/stubs/cuda.cpp`: `#include <nvToolsExt.h>` → `#include <nvtx3/nvToolsExt.h>`.

### 2.8 Problem: torch 2.4 can't parse architecture `12.0`

The first build attempt failed at CMake configure:

```
CMake Error at cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake:227 (message):
  Unknown CUDA Architecture Name 12.0 in CUDA_SELECT_NVCC_ARCH_FLAGS
```

The regex only accepted single-digit major versions (`9.0`). We allowed two digits (`[0-9]` → `[0-9]+`), as newer PyTorch does. After this, configure printed `-gencode;arch=compute_120,code=sm_120`.

**The complete patch** (saved as `~/miniconda3/envs/pbhc/share/pbhc-wheels/torch-2.4.1-sm120.patch`):

```diff
--- a/cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
+++ b/cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
@@ -186,7 +186,7 @@ function(CUDA_SELECT_NVCC_ARCH_FLAGS out_variable)
-    if(arch_name MATCHES "^([0-9]\\.[0-9]a?(\\([0-9]\\.[0-9]\\))?)$")
+    if(arch_name MATCHES "^([0-9]+\\.[0-9]a?(\\([0-9]+\\.[0-9]\\))?)$")
--- a/cmake/public/cuda.cmake
+++ b/cmake/public/cuda.cmake
@@ -66,9 +66,7 @@
-if(NOT TARGET CUDA::nvToolsExt)
-  message(FATAL_ERROR "Failed to find nvToolsExt")
-endif()
+# PBHC local patch: CUDA >= 12.9 ships NVTX3 header-only (no libnvToolsExt)
@@ -186,7 +184,7 @@
 add_library(torch::nvtoolsext INTERFACE IMPORTED)
 set_property(
     TARGET torch::nvtoolsext PROPERTY INTERFACE_LINK_LIBRARIES
-    CUDA::nvToolsExt)
+    ${CMAKE_DL_LIBS})
--- a/torch/csrc/cuda/shared/nvtx.cpp
+++ b/torch/csrc/cuda/shared/nvtx.cpp
-#include <nvToolsExt.h>
+#include <nvtx3/nvToolsExt.h>
--- a/torch/csrc/profiler/stubs/cuda.cpp
+++ b/torch/csrc/profiler/stubs/cuda.cpp
-#include <nvToolsExt.h>
+#include <nvtx3/nvToolsExt.h>
```

Apply it to a fresh checkout with `git apply torch-2.4.1-sm120.patch`.

### 2.9 Build the wheel

```bash
conda activate pbhc
cd pytorch
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH \
  TORCH_CUDA_ARCH_LIST="12.0" USE_CUDA=1 USE_CUDNN=1 \
  CUDNN_INCLUDE_DIR=$SP/nvidia/cudnn/include CUDNN_LIBRARY=$SP/nvidia/cudnn/lib \
  USE_NCCL=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_CUSPARSELT=0 USE_FBGEMM=0 \
  BUILD_TEST=0 USE_ROCM=0 USE_XPU=0 USE_MPI=0 USE_DISTRIBUTED=1 \
  CMAKE_PREFIX_PATH=$CONDA_PREFIX MAX_JOBS=20 PYTORCH_BUILD_VERSION=2.4.1 PYTORCH_BUILD_NUMBER=1
python setup.py bdist_wheel        # -> dist/torch-2.4.1-cp38-cp38-linux_x86_64.whl
```

**Why these flags:** PBHC only uses core torch (`nn`, `optim`, `distributions`, `utils.tensorboard`); it uses no torchvision, multi-GPU or fused attention. So we compiled only for `sm_120` and turned off NCCL, flash/memory-efficient attention, cuSPARSELt, FBGEMM (quantization) and tests. This cut the build to roughly 8,100 steps, about **25 minutes** on 24 cores, using ~6 GB of disk at its peak.

### 2.10 Problem: cuDNN was silently disabled

The first wheel worked on the GPU but reported `torch.backends.cudnn.version() == None`. The build log said:

```
-- Could NOT find CUDNN (missing: CUDNN_LIBRARY_PATH)
   Cannot find cuDNN library.  Turning the option off
```

torch's `FindCUDNN.cmake` searches the `CUDNN_LIBRARY` directory for a file named exactly `libcudnn.so`, but the pip cuDNN package only ships `libcudnn.so.9`.

**Fix:** add the missing symlink inside the env, re-enable cuDNN in the existing build directory, and rebuild incrementally (only 531 steps, ~10 min):

```bash
ln -sf libcudnn.so.9 $SP/nvidia/cudnn/lib/libcudnn.so
cd build && cmake -DUSE_CUDNN=ON -DCUDNN_LIBRARY=$SP/nvidia/cudnn/lib \
                  -DCUDNN_INCLUDE_DIR=$SP/nvidia/cudnn/include \
                  -UCUDNN_LIBRARY_PATH -UCUDNN_INCLUDE_PATH . && cd ..
python setup.py bdist_wheel        # same exports as 2.9
```

Note that 2.9 above already shows `CUDNN_LIBRARY` as the *directory*. Create the symlink before the first build and cuDNN is found immediately.

PBHC does use convolutions (`Conv1d` in `humanoidverse/agents/modules/encoder_modules.py`). Without cuDNN they still work, just more slowly.

### 2.11 Install the wheel and keep a copy

```bash
mkdir -p $CONDA_PREFIX/share/pbhc-wheels
cp dist/torch-2.4.1-cp38-cp38-linux_x86_64.whl $CONDA_PREFIX/share/pbhc-wheels/
pip install --force-reinstall --no-deps $CONDA_PREFIX/share/pbhc-wheels/torch-2.4.1-cp38-cp38-linux_x86_64.whl
```

Saved in `~/miniconda3/envs/pbhc/share/pbhc-wheels/`: the wheel (~220 MB), the patch, and both build logs. Afterwards we deleted the ~8 GB source/build folder.

### 2.12 Install PBHC (as in `INSTALL.md`)

```bash
cd ~/ws/PBHC
pip install -e .
pip install -e humanoidverse/isaac_utils
pip install rerun-sdk==0.22.0 trimesh        # optional visualization deps
```

No problems here; `setup.py` does not list torch, so it doesn't touch our build.

## 3. Verification

| Test | Result |
|---|---|
| `torch.__version__`, CUDA, cuDNN, arch list | `2.4.1`, CUDA `12.9`, cuDNN `91002`, `['sm_120']`, RTX 5090 |
| GPU ops: matmul, `Conv1d` fwd/bwd (cuDNN), MLP backward + Adam, `torch.distributions`, tensorboard | All pass |
| Isaac Gym GPU PhysX, CPU pipeline | Pass |
| Isaac Gym GPU pipeline + `gymtorch` (JIT-compiled against our torch) | Pass: sim state read directly into CUDA tensors |
| `isaacgym/python/examples/1080_balls_of_solitude.py` (from `INSTALL.md`) | Ran with viewer for 20 s without errors (it loops until the window is closed) |
| PBHC training (README command, G1, Horse-stance, 128 envs, headless, 3 min) | Reached iteration 100+, mean reward −3.0 → −2.6 |
| `eval_agent.py` on `example/pretrained_horse_stance_pose/model_50000.pt` | Loads, exports ONNX, runs viewer |
| ONNX re-exported with our torch vs. the ONNX shipped in the repo | Identical outputs (max abs diff **0**) on 64 random inputs |
| `sample_eps.py` (1 env, 1 episode) | E_gmpbpe **29.5 mm**, E_mpbpe **19.0 mm**, E_mpjpe **501 ×10⁻³ rad** |
| `ratio_eps.py` (100 envs, 100 episodes) | Episode-length ratio **98.4%** (343 / 349 frames) |

For reference, the paper reports 53 ± 18 mm E_gmpbpe for its "easy" motions, which include Horse-stance pose.

## 4. Final package versions (key ones)

| Package | Version | Source |
|---|---|---|
| Python | 3.8.20 | conda |
| torch | 2.4.1 (CUDA 12.9, cuDNN 9.10, `sm_120` only) | **built from source** |
| nvidia-cudnn-cu12 | 9.10.2.21 | pip |
| isaacgym | 1.0rc4 (Preview 4) | NVIDIA tarball, editable install |
| pbhc, isaac_utils | 0.0.1 | editable install |
| numpy | 1.24.4 | pinned by PBHC |
| mujoco | 3.2.3 | pinned by PBHC |
| hydra-core | 1.3.8 | pip |
| onnx / onnxruntime | 1.17.0 / 1.19.2 | pip |
| rerun-sdk / trimesh | 0.22.0 / 4.12.2 | pip (optional) |
| cmake / ninja / mkl-static | 3.31.10 / 1.13.2 / 2024.2.0 | pip (build-time only) |
| torchvision | **not installed** | not used by PBHC |

## 5. Known warnings (all harmless)

| Message | Why it's fine |
|---|---|
| `pip check`: `isaacgym requires torchvision, which is not installed` | PBHC never imports torchvision. Do **not** pip-install it (it would pull in a stock torch). |
| `pip check`: `nvidia-cudnn-cu12 requires nvidia-cublas-cu12` | cuBLAS comes from the system CUDA 12.9 via `LD_LIBRARY_PATH`. |
| `MESA: warning: Driver does not support the 0x7d67 PCI ID` | Mesa's Vulkan driver doesn't recognise the Intel integrated GPU while listing devices; rendering uses the NVIDIA GPU. |
| `WARNING: Forcing CPU pipeline.` (Isaac Gym examples) | Normal for those examples. |
| `Hydra14MigrationWarning`, resolver `'eval' is already registered` | Hydra version messages from PBHC's config code. |

## 6. Gotchas and maintenance

- **Activate the env first** (`conda activate pbhc`). The activation hook sets `LD_LIBRARY_PATH` and `CUDA_HOME`; without it Isaac Gym and torch fail to load their libraries.
- **The env depends on `/usr/local/cuda-12.9`** (runtime libraries and `nvcc` for `gymtorch`'s JIT build). Don't uninstall that toolkit.
- **If torch gets overwritten**, reinstall our build:
  ```bash
  pip install --force-reinstall --no-deps ~/miniconda3/envs/pbhc/share/pbhc-wheels/torch-2.4.1-cp38-cp38-linux_x86_64.whl
  ```
- **The wheel works only on `sm_120` GPUs (RTX 50-series).** Group members on other GPUs (e.g. RTX 4090 / 30-series) should follow `INSTALL.md` unmodified and keep the stock `torch 2.4.1` from pip.
- **`eval_agent.py` overwrites `example/<ckpt>/exported/*.onnx`** each run, so git will show it as modified. `git checkout -- example/` restores it.
- **Outputs:** training runs write to `logs/`, evaluation to `logs_eval/` (gitignored), and `sample_eps.py`/`ratio_eps.py` write metrics JSON to `example/<ckpt>/metrics/`.
- **`gymtorch` JIT cache:** `~/.cache/torch_extensions/py38_cu129/gymtorch`. Delete it if you ever rebuild or replace torch; it recompiles automatically on next import.
- **Disk:** the root disk was ~98% full during setup. The source build needs ~8 GB free temporarily.

## 7. Reproducing on another RTX 50-series machine

The fastest route is to copy the saved wheel and skip the build:

1. Install CUDA 12.9 (≥ 12.8) to `/usr/local/cuda-12.9`.
2. Section 2.1–2.3: create the env, install Isaac Gym, add the activation hooks.
3. `pip install "nvidia-cudnn-cu12>=9.8,<9.11"` and remove the stock torch (2.4).
4. Install the saved wheel (2.11), then PBHC (2.12).
5. Run the checks in Section 3.

To rebuild from scratch instead, follow 2.5–2.11 and apply the saved patch to a fresh `v2.4.1` checkout.
