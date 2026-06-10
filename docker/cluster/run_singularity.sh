#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current uwlab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the UW Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# make sure that all directories exists in cache directory
setup_directories
# copy all cache files
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# make sure logs directory exists (in the permanent uwlab directory)
mkdir -p "$CLUSTER_UWLAB_DIR/logs"
touch "$CLUSTER_UWLAB_DIR/logs/.keep"

# copy the temporary uwlab directory with the latest changes to the compute node
cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$1")

# copy container to the compute node
tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR

# Forward selected host env vars into the apptainer container.
# Apptainer/Singularity only passes env vars that start with APPTAINERENV_ /
# SINGULARITYENV_ (the prefix is stripped inside).
# WANDB_USERNAME is what rsl_rl/utils/wandb_utils.py actually reads for the
# entity (despite being non-standard); WANDB_ENTITY is set too as a future-
# proofing courtesy. WANDB_PROJECT is intentionally NOT forwarded -- rsl_rl
# reads project from cfg["wandb_project"] and ignores the env var; pass it
# via --log_project_name on the training command line instead.
for var in WANDB_API_KEY WANDB_USERNAME WANDB_ENTITY WANDB_MODE WANDB_RUN_GROUP; do
    if [[ -n "${!var}" ]]; then
        export "APPTAINERENV_${var}=${!var}"
    fi
done

# Build optional asset-mirror bind mounts and matching env-var overrides for
# uwlab_assets so USDs / split datasets load from local disk instead of HF.
# See source/uwlab_assets/uwlab_assets/__init__.py: the constants honor env
# overrides, and resolve_cloud_path() short-circuits when given a local path.
ASSET_BINDS=""
if [[ -n "${CLUSTER_UWLAB_ASSETS_DIR}" && -d "${CLUSTER_UWLAB_ASSETS_DIR}" ]]; then
    ASSET_BINDS+=" -B ${CLUSTER_UWLAB_ASSETS_DIR}:/workspace/uwlab-assets:ro"
    export APPTAINERENV_UWLAB_CLOUD_ASSETS_DIR=/workspace/uwlab-assets
fi
if [[ -n "${CLUSTER_SPLIT_ASSETS_DIR}" && -d "${CLUSTER_SPLIT_ASSETS_DIR}" ]]; then
    ASSET_BINDS+=" -B ${CLUSTER_SPLIT_ASSETS_DIR}:/workspace/datasets-split:ro"
    export APPTAINERENV_UWLAB_SPLIT_ASSETS_DIR=/workspace/datasets-split
fi

# Outputs (hydra cwd) and logs need to live on a writable bind, not in the
# read-only SIF layer. Place them under the persistent cache dir.
mkdir -p "${CLUSTER_ISAAC_SIM_CACHE_DIR}/outputs" "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs-uwlab" "${CLUSTER_ISAAC_SIM_CACHE_DIR}/kit-data"

# execute command in singularity container
# NOTE: We bind only source/, scripts/, scripts_v2/ from the synced code copy
# rather than the whole repo. This preserves the baked-in _isaaclab/ symlink
# inside the SIF (which isn't present in the repo on the cluster).
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B ${CLUSTER_ISAAC_SIM_CACHE_DIR}/kit-data:${DOCKER_ISAACSIM_ROOT_PATH}/kit/data:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B ${CLUSTER_ISAAC_SIM_CACHE_DIR}/outputs:/workspace/outputs:rw \
    -B ${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs-uwlab:/workspace/logs:rw \
    -B $TMPDIR/$dir_name/source:/workspace/uwlab/source:rw \
    -B $TMPDIR/$dir_name/scripts:/workspace/uwlab/scripts:rw \
    -B $TMPDIR/$dir_name/scripts_v2:/workspace/uwlab/scripts_v2:rw \
    -B $CLUSTER_UWLAB_DIR/logs:/workspace/uwlab/logs:rw \
    ${ASSET_BINDS} \
    --nv --writable-tmpfs --containall $TMPDIR/$2.sif \
    bash -c "export UWLAB_PATH=/workspace/uwlab && cd /workspace/outputs && /isaac-sim/python.sh /workspace/uwlab/${CLUSTER_PYTHON_EXECUTABLE} ${@:3}"

# copy resulting cache files back to host
rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

# if defined, remove the temporary uwlab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

echo "(run_singularity.py): Return"
