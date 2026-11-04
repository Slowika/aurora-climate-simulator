# Aurora-CS

Aurora-CS is a CO2-conditioned climate emulator built on
[Aurora](https://github.com/microsoft/aurora).
The aim is to use Aurora for long-term climate simulations, while taking into account the variables
that matter for long-term climate prediction.
CO2 is supported today, and others, such as sea surface temperature (SST), are planned.
The pretrained Aurora weights stay frozen.
A small forcing encoder and per-block, zero-initialised AdaLN-Zero adapters learn to condition the
forecast on atmospheric CO2.
The result is a 6-hourly autoregressive rollout under a chosen CO2 trajectory.

This repo is the canonical implementation of the AuroraCS model (the `aurora_cs` package).
It also holds the associated scripts, run as Azure ML jobs: data fetching as command jobs,
and training and rollout as pipelines.

## Repository structure

```
aurora-climate-simulator/
├── aurora_cs/                      The `aurora_cs` model package
├── components/                     Azure ML component definitions
|   ├── evaluate_rollout/           Rollout prediction evaluation
|   ├── prepare_forcing_trajectory/ Rollout CO2 forcing trajectory preparation
|   ├── prepare_initial_condition/  Rollout initial condition preparation
|   ├── rollout_aurora_cs/          Autoregressive rollout execution
|   └── train_aurora_cs/            Aurora CS training execution
├── data/                           Data production scripts and Azure ML helpers
├── environments/                   Azure ML Docker/Conda environment definitions
|   ├── data/                       Data fetching environment
|   └── torch/                      Training and rollout environment
├── jobs/                           Azure ML job YAML definitions to submit (placeholders to fill in)
└── tests/                          Unit tests (`pytest`)
```

This structure is likely to change, and the current organisation is not final.

## Usage

### Local

Install [`uv`](https://docs.astral.sh/uv/) and run the below commands:

```bash
 uv venv
 uv sync
```

Install the [`az` CLI and `ml` extension](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public).

### Prerequisites

- An Azure ML workspace with compute.
  A GPU is strongly recommended, since training and rollout are impractically slow on CPU.
  The data jobs and the rollout preparation steps are fine on CPU.
- The compute's managed identity needs Storage Blob Data Contributor on the workspace storage
  account, and a role that lets it register data assets in the workspace.
- A restricted network may block outbound access.
  Allow the hosts the jobs reach: `storage.googleapis.com` (ERA5), the ESGF servers
  `esgf-data.dkrz.de` and `esgf.ceda.ac.uk` (CO2), `huggingface.co` and `*.hf.co` (the pretrained
  checkpoint, downloaded at training time), plus PyPI, conda and `github.com` for environment
  builds.
- Register the two environments once:
  ```bash
  az ml environment create -f environments/data/environment.yml
  az ml environment create -f environments/torch/environment.yml
  ```

Each job YAML in `jobs/` has `<PLACEHOLDERS>` for values you can supply at submit time with `--set`.
Set up your shell once (the commands below use these variables):

```bash
az configure --defaults group=<resource-group> workspace=<workspace>
CPU_COMPUTE=<cpu-cluster>
GPU_COMPUTE=<gpu-cluster>
# Workspace and identity inputs used by the data jobs:
WS="inputs.subscription_id=<subscription-id> inputs.resource_group=<resource-group> \
inputs.workspace_name=<workspace> inputs.managed_identity_client_id=<identity-client-id>"
```

Job names must be unique, so give every submission its own `--set name=...`.
The examples encode the key parameters in the name; change them when you change the parameters.

### 1. Fetch data (once)

`jobs/fetch_era5_job.yml` fetches ERA5 initial conditions, and `jobs/fetch_co2_forcing_job.yml`
fetches the CO2 series for an SSP scenario (`ssp126`, `ssp245`, `ssp370` or `ssp585`; coverage
starts 2015-01).
Both register an Azure ML data asset.
Run each once for a given date range, unless you are adding a disjoint range.
A month interrupted part-way (for example by a failed job) is detected and fetched again on the
next run, because a partly written shard still has its metadata but reads back as NaN where
chunks are missing. Training also checks the shards in its window before starting and fails
with the list of damaged months.

```bash
az ml job create -f jobs/fetch_era5_job.yml --set name=fetch-era5-2020-01 \
  --set compute=azureml:$CPU_COMPUTE --set $WS \
  --set inputs.start_date=2020-01-01 inputs.end_date=2020-01-31 \
  --set inputs.prefix=era5 inputs.asset_name=era5-weather

az ml job create -f jobs/fetch_co2_forcing_job.yml --set name=fetch-co2-ssp245-2020-01 \
  --set compute=azureml:$CPU_COMPUTE --set $WS \
  --set inputs.start_date=2020-01-01 inputs.end_date=2020-02-01 \
  --set inputs.scenario=ssp245 inputs.output_path=co2.csv inputs.asset_name=co2-forcing
```

### 2. Train

`jobs/train_aurora_cs_job.yml` downloads the pretrained Aurora checkpoint, trains the conditioning
network on one-step prediction over the ERA5 window, and registers the result as a model in Azure
ML.

On Azure ML, training uses the compute's user-assigned managed identity for MLflow tracking
and model registration, selected through `DEFAULT_IDENTITY_CLIENT_ID`. Give that identity
AzureML Data Scientist on the workspace, in addition to its storage permissions.
The script discards the submitter-scoped tracking token so an expiring user PIM activation
does not interrupt long runs. Local training keeps its existing MLflow authentication.

Use `inputs.start_date=1990-01-01 inputs.end_date=2015-12-31` to restrict training to
those dates, inclusive. Both history steps and the target must lie inside the window.
Each bound defaults to `unbounded`, preserving training on all available ERA5 timestamps.
Filtering does not fetch missing ERA5 data: the asset must already contain the requested
months, and missing/gapped examples are skipped. Ensure the CO2 CSV covers those months too.
For multi-year training, increase `limits.timeout` from its two-hour default.

Static ERA5 fields are selected at the fetch start date, not the archive's first timestamp
(1900-01-01, which contains missing data). Training rejects non-finite static fields before
loading the model, and checks weather inputs, forcing, targets, predictions, losses and
gradients instead of continuing with NaNs.
Previously fetched assets may contain an all-NaN `static.zarr`: fixing the fetcher does not
repair existing blobs, and rerunning it skips an existing static shard. Regenerate that shard
from a valid timestamp before retraining; the monthly weather shards need not be refetched.

```bash
az ml job create -f jobs/train_aurora_cs_job.yml --set name=train-aurora-cs-2020-01-5ep \
  --set compute=azureml:$GPU_COMPUTE \
  --set inputs.era5_data.path=azureml:era5-weather@latest \
  --set inputs.co2_data.path=azureml:co2-forcing@latest \
  --set inputs.model_name=aurora-cs --set inputs.epochs=5
```

### 3. Roll out (repeatable)

`jobs/rollout_pipeline_job.yml` can be run as often as you like.
Its inputs are the two data assets, the registered model as `<name>/<version>`, a start time and a
number of steps.
The start time must lie inside the fetched ERA5 range, and the CO2 series must cover the rollout
window.

```bash
az ml job create -f jobs/rollout_pipeline_job.yml --set name=rollout-aurora-cs-2020-01-02-40steps \
  --set settings.default_compute=azureml:$CPU_COMPUTE \
  --set jobs.rollout_aurora_cs.compute=azureml:$GPU_COMPUTE \
  --set inputs.era5_data.path=azureml:era5-weather@latest \
  --set inputs.co2_data.path=azureml:co2-forcing@latest \
  --set inputs.model_id=aurora-cs/1 \
  --set inputs.start_time=2020-01-02T00:00:00 --set inputs.num_steps=40 \
  --set inputs.output_prefix=rollouts/2020-01-02 --set inputs.output_asset_name=rollout-2020-01-02 \
  --set inputs.subscription_id=$SUBSCRIPTION_ID --set inputs.resource_group=$RESOURCE_GROUP \
  --set inputs.workspace_name=$WORKSPACE_NAME --set inputs.managed_identity_client_id=$MI_CLIENT_ID
```

The pipeline runs four components.
The two prepare steps are independent and feed the rollout:

- `prepare_initial_condition` takes the ERA5 asset and the start time.
  It writes `batch.pt`, an `aurora.Batch` holding the two-step history window (the start time and
  the step 6 hours before it) of surface, atmospheric and static variables.
- `prepare_forcing_trajectory` takes the CO2 asset, the start time and the number of steps.
  It writes a 1-D tensor with the CO2 concentration (ppm) for each 6-hourly rollout step, repeating
  the monthly value without interpolation.
- `rollout_aurora_cs` takes both outputs and the registered model.
  At each step the model predicts the next 6 hours conditioned on that step's CO2 value, and the
  prediction is fed back in as the newest history step.
  It logs the CO2 value and a simple summary of each prediction to MLflow, and writes every step
  as NetCDF to `<output_prefix>/<YYYYmmddTHHMMSS>.nc` on the workspace blob datastore (the
  timestamp is the time the prediction refers to), and registers the folder as the data asset
  `output_asset_name`.
- `evaluate_rollout` takes the ERA5 asset, plus optional predictions and baseline rollout (neither is wired
  up yet).
  It is meant to score the predictions against ERA5, and against raw Aurora to show what the CO2
  conditioning adds.
  It is currently a stub that logs a warning and produces no output.

## Upcoming features

1. `evaluate_rollout` will be implemented, scoring predictions against ERA5 and, optionally, a
   baseline rollout.
   This needs a baseline mode for `rollout_aurora_cs`, which runs the frozen Aurora without forcing
   from the same initial condition.
   How the baseline is run and passed to `evaluate_rollout` is not decided yet.
   The current pipeline structure may change, for example by splitting rollout and evaluation into
   separate pipelines.
2. The AuroraCS conditioning network will accept a vector of forcing variables instead of CO2
   alone, with sea surface temperature (SST) the first addition.
3. Continuous integration (CI) will be added.
   There is currently no CI at all, so the unit tests only run when someone runs them by hand.
   CI will run them on both CPU and GPU runners.
   Some tests need a GPU and are skipped on machines without one.

## Tests

```bash
pytest
```
