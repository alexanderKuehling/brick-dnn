# Scaling beyond one server

Where trainings run today, and what would change when one server is no longer enough. The
first part describes what is built; the second is a **plan, not built yet**.

## Stage 1: one server, a queue and a worker (built)

Everything runs with Docker Compose on one server ([deploy/server](../deploy/server/README.md)).
The web server and the trainings are separate:

```text
browser ─▶ Caddy ─▶ nginx ─▶ Django (backend) ──▶ database: runs, queue, workers, usage
                                  ▲                     ▲
                                  │ state.json,         │ claim, heartbeat, usage
                                  │ weights (volume)    │
                               data volume ◀──── worker: manage.py training_worker
                                                  └─ one runner process per training
```

- **The queue is in the database** ([training.md](training.md#the-training-queue)): a start
  stores the run as `queued`; the worker takes the next one when it has a free slot, runs of
  plans with a higher priority first.
- **The worker is its own container** with fixed CPU and memory limits, so trainings cannot
  take what the site needs. It reports through the database (status, heartbeat, usage) and
  through the run directory on the data volume, which web server and worker share
  (`job.json`, `state.json`, the weights, uploaded datasets).
- **The numbers that say when this is too small** are in the Django admin (*Training
  workers*) and in `manage.py training_queue`: how long trainings waited for a slot in the
  last seven days (median, 95th percentile, per plan) and how busy the worker was.

Before leaving one server there is room on it: a larger server (the limits follow its size,
see [deploy/server](../deploy/server/README.md#cpu-and-memory-the-site-comes-first)) and more
slots (`BRICKDNN_WORKER_SLOTS`, with fewer threads per training).

What a worker needs is deliberately little, so that the same command can run elsewhere
without changes to how a job is run:

| The worker … | through |
| --- | --- |
| takes a job, safely next to other workers | the database (`SELECT … FOR UPDATE SKIP LOCKED`, an advisory lock per account) |
| says that it is alive; its runs are started again if it stops saying so | the database (`TrainingWorker.heartbeat_at`) |
| learns about stop requests and deleted runs | the database (and the `STOP` file it writes for the runner) |
| counts usage and enforces budgets | the database (`UsageRecord`, the limits of the plan) |
| reads datasets, writes progress and weights | the data directory |
| can end after one job | `manage.py training_worker --once` (with `--idle-exit SECONDS` if there is none) |

Only the data directory ties the worker to the server of the web app.

## Stage 2: Kubernetes with autoscaling (planned, not built)

The plan for when the queue's numbers say that one server is not enough (roughly: ten paying
customers, or waits that stay long although the worker is busy all day). Nothing of this
exists in the repository yet; it is here so that the first stage does not build in the wrong
direction.

**The same worker as a Job.** The container image and `manage.py training_worker --once
--idle-exit 60` stay as they are. Instead of one long-running worker container, a controller
starts one Kubernetes Job per waiting run (for example KEDA's `ScaledJob` with a PostgreSQL
query on the queued runs, or a small loop in the web app's namespace); each Job claims one
run, trains it and ends. The cluster autoscaler adds and removes nodes for these pods, so
capacity follows the queue and an idle queue costs nothing. Priority, fairness, the limits
per account, budgets, heartbeats and the restart rules are in the database and work
unchanged with many workers (claiming is already safe with several, and tested on
PostgreSQL). A worker on a second plain server would work the same way.

**What has to change first** (the open points):

1. **Files: object storage instead of the shared volume.** Datasets, run directories and
   weights must be reachable from every node. Planned: an S3-compatible bucket (Hetzner
   Object Storage). The worker downloads the dataset before a job and uploads the weights and
   the log when it ends; the web server serves downloads from the bucket (signed links).
   The code already goes through a few functions for this (`training_service.run_dir`,
   `dataset_path`, `read_run_state`, `weights_path`, `start_claimed`), which is where a
   storage backend would be plugged in.
2. **Progress: through the database instead of `state.json`.** The web server reads live
   progress from a file on the shared volume today. Planned: the worker writes what it reads
   from the runner's `state.json` (which stays the runner's interface) to the run's row or a
   progress table every second or two, and `read_run_state` reads from there.
3. **Predictions** still start a short process inside the web container, which needs PyTorch
   and the weights on local disk. With object storage they need a local cache of the weights,
   or they become short jobs of their own.
4. **Per-job resources.** A Job can ask for CPUs and memory per run (from the plan and the
   estimate) instead of the fixed slots of one worker; GPU nodes become possible.
5. **Leftovers of a lost pod.** On one machine a worker kills a runner that outlived its
   predecessor; across nodes a pod that lost its database connection for longer than
   `BRICKDNN_WORKER_TIMEOUT` stops its own trainings, and with per-job storage a late write of
   a lost pod must not overwrite the files of the attempt that replaced it (write under an
   attempt id). Heartbeats are compared with the clock of whoever looks; machines whose
   clocks differ by more than a few seconds should use the time of the database instead.
6. **Deploys.** Rolling out a new image no longer interrupts running trainings: Jobs that
   started keep their image until they end.

The web app, the database (Supabase) and the realtime service can move to the cluster as
ordinary Deployments at the same time or later; they do not depend on where the workers run.
