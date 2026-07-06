# Deploying the GoalInsight web app

Two ways to run the web app on AWS:

## A. One-click: fresh GPU EC2 + ALB + Cognito (public internet)

```bash
bash deploy/deploy_ec2.sh you@example.com
```

**Everything is provisioned by one CloudFormation stack**
(`deploy/full-stack.yaml`) — the EC2 instance, its IAM role + instance
profile, its security group, the ALB, and the Cognito user pool. Deleting
the stack removes all of it; nothing is left dangling outside CFN. The
script itself only does what CFN can't:

1. Discovers the default VPC + two public subnets (override with
   `--vpc-id` / `--subnet-ids`).
2. Generates a self-signed cert and imports it into ACM (ACM
   `ImportCertificate` is not a CFN resource type — this is the one
   out-of-band prerequisite).
3. Deploys / updates the stack, passing in the VPC, subnets, cert ARN,
   caller IP (for SSH), admin email, etc.

The stack then:

- Creates the **IAM role + instance profile** (Bedrock for chat, SSM for a
  keyless shell).
- Creates the **instance security group** — **SSH 22 only**, locked to your
  current IP (omitted entirely if the IP can't be resolved). App port 8000
  is opened **only to the ALB's security group** — never public.
- Launches a GPU instance (default `g5.xlarge`) from the Deep Learning AMI
  (resolved in-stack via an SSM parameter). Its UserData clones the repo and
  runs `deploy/ec2_userdata.sh`: installs Docker + the NVIDIA container
  toolkit, **builds the deployment image** (`deploy/Dockerfile`) on the
  instance, downloads model weights into the host workspace, and starts the
  **container** under systemd (`deploy/goal-insight-web.service`).
- Wires up the ALB (HTTP→HTTPS redirect, HTTPS listener with Cognito auth →
  forward to the instance).

## Teardown

```bash
bash deploy/teardown.sh [--suffix <s>] [--purge-cert] [--yes]
```

Deletes the whole stack (instance, IAM, SGs, ALB, Cognito). The imported
ACM cert is shared/reusable and left in place unless `--purge-cert` is
passed. Use the same `--suffix` you deployed with.

### Containerized runtime

The app runs as a Docker container, not bare-metal:

- **Image** `deploy/Dockerfile` — CUDA 12.1 + Python 3.12 + the offline
  requirements set + `pip install -e .`. Chat is **enabled** (unlike the
  credential-free `deploy/offline/Dockerfile`), and no model weights are
  baked in.
- **Build** happens on the instance at first boot (no ECR needed). This is
  the bulk of the ~15–25 min provisioning time.
- **systemd unit** runs `docker run --gpus all -p 8000:8000
  -v .../workspace:/workspace -e AWS_REGION=us-east-1 goalinsight:deploy`
  with `Restart=always`. The bind-mounted workspace holds videos, runs, and
  downloaded weights so they survive container recreation.
- **Weights** (YOLO / OSNet / PnLCalib, plus the `ev_posw` futsal keypoint
  fine-tune from a GitHub Release) download into the workspace volume, so
  they persist and show up in the web "Pick a keypoint model" picker.

Options: `--region`, `--instance-type`, `--key-name`, `--branch`,
`--vpc-id`, `--subnet-ids`, `--volume-size`, `--suffix`. All also settable
via env var.

### Parallel deployments (`--suffix`)

`full-stack.yaml` lets CFN auto-name its resources, so parallel stacks don't
collide by default. Pass `--suffix -v2` (or `SUFFIX=-v2`) to get a distinct
stack name (`goal-insight-full-v2`) and Cognito domain, leaving any existing
deployment untouched. Pass the same `--suffix` to `teardown.sh` to remove it.

**Caveats**

- `g5.xlarge` (A10G 24 GB) is ~$1/hr on-demand — not free tier.
- First boot takes ~15–25 min (install Docker + NVIDIA toolkit, then
  `docker build` the image = torch + ML stack). The ALB target stays
  **unhealthy** until the container is up — that's expected.
- The self-signed cert makes browsers warn once ("Advanced → Continue").
- Bedrock model access for `us.anthropic.claude-opus-4-7` in `us-east-1`
  must be enabled in the account, or chat calls will 403.
- Re-running `deploy_ec2.sh` performs a CloudFormation stack update in
  place — it does not create duplicate resources.

### Live deployment (deployed 2026-07-06)

A `-v2` stack is currently running. NOTE: it was created with an earlier
version of these scripts (split `bootstrap.sh` + `alb-cognito.yaml`, bare-
metal venv), before the single-stack + containerized rework, so its stack
name is `goal-insight-alb-cognito-v2` and some resources (IAM role, instance
SG) live outside that stack. Re-deploy with the current `deploy_ec2.sh` to
get the fully-CFN-managed, containerized layout.

| | Value |
|---|---|
| Region | `us-east-1` |
| CFN stack | `goal-insight-alb-cognito-v2` (legacy split stack) |
| Instance | `i-0ea3b5413e91ff78c` (`g5.xlarge`) |
| URL | `https://goal-insight-alb-v2-1803414088.us-east-1.elb.amazonaws.com` |
| Cognito admin | `tangjiee@amazon.com` |

Watch first-boot progress:

```bash
aws ssm start-session --target <instance-id>
sudo tail -f /var/log/goal-insight-provision.log
```

Only SSH 22 (from your IP) is open on the instance. To connect without a
key pair, use SSM Session Manager as above.

## B. Local / manual launch on an already-provisioned host

`deploy/start_web.sh` runs the server in the foreground on an existing
checkout with a prebuilt `.venv` (this is what the current dev host uses).
It is unchanged and independent of the one-click path above.

## C. Front an already-existing EC2 (legacy split stack)

`deploy/bootstrap.sh` + `deploy/alb-cognito.yaml` deploy only the ALB +
Cognito in front of an EC2 instance you already run (you pass its instance
ID / SG). This is the original pre-`full-stack.yaml` path, kept for the
existing raw2real prod stack. For new deployments prefer the one-click
`deploy_ec2.sh` above, which provisions the instance too.
