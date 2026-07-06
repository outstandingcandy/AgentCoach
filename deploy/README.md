# Deploying the GoalInsight web app

Two ways to run the web app on AWS:

## A. One-click: fresh GPU EC2 + ALB + Cognito (public internet)

```bash
bash deploy/deploy_ec2.sh you@example.com
```

Provisions everything from scratch and wires up public HTTPS access:

1. Discovers the default VPC + two public subnets (override with
   `--vpc-id` / `--subnet-ids`).
2. Creates an IAM role + instance profile (Bedrock for chat, SSM for a
   keyless shell).
3. Creates the instance security group — **SSH 22 only**, locked to your
   current IP. The app port 8000 is opened **only to the ALB's security
   group** by the CloudFormation stack; it is never public.
4. Launches a GPU instance (default `g5.xlarge`) from the Deep Learning
   AMI. First boot runs `deploy/ec2_userdata.sh`: clones the repo, builds
   the venv, `pip install`s the app, and starts it under systemd
   (`deploy/goal-insight-web.service`).
5. Hands off to `deploy/bootstrap.sh` to import a self-signed cert and
   deploy the ALB + Cognito stack (`deploy/alb-cognito.yaml`) against the
   new instance.

Options: `--region`, `--instance-type`, `--key-name`, `--branch`,
`--vpc-id`, `--subnet-ids`, `--volume-size`, `--suffix`. All also settable
via env var.

### Parallel deployments (`--suffix`)

The CFN template uses fixed resource names (ALB, target group, ALB SG,
Cognito user pool), so a second stack in the same region would collide.
Pass `--suffix -v2` (or `SUFFIX=-v2`) to append a unique suffix to the CFN
stack name, the `ResourceSuffix` template param, the Cognito domain, the
instance `Name` tag, and the instance SG name — leaving any existing
deployment untouched. With no suffix the names reduce to the originals
(backward compatible with the first stack).

**Caveats**

- `g5.xlarge` (A10G 24 GB) is ~$1/hr on-demand — not free tier.
- First boot takes ~15–25 min (torch/mmcv/mmocr install). The ALB target
  stays **unhealthy** until the service is up — that's expected.
- The self-signed cert makes browsers warn once ("Advanced → Continue").
- Bedrock model access for `us.anthropic.claude-opus-4-7` in `us-east-1`
  must be enabled in the account, or chat calls will 403.
- Re-running is safe: IAM role, security group, instance (by
  `Name=goal-insight-web<suffix>` tag) and the CFN stack are all reused,
  not duplicated.

### Live deployment (deployed 2026-07-06)

A parallel `-v2` stack is currently running alongside the original prod
stack:

| | Value |
|---|---|
| Region | `us-east-1` |
| CFN stack | `goal-insight-alb-cognito-v2` |
| Instance | `i-0996cd3468c0a557a` (`g5.xlarge`) |
| Instance SG | `sg-0e68ca0c0aa575023` (SSH 22 from admin IP only; :8000 from ALB SG only) |
| URL | `https://goal-insight-alb-v2-1803414088.us-east-1.elb.amazonaws.com` |
| Cognito admin | `tangjiee@amazon.com` |

Reproduce with: `SUFFIX=-v2 bash deploy/deploy_ec2.sh tangjiee@amazon.com`

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
