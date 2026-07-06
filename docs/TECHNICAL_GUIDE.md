# GoalInsight 技术文档

> 这是项目的整合版技术指南，整合了 README、architecture、CLAUDE.md、SageMaker、AgentCore Runtime、PPT 提纲六份资料里**面向人类读者**的内容。
>
> 想跑起来 → §3 Quick Start；想看架构 → §2；想看每个 stage 内部 → §4；想看 web/chat → §5；想部署到 AWS → §6。

---

## 1. 这是什么

GoalInsight 把一段固定机位的足球比赛视频，转成结构化数据 + LLM 可对话的分析平台：

- **输入**：一段 MP4（建议 4K，固定机位）
- **输出**：球场标定参数、每一帧每个球员/球的坐标、事件清单（传球/射门/进球/抢断/拦截）、自动剪辑的精彩集锦视频、以及一个能用自然语言问"几号球员跑了多远"的 chat 界面
- **设计目标**：业余比赛能自己跑出来跟商业产品同样的产物（Veo / Pixellot / Trace 给职业队的那些）；本地或自有 AWS 上跑，所有比赛数据不出本机
- **次要目标**：把现代 Bedrock + AgentCore tool-use 在一个非平凡数据集上跑起来当沙盒

**三个一句话关键词**：`field calibration` · `multi-object tracking` · `event detection` · `agent over the resulting data`

---

## 2. 整体架构

### 2.1 端到端流水线（七个 stage）

```
field_registration → tracking → track_consolidation → event_detection
                                                          ↓
            annotated_video ← highlights ← player_profile
```

每个 stage 把 JSON / pkl 写到 `output/<run>/<stage>/`，下一个 stage 只读上一个的产物。**任意 stage 可单跑、可断点续跑**（`--stages tracking,event_detection`、`--skip-existing`）。

| Stage | 输入 | 输出 | 关键算法 |
|---|---|---|---|
| **field_registration** | 视频帧 | `homographies.pkl`, `camera_poses.json`, `calibration_metadata.json` | PnLCalib HRNet / BroadTrack / NBJW / Physical / Homography 五个后端 |
| **tracking** | 视频 + 标定 | `tracks.json`, `ball_tracks.json`, `track_features.json`, `team_assignments.json` | YOLOv8 检测 → StrongSORT/BoT-SORT 追踪 → OSNet/PRTReID 重识别 → KMeans/tracklet 队伍分类；球：YOLO class 32 + center-distance ByteTrack + 二阶段 3D 拟合 |
| **track_consolidation** | tracking 输出 | `players.json`, `tracks.json`（重写带稳定 player_id） | ReID 聚类 → 球衣号 VLM 投票 → 队伍切分 → 孤立 track 吸收 → 命名（A-9 / B-10 / A-GK） |
| **event_detection** | 上述全部 | `events.json` | possession 状态机 → pass / shot / carry / defensive 五个 detector，依赖图 `possession → 其余四个` |
| **player_profile** | tracking + consolidation + (optional) 视频 | `players_profile.json`, `crops/<pid>_{front,back}.jpg`, `heatmaps/<pid>.png`, `spotlights/<pid>.mp4` | 每球员前/后照片、跑动热力图、距离/速度统计、可选 follow-cam 跟拍视频 |
| **highlights** | events + 视频 | `goal_highlight_*.mp4` | recipe-based agent：detector → analyzer → composer，每段 4-segment（建立期 → 射门 → 庆祝 → 慢动作回放） |
| **annotated_video** | 全部 + 视频 | `annotated.mp4` | 全场 HUD 渲染（球员框 + 号码 + 队伍色 + 球轨迹 + 事件横幅） |

### 2.2 Web 应用

单一 FastAPI 实例，多页 vanilla SPA，根目录是一个 **workspace**：

```
workspace/
  videos/                         # 上传/软链的源视频
  annotations/                    # 手工标注（PnLCalib finetune 用）
  models/<ts>/                    # finetune 产生的 best_model.pt
  runs/<run>/                     # 每次 pipeline 输出
    field_registration/  tracking/  track_consolidation/
    event_detection/  player_profile/  highlights/  annotated_video/
    logs/<stage>.log
  jobs.json                       # 任务状态表
  chat_artifacts/<run>/           # chat sandbox 产生的图表
```

页面：

- `/` shell（顶部 Tab 切换）
- `/library` 视频上传 + 已有 run 索引
- `/annotate` 标注 keypoints / lines（PnLCalib finetune 训练集）
- `/pipeline?run=<r>` 阶段化运行控制台 + 中间产物预览 + 训练子卡
- `/match/<run>` 完整比赛页：视频 + 2D 俯视图 + 球员卡片 + 事件列表
- `/insights/<run>?session=<sid>` 视频 + LLM 聊天 + 内置可视化抽屉

启动：`goalinsight-web --workspace ./workspace --port 8000`

### 2.3 Chat 架构（两种模式）

- **本地模式**（默认）：`goalinsight/web/chat.py:ChatEngine` 直接和 Bedrock 跑 tool-use 循环。五个工具：`list_events`、`get_player_stats`、`get_team_stats`、`get_frame_snapshot`、`run_python`（在 AgentCore **Code Interpreter** 沙盒里执行任意 Python，画的图回流成 markdown 图片）。
- **远程模式**（opt-in）：设了 `GOALINSIGHT_AGENTCORE_RUNTIME_ARN` 时，FastAPI 把 chat 走 `bedrock-agentcore.InvokeAgentRuntime`，agent 跑在 AWS 托管的 **AgentCore Runtime** 容器里。浏览器到 FastAPI 的 SSE 流不变，FastAPI 透传。每个 session 第一次时从 S3 拉一份 run JSON，然后整个 MicroVM 生命周期常驻。

> 不管哪种模式：**比赛原始视频和提取数据从不离开机器**。出门的只有 LLM tokens 和 Python 代码片段。

---

## 3. Quick Start

### 3.1 安装

测试环境：Python 3.12 / Ubuntu 22.04 / NVIDIA L40S 或 A10G。CPU-only 能跑较小 stage，但 tracking + calibration 实际是 GPU-bound。

```bash
git clone https://github.com/outstandingcandy/AgentCoach.git
cd AgentCoach
python3.12 -m venv venv && source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 3.2 跑流水线（CLI）

```bash
goalinsight \
  --video data/raw_videos/<your-clip>.mp4 \
  --output output/ \
  --config configs/clip_000_finetuned.yaml \
  --stages field_registration,tracking,event_detection,highlights
```

常用 flag：

| Flag | 作用 |
|---|---|
| `--video`, `--output`, `--config` | 必需 |
| `--stages a,b,c` | 选 stage 子集 |
| `--keypoint-model <path>` | 覆盖 config 里的 PnLCalib 权重 |
| `--remote-stages field_registration,tracking` | 把这两个 stage 上 SageMaker 跑 |
| `--skip-existing` | 跳过已有产物的 stage |
| `--no-timestamp`, `--run-name foo` | 自定义输出子目录 |
| `--no-viz` | 跳过 tracking 可视化视频（省时省盘） |

输出目录：

```
output/<run>/
  field_registration/   homographies.pkl, camera_poses.{pkl,json}, calibration_metadata.json
  tracking/             tracks.json, ball_tracks.json, team_assignments.json, tracking.mp4
  track_consolidation/  players.json, tracks.json (rewrite), consolidated.mp4
  event_detection/      events.json, goals.json
  player_profile/       players_profile.json, crops/, heatmaps/, spotlights/
  highlights/           goal_highlight_0001.mp4 ...
  annotated_video/      annotated.mp4
```

### 3.3 跑 Web UI

```bash
goalinsight-web --workspace ./workspace --port 8000
```

→ 浏览器开 http://localhost:8000/。从 `/library` 上传视频 → `/pipeline` 触发 stage → 完成后 `/match/<run>` 看球，`/insights/<run>` 用 chat 提问。

### 3.4 复现 kids_soccer demo

60 秒 demo 视频 + finetune 权重在 S3（~520 MB，没在 git 里）：

```bash
export GOALINSIGHT_S3_BUCKET=<your-bucket>
mkdir -p workspace/videos data/finetuned_models/run_20260605_073045/models \
                          data/finetuned_line_models/run_20260605_073744/models

aws s3 cp "s3://$GOALINSIGHT_S3_BUCKET/raw_videos/kids_soccer_clip_1250_1310.mp4" workspace/videos/
aws s3 cp "s3://$GOALINSIGHT_S3_BUCKET/finetuned_models/run_20260605_073045/best_model_final.pt" \
          data/finetuned_models/run_20260605_073045/models/
aws s3 cp "s3://$GOALINSIGHT_S3_BUCKET/finetuned_line_models/run_20260605_073744/best_model_final.pt" \
          data/finetuned_line_models/run_20260605_073744/models/

goalinsight --video workspace/videos/kids_soccer_clip_1250_1310.mp4 \
            --output output/kids_demo --config configs/kids_soccer_physical.yaml
```

7 帧 v2 finetune 训练集已 checked-in 到 `output/annotations/kids_soccer_v2/`，可以直接复现 keypoint/line finetune。

---

## 4. Stage 深入说明

### 4.1 Field Registration — 相机标定

把每帧画面映射到真实球场坐标系，所有「以米为单位」的下游判断都依赖它。

| Backend | 说明 | 适用场景 |
|---|---|---|
| **PnLCalib**（默认） | 迭代 PnP + 多候选扫 + LM 优化 + 5 参数畸变；HRNet 检测 keypoints / lines | FIFA 标准球场，broadcast 视角 |
| **BroadTrack** | 9 参相机模型 + Cauchy robust loss + 弧长线段约束 | 同上，对抖动更鲁棒 |
| **NBJW** | PnLCalib 的另一权重组合 | 备选 |
| **Physical** | 固定相机内参（来自 `configs/camera_profiles.yaml`），优化 7-DOF 外参，2-pass | **非 FIFA 球场**（青少年球场尺寸异常），最稳 |
| **Homography** | 直接 DLT 单应矩阵 | 简单、最快、精度最低 |

PnLCalib 的 HRNet keypoint 和 line head 支持 finetune（`scripts/train_finetune.py`），训练集来自 web app 的 Annotate Tab 或 `workspace/annotations/<video_stem>/`。

**关键点格式**：输入 115 个 SoccerNet-GSR keypoints；内部用 57 个 PnLCalib keypoints；4 个非地平面横梁点（IDs 12/14/16/18）在 z=-2.44m。

### 4.2 Tracking — 多目标追踪

`tracking/orchestrator.py` 跑一个多线程 I/O pipeline：帧预取 → YOLOv8 推理 → tracking/ReID/team 分类 → 输出写盘。

**多目标追踪 backend**（`tracking.backend`）：

- **StrongSORT**（默认）：级联匹配 (tentative-IoU → tentative-pitch → confirmed-IoU → confirmed-ReID)，pitch 距离门限以**米**计、不是像素，Kalman 滑行处理，"静止 track 杀手"过滤广告板/护栏假阳性。在 `tests/tracking/` 有单测覆盖。
- **BoT-SORT**：GMC 感知备选；ReID 走它自己的 embedding 接口。

**ReID 后端**（`reid.backend`）：

- **OSNet**（默认）：512 维，TorchReID 的 `osnet_x1_0`
- **PRTReID**：256 维 part-based ReID（带 albumentations 2.x 兼容垫片）

**球追踪 + 3D 轨迹**：YOLO class 32 检测，ByteTrack/BOTSORT 用 center-distance 匹配（小框 IoU 没意义），二阶段拟合：(1) 像素加速度突变断点分段 (2) 每段判断地滚 vs 空中段，地滚走 Z=0 投影，空中段拟合 `P(t) = [x0+vx·dt, y0+vy·dt, z0+vz·dt-0.5·g·dt²]` 带边界 ground-contact 锚点。

### 4.3 Track Consolidation — 稳定身份

跟踪器吐 `track_id` 因遮挡反复断裂；这一步把它们合并回稳定的 `player_id`（`A-9` / `B-10` / `A-GK`）。五阶段 greedy pipeline：

- **A** ReID-first 聚类（cosine ≥ `same_person_threshold`，时间不能共现）
- **B** 每聚类做球衣号置信度加权投票（image-list LLM 调用，融合冗余高置信样本，挽救低置信误读）
- **C** Team split：同一聚类跨队的 track 拆开
- **D** 孤立 track 在更松阈值下并入已有聚类
- **E** 命名（A-9 / B-10 / A-GK / B-unk-01）

球衣号识别后端（`track_consolidation.jersey.backend`）：`claude` / `gemini` / `qwen` / `rapidocr`。

### 4.4 Event Detection — 事件检测

config-driven detector 框架，**possession 是地基**，其它 4 个事件靠它推导。

**Detector 依赖图**：`possession → {pass, shot, carry, defensive}`

| Detector | 输出 |
|---|---|
| **possession** | 状态机，跟踪 ball-player 邻近持续帧，emit possession spans |
| **pass** | possession 切换 + 球速突变；分类 successful/failed；捕捉一脚出球（中间无 carry） |
| **shot** | 球速 + 朝球门轨迹；outcome ∈ {Goal, Saved, Off_Target, Blocked}。射手归属用 **pre-kick possession** 而不是「球离脚瞬间最近的人」——后者会把短暂逼近的防守球员误算成射手 |
| **carry** | 持续控球期间有显著前向推进 |
| **defensive** | tackle（控球切换 + 球被踢偏）、interception（传球失败 + 控球切换） |

每个 event 是一个 `MatchEvent` dataclass，有 `event_type` / `frame` / `player_id` / `team_id` / `metadata`，全部写到 `events.json`。

### 4.5 Player Profile — 球员档案

每个 player_id 产出：

- `crops/<pid>_front.jpg` / `<pid>_back.jpg`：分别是面向 / 背向相机的最佳照片
- `heatmaps/<pid>.png`：球场跑动热力图
- `players_profile.json`：跑动距离 / 平均/最大速度 / 触球数 / 传球数 / 射门数 / 进球数
- `spotlights/<pid>.mp4`（**opt-in**）：跟拍视频，球员居中、约 2/3 画面高，带 spotlight 椭圆 + 球员名牌；从原始 4K 源直接裁出 1080p。`<pid>.frames.json` 是 spotlight 时间轴 → broadcast 帧索引的映射，给前端俯视图同步用。

### 4.6 Highlights — 精彩集锦

Recipe-based 三步：**Detector → Analyzer → Composer**。

- `GoalEventDetector`：从 `events.json` 过滤 `type=GOAL`
- `ScorerAnalyzer`：用 event metadata 里的 `player_id` / `team_id`（不重复算射手归属），产出 4-segment 计划：建立期（wide，跟球）→ 射门（closeup，球穿过画面）→ 庆祝（medium，跟射手；track 丢了就截断）→ 回放（射门段慢动作 + RIFE 光流插帧）
- `SegmentComposer`：按 segment 渲染，含视效（射手聚光灯、球轨迹拖尾），可选 video2x 超分 → 裁剪/特效都在高分辨率上做

### 4.7 Annotated Video — 全场叠加

读全部上游产物，写一个完整比赛的 `annotated.mp4`，HUD 包括：球员队伍色框 + jersey 号、球的轨迹拖尾、地图投影球场线、事件横幅。可选 video2x 超分。

---

## 5. Web 应用

### 5.1 库 + Pipeline 控制台

`/library` 上传 mp4 → 落到 `workspace/videos/`，可选 cover 截图。`/pipeline?run=<r>` 左列 stage 卡片（field_registration / tracking / track_consolidation / event_detection / player_profile / highlights / annotated_video / consolidated_overlay）+ 训练子卡（finetune_keypoints / finetune_lines）；右列 SSE 实时 log + 产物链接。任务都走 `JobManager`（`workspace/jobs.json` 持久化）。

### 5.2 Insights 页（视频 + chat + 内置可视化）

`/insights/<run>?session=<sid>`：

- 左：annotated 视频 + 顶部 meta（队伍、事件计数）
- 右：多 session 持久化 chat
- 右下抽屉：Heatmap / Stats / Shot map / Pass network 一键出图

Chat 五个工具：

| 工具 | 用途 |
|---|---|
| `list_events` | 按类型/队伍/球员/时间窗口过滤事件 |
| `get_player_stats` | 单球员距离 / 最高速 / 触球 / 传球 / 射门 / 进球 |
| `get_team_stats` | 控球率 / 传球成功率 / 射门 / 抢断 / 拦截 |
| `get_frame_snapshot` | 「这一刻屏幕上有谁、球在哪」 |
| `run_python` | 在 AgentCore Code Interpreter 沙盒执行 Python，画图回流为内联 markdown image |

### 5.3 Match 页

`/match/<run>`：上方比赛视频，右侧 2D 俯视图（实时跟随播放头），下方球员按钮。点球员按钮：
- 主视频切换到该球员的 spotlight clip
- 俯视图按 sidecar mapping 把 spotlight 时间换算回 broadcast 帧 → 同帧画 22 名球员，被选中的球员高亮（白色描边）
- 左上角"Spotlight: A-9 / Back to broadcast"返回按钮

### 5.4 Annotator

`/annotate` 是 PnLCalib finetune 训练集生产工具：在原始视频帧上手工点 keypoint 或画 line，自动派生交点，模型自动建议剩余 keypoint，全部 promote/accept/reject 后保存为 `frame_<idx>_all_points.json`，直接给 `scripts/train_finetune.py` 当训练集。

---

## 6. AWS 部署（可选）

两个独立模块，都可以 opt-in 单独开启：

### 6.1 SageMaker Processing Jobs（远程 stage 执行）

把 `field_registration` 和 `tracking`（最贵的两个）上 SageMaker GPU 集群跑：

```bash
# 一次性
bash sagemaker/setup_aws.sh         # 写 sagemaker.{region,role_arn,image_uri,s3_bucket} 进 config
bash sagemaker/upload_weights.sh    # 上传 PnLCalib + YOLO 权重到 S3
bash sagemaker/build_and_push.sh    # 构建 ECR 镜像

# 每次跑
goalinsight ... --remote-stages field_registration,tracking
```

`pipeline/_remote.py` 会上传输入、提交 job、轮询、把白名单产物拉回本地，落到跟本地版完全相同的目录布局。下游 stage 不知道上游是本地跑还是远端跑。

详细文档：[`sagemaker/README.md`](../sagemaker/README.md)。

### 6.2 AgentCore Runtime（远程 chat agent）

把 chat agent（`ChatEngine` + 工具分发）从本地 FastAPI 进程里抽出来，放进 AWS 托管的 AgentCore Runtime 容器：

```bash
cd deploy/agentcore_runtime
bash deploy.sh    # 构建 ARM64 镜像 + 推 ECR + 部署 runtime
export GOALINSIGHT_AGENTCORE_RUNTIME_ARN=arn:aws:bedrock-agentcore:...
goalinsight-web --workspace ./workspace --port 8000
```

浏览器路径不变，FastAPI 走 `bedrock-agentcore.InvokeAgentRuntime` 透传 SSE。每个 session 第一次从 S3 拉 run JSON，常驻 MicroVM。

详细文档：[`deploy/agentcore_runtime/README.md`](../deploy/agentcore_runtime/README.md)。

### 6.3 ALB + Cognito（生产入口）

`deploy/alb-cognito.yaml` 是一个 CloudFormation stack，把 `goal-insight-web` 服务（端口 8000）放到 ALB 后面，前面 Cognito 做 OIDC 认证。`deploy/bootstrap.sh` 是 EC2 user-data 脚本，新机器开机即装。

---

## 7. 配置

YAML 配置在 `configs/`，user config 通过深合并叠到 `configs/default.yaml` 上（`merge_configs`）。重要 keys：

| Key | 说明 |
|---|---|
| `pipeline.stages` | 要跑的 stage 列表 |
| `field_registration.backend` | `pnlcalib` / `broadtrack` / `physical` / `nbjw` / `homography` |
| `field_registration.keypoint_threshold`, `ransac_threshold` | 默认 30px |
| `video.process_fps` | 全局帧采样率（`video.tracking_fps` override 给 tracking） |
| `tracking.backend` | `strongsort`（默认）/ `botsort` |
| `tracking.dump_yolo_raw` | 把 YOLO raw 检测落盘到 `yolo_raw/` 给离线 audit |
| `reid.backend` | `osnet` / `prtreid` |
| `team_classification.backend` | `kmeans` / `tracklet` |
| `track_consolidation.jersey.backend` | `claude` / `gemini` / `qwen` |
| `events.detectors` | enabled detector 列表 |
| `player_profile.spotlights.*` | spotlight 视频开关 + 输出尺寸 / 球员高度比例 / 出场跳过 / 椭圆 / 名牌 |
| `highlights.recipes` | 集锦 recipe 列表 |
| `video_enhancement.{enabled,mode}` | video2x 超分（binary 或 docker） |
| `output.save_visualizations` | 是否生成各 stage 可视化产物 |

每个视频可以放一份 sparse `overrides.yaml` 在视频文件旁边，覆盖默认（kids/youth 配置就这么用）。

---

## 8. 测试

`tests/tracking/` 覆盖 StrongSORT 包（gates / matching / lifecycle）。`pytest tests/`。

仓库根目录的 `test_*.py` / `debug_*.py` 是 gitignored 的一次性脚本，不是正式测试。

---

## 9. 进一步阅读

| 文档 | 内容 |
|---|---|
| [`README.md`](../README.md) | 上手引导（这份文档前 1/3 的精简版） |
| [`architecture.md`](../architecture.md) | mermaid 流水线图 |
| [`docs/architecture/`](architecture/) | 渲染好的架构 PNG（pipeline_local / pipeline_remote / chat） |
| [`CLAUDE.md`](../CLAUDE.md) | 给 AI 看的 cheat-sheet（结构跟本文档一致，更密） |
| [`sagemaker/README.md`](../sagemaker/README.md) | SageMaker 远程 stage 部署 |
| [`deploy/agentcore_runtime/README.md`](../deploy/agentcore_runtime/README.md) | AgentCore Runtime 部署 |
| [`ppt_outline.md`](../ppt_outline.md) | 9 页技术分享 PPT 提纲 |

---

*最后更新：2026-06-18。本文档对照实际代码维护，如发现描述与代码不符以代码为准。*
