# GoalInsight 技术架构 PPT — 8 页提纲

> 直接在这个文件里改:增 / 删 / 改标题 / 改 rhythm / 改要点都行。
> 改完跟我说一声"按 ppt_outline.md 出 PPT",我会读这个文件并按你的最新版本生成。
>
> 字段含义:
> - `title`:页标题(中文,显示在页面上)
> - `rhythm`:`anchor` (封面/收尾) / `dense` (信息密集) / `breathing` (低密度强调)
> - `points`:页面要展示的关键信息(每条 ≈ 一段或一个 chip),列表越长这页越密
> - `notes`:这页 speaker note 的口语化方向,一两句即可

---

## P01 — 封面

- title: GoalInsight · 技术架构概览
- rhythm: anchor
- points:
  - 主标题:GoalInsight
  - 副标题:技术架构概览 — 从视频到 Agent 的端到端足球分析流水线
  - 角标:Engineering deep-dive · 2026
- notes: 简单介绍今天讲的是这个项目的技术架构,面向工程师。

---

## P02 — 一句话讲清这个项目

- title: 一句话讲清这个项目
- rhythm: breathing
- points:
  - 中心句:**利用代码对足球视频进行战术分析的 Agent**
  - 三步链:视频 → 结构化数据 → Agent
  - 关键能力:Agent 把自然语言问题翻译成 Python,在沙箱里跑数据、出图、回答
  - 例子提问:"前锋进球前的跑动热区"、"进球时防守球员站位"
  - 4 个关键词 chip:`field calibration` / `multi-object tracking` / `event detection` / `code-writing agent`
- notes: 输入一段比赛视频,产出结构化数据,然后 Agent 会读懂问题、写 Python、跑出战术分析答案 — 用户直接用自然语言提问就行。

---

## P03 — 视频处理 Pipeline 流程图

- title: End-to-End Pipeline · 5 个 stage 顺序执行
- rhythm: dense
- points:
  - 横向流程链:`field_registration → tracking → track_consolidation → event_detection → player_profile`
- notes: 五个 stage,前一个的 JSON 输出是后一个的输入,文件契约让任意 stage 可断点续跑。

---

## P04 — 视觉前端 · Calibration

- title: Stage 1 · Field Registration · 相机标定 (physical backend)
- rhythm: dense
- points:
  - 目标:把每帧像素映射到真实球场坐标(米),让下游所有距离/速度判断都成立
  - 为什么选 physical:业余比赛镜头固定、场地常常非 FIFA 标准 — 用已知内参约束、只优化 6-DOF 外参,比强行学相机内参更稳
  - **关键点 + 场地线检测**:HRNet 出每帧的 keypoints + line segments → 与球场模板匹配
  - **Pass 1 · Per-frame PnP**:每帧 RANSAC PnP 求外参 (rvec, tvec);带 temporal warm-start,前一帧 pose 给下一帧做初值,reproj error 超 15px 才重置
  - **Pass 2 · Locked-C 精修**:相机机位固定 → 取 Pass 1 trust 帧的 tvec 中位数作为相机位置 C,锁住 C 后只优化朝向,消除单帧抖动
  - **Chain gap-fill**:reproj error 高 / 关键点少的帧用前后帧 pose 做时序插值兜底
- notes: 用相机内参先验 + 6-DOF 外参的两 pass + 时序兜底,把每帧画面映射到真实球场坐标。physical backend 在非标场地最稳,因为它不指望从画面里学一个准确的相机模型。

---

## P05 — 视觉前端 · Tracking + Consolidation

- title: Stage 2 + 3 · 从像素 box 到稳定 player_id
- rhythm: dense
- points:
  - **Stage 2 · Tracking** — 检测 + 关联
    - 目标检测:YOLOv8(球员 + 球)
    - 多目标追踪:StrongSORT(IoU + ReID 关联)
    - 输出:`tracks.json` / `ball_tracks.json`(track_id 会因遮挡反复断裂)
  - **Stage 3 · Track Consolidation** — 把碎片化 track_id 合并成稳定 player_id
    - **A** ReID 聚类(cosine ≥ same_person_threshold,无时间共现)
    - **B** 球衣号 VLM 投票
    - **C** Team split
  - 输出:`players.json` / `tracks_consolidated.json` —— 下游事件 metadata 用稳定 ID `A-9` / `B-10`,不再是 raw int
- notes: 第二个 stage 检测每一帧的人和球、把同一个目标在时间上串起来;第三个 stage 把因遮挡断裂的 track_id 用 ReID + 球衣号合并回稳定球员身份,LLM 工具调用才有意义。

---

## P06 — Event Detection

- title: Event Detection · Possession 是地基,其它四个事件靠它推导
- rhythm: dense
- points:
  - 依赖图:`possession → {pass, shot, carry, defensive}`
  - **possession**:状态机,跟踪 ball-player 邻近持续帧
  - **pass**:possession transition + ball speed jump,捕捉 one-touch
  - **shot**:speed + 朝球门轨迹,outcome ∈ {Goal, Saved, Off_Target, Blocked};用 pre-kick possession 归属射手
  - **carry / defensive**:dribble / tackle / interception
- notes: Possession 是地基,其它四个 detector 都从它的输出推导事件;事件元数据带 player_id 给下游用。

---

## P07 — ⭐ AgentCore Runtime + Code Interpreter

- title: AgentCore Runtime + Code Interpreter · 把 Agent 搬到 AWS
- rhythm: dense
- points:
  - **两块 AWS 托管 Agent 基建**:Runtime 跑 Agent · Code Interpreter 跑 Agent 写的代码
  - **Runtime · Agent 容器化**(`deploy/agentcore_runtime/`)
  - **Code Interpreter · Agent 写的 Python 在沙箱里跑**(`code_sandbox.py`)
  - **战术分析的实际链路**:用户问"前锋进球前的跑动热区" → Runtime Agent 解读问题 → 工具调用 `run_python(...)` → Code Interpreter 跑 pandas/matplotlib → PNG + 文字回答 SSE 流回前端
- notes: AgentCore 提供两块基建:Runtime 把 Agent 容器化、共享 session 状态;Code Interpreter 让 Agent 写的 Python 在隔离沙箱里跑数据、出图。两者一起就是"会写代码做战术分析的 Agent"在 AWS 侧的落地。

---

## P08 — 整体架构图

- title: 整体架构图 · 视频处理 + Agent 双栈
- rhythm: dense
- points:
  - **左侧 · 视频处理栈**(本地 / 离线)
    - `raw video` → `field_registration` → `tracking` → `track_consolidation` → `event_detection` → `player_profile`
    - 产物:`tracks.json` · `players.json` · `events.json` · `players_profile.json` 落到 S3 `goalinsight-pipeline-<account>/runs/<run>/`
  - **中间 · S3 数据契约**:把视频处理栈和 Agent 栈解耦的边界 — 一边写,另一边按 run 名拉
  - **右侧 · Agent 栈**(在线 / 用户提问时触发)
    - **FastAPI Web App**:5 个 Tab(library / pipeline / insights / match / annotate)+ chat SSE 代理
    - **AgentCore Runtime**(opt-in):ARN 已设时 chat 路由到容器,容器内 ChatEngine 调 Bedrock Claude + 5 个 tool
    - **AgentCore Code Interpreter**:`run_python` 工具的执行后端,沙箱里跑 pandas / matplotlib
  - **典型一次提问的旅程**:Browser → FastAPI `/chat` SSE → InvokeAgentRuntime → ChatEngine.stream() → tool=run_python → Code Interpreter → PNG → S3 artifact → SSE delta 回前端渲染
  - **配置开关**:`GOALINSIGHT_AGENTCORE_RUNTIME_ARN` 控本地/远程 · `GOALINSIGHT_S3_BUCKET` 数据源 · `GOALINSIGHT_CHAT_ARTIFACT_BUCKET` 出图存桶
- notes: 用一张图把今天讲过的所有部分串起来:左边离线把视频处理成结构化 JSON,中间用 S3 当契约,右边在线让 Agent 读 JSON、写代码、出战术分析。
