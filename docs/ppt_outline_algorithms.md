# GoalInsight 算法 PPT — 12 页提纲

> 基于 `docs/ALGORITHMS.md` 生成，面向算法 / 研究者读者。
> 直接在这个文件里改：增 / 删 / 改标题 / 改 rhythm / 改要点都行。
> 改完跟我说一声"按 ppt_outline_algorithms.md 出 PPT"，我会读这个文件并按最新版本生成。
>
> 字段含义：
> - `title`：页标题（中文，显示在页面上）
> - `rhythm`：`anchor`（封面/收尾） / `dense`（信息密集） / `breathing`（低密度强调）
> - `points`：页面要展示的关键信息（每条 ≈ 一段或一个 chip），列表越长这页越密
> - `notes`：这页 speaker note 的口语化方向，一两句即可

---

## P01 — 封面

- title: GoalInsight · 算法概览
- rhythm: anchor
- points:
  - 主标题：GoalInsight
  - 副标题：从一段固定机位的足球比赛视频，提取**结构化数据 + 语义事件**，并支持自然语言提问
  - 角标：Algorithms deep-dive · 2026
- notes: 这一份是面向算法 / 研究者的版本，重点讲思路和取舍，不讲接口。

---

## P02 — 背景 · 问题与端到端模型的局限

- title: 视频的细粒度理解 · 端到端 VLM 做不到
- rhythm: dense
- points:
  - **目标问题**：对比赛视频做**细粒度**理解——精确到每一脚传球成败、每一次抢断的施动者、每个球员整场跑了多少米；输出要可量化、可索引、可被自然语言精确提问；不是粗粒度问答（"这是什么运动？"）
  - **为什么不能直接用端到端 VLM**——5 个失败模式：
  - · 时间分辨率不够：商用 VLM 1-2 fps 抽帧，但传球/射门发生在 1/15 秒级 → 漏帧
  - · 空间分辨率不够：压到几百像素后，远景球（~20 px）和球衣号几乎消失 → 看不见
  - · 没有几何：知道"有个穿蓝衣的人"但不知道他在球场坐标 → 算不了跑动 / 越位 / 控球
  - · 没有稳定身份：每次推理独立，"射门那人"和"传球那人"是不是同一个无法回答
  - · token 经济：90 分钟 60 fps 视频 ~325k 帧，每问一次重看一遍不可持续
- notes: 这一页讲清"问题 + 为什么不能用现成方案"。强调"细粒度"——每一脚、每一次、每个球员，不是概括性印象。下页讲我们的方法。

---

## P03 — 我们的方法 · 先结构化，再让 LLM 写代码

- title: 解法两步走 · 看视频与理解视频解耦
- rhythm: dense
- points:
  - **第一步——把视频压成结构化数据**：多种技术配合（标定 + 检测 + 跟踪 + ReID + VLM 投票 + 规则状态机），每帧每球员/球的世界坐标、稳定身份、语义事件全部落到 JSON
  - **第二步——让 LLM 写代码分析这些数据**：不把 JSON 塞 prompt 让模型自己读，而是 LLM 调工具拿数 + Code Interpreter 沙盒里跑 Python 算/画
  - **题眼**：把"看视频"和"理解视频"**解耦**——前者由 5 个 stage 用专门技术做，后者交给 LLM。视频细粒度的难题（时间/空间/几何/身份）在结构化阶段就被解掉了，LLM 只面对干净的 JSON 和万能的 Python 逃生舱
  - **交付物**：① 结构化数据（坐标 / 身份 / 队伍） ② 语义事件（传球 / 射门 / 抢断 / 拦截） ③ 球员画像（跑动 / 速度 / 热力图 / 前后照片）
  - **路线图**：后续 9 页 = 怎么把视频压成结构化数据（5 个 stage） + 怎么让 LLM 在数据上深入理解（agent 层）
- notes: 这是整个 deck 的"题眼"页。讲完听众应该清楚"为什么这么做"——后面 9 张片子都是这个判断的具体展开。

---

## P04 — 流水线骨架 · 视频理解的 5 大难点

- title: 5 个 stage 顺序执行 · 每一道难关对应一个解
- rhythm: dense
- points:
  - 横向流程链：`field_registration → tracking → track_consolidation → event_detection → player_profile`
  - **5 个 stage 各自解决一道难点**：
  - · 业余视频内参未知 + 慢变焦（fx 6000→9000）+ **没有训练数据** → field_registration（profile + 4 帧 finetune）
  - · 球非常小（~20 px）+ 运动模糊 + 球员频繁遮挡 → tracking（4K 输入 + 世界坐标 gating + Kalman coast）
  - · 远景球衣 OCR 不可靠 + 同队 ReID 难区分 → track_consolidation（5 阶段贪心 + Claude image-list）
  - · 没有事件标注、纯规则可解释 → event_detection（possession 状态机派生）
  - · 球员级聚合需求 → player_profile
  - **设计原则**：JSON 契约（stage 间只读盘） / stage 内复杂 stage 间极简 / 世界坐标系 / fps 解耦
- notes: 这一页同时讲清"做什么"和"为什么这样切"——每个 stage 不是按业务功能切的，是按要解的难点切的。后面 5 张片子各展开一个。

---

## P05 — Stage 1 · Field Registration · Physical 后端

- title: 标定 · 固定内参 + 几何先验 + LM 优化
- rhythm: dense
- points:
  - 思路：**不让网络端到端预测相机参数**，给一组合理先验把问题约束成有限自由度 NLLS
  - 流程：profile 内参 → 手测 cam_pos 先验 → HRNet 检测 57 keypoints → PnP RANSAC → LM 精修
  - 自由度：rvec(3) + tvec(3) + fx(1) = **7-DOF**
  - **两遍解法**：Pass 1 软先验自由解 → Pass 2 锁定 cam_pos 中位数 → 4-DOF 重解
  - **拽回可行域**：fx 用硬 bounds（HFOV 角度表达，分辨率无关）；俯仰角用软 barrier（100 px/度）
  - **LightGlue/SuperPoint chain gap-fill**：远景帧 KP 太少时从邻帧 chain 过来；4K → 1080p 匹配速度 4×
- notes: 这一节是地基，所有"以米为单位"的下游都靠它精确到几个像素。

---

## P06 — ⭐ 4 帧标注 = 系统能跑

- title: 关键点 finetune — 4 帧标注就够了
- rhythm: breathing
- points:
  - 通用 PnLCalib HRNet 在自有视频上 **reproj_error 30+ px**（不可用）
  - 标几千张帧不现实——业余视频无标注，单人每帧 1-2 分钟
  - **4 帧手工标注** finetune 后 **reproj_error 5-10 px**（可用）
  - 反直觉的成功案例：固定机位视角分布**极窄**，4 帧 cover 整个分布
  - 经验：kids_soccer 7 帧、sunday_soccer 4 帧，验证 loss ~3e-5
  - 这个 trick **直接决定系统能不能跑**
- notes: 这是整个系统最关键的算法决策——其他都建立在 calibration 准确的基础上。重点强调"少标注 + 强物理先验"的思路。

---

## P07 — Stage 2 · Tracking · StrongSORT + PRTReID

- title: 多目标跟踪 · 4 阶段级联匹配 + part-based ReID
- rhythm: dense
- points:
  - 检测：YOLOv8 单 forward 同时输出人 + 球；保持 4K 原生分辨率（小球需要）
  - **4 阶段级联**：tentative-IoU → tentative-pitch → confirmed-IoU → confirmed-ReID
  - **关键创新**：在世界坐标系而不是像素空间用距离做 gating（同样 1m 在远近画面差几倍像素）
  - **Kalman coast**：遮挡时不立刻杀 track，让 KF 滑行几帧等 ReID 重连
  - **stationary killer**：广告牌/护栏不动 → 单独检查清掉
  - **PRTReID**（256-d, part-based）而不是 OSNet（512-d, global）：同队球员需要 part 级信号区分
- notes: 重点讲"世界坐标 gating"和"为什么 part-based"。这一页主要讲球员；下一页专门讲球。

---

## P08 — 球的检测与 3D 轨迹 · 物理模型分段拟合

- title: 球小、模糊、还要懂"高度"——三层处理
- rhythm: dense
- points:
  - **难点**：4K 上典型 ~47 px、远景 ~20 px；运动模糊把球拉成椭圆；纯像素信息不够判断"地滚 vs 高球"
  - **检测层**：YOLO class 32（sports ball）+ 低置信度阈值；可选**两阶段**——全图扫漏掉的帧用前后帧位置插值出小窗口 enlarge 再扫
  - **2D 跟踪层**：ByteTrack/BOTSORT 但 IoU 在小框上不稳 → 改用 **center-distance** 匹配
  - **3D 轨迹层**（关键 trick）：批处理两遍拟合
  - · Pass 1 ——**踢球瞬间分段**：基于像素加速度突变找 kick boundary，把整段轨迹切成独立 segment
  - · Pass 2 ——**每段判定地滚 vs 空中**：地滚段直接投影到 z=0 球场平面；空中段拟合带重力的运动模型 `P(t) = P₀ + V₀·t + ½·g·t²`，g=9.81 m/s²，前后段"落地"时刻作为锚点
  - **为什么物理模型**：避免"看起来高 = 实际高"的视角畸变误判（远处球 motion blur 也像在飞）；用物理常数把视觉问题约束回物理问题，**训练集无关**
- notes: 球这一节是整个 tracking 的高光算法点。重点讲"为什么不直接用 NN 回归 z 高度"——物理模型零训练数据、可解释、跨场景复用，远优于 data-driven 方案。

---

## P09 — Stage 3 · Track Consolidation · 5 阶段贪心 + Claude VLM

- title: 身份消歧 · 把 200+ 碎片 track 合成 22 个稳定身份
- rhythm: dense
- points:
  - 问题：tracker 每次遮挡 + 重出现都给新 id，10 分钟视频 22 球员产 200+ track
  - **5 阶段贪心**：
  - A. ReID-first 严格聚类（cosine ≥ 0.95 + **不能时间共现** 硬约束）
  - B. **Claude image-list 球衣号投票**（4 crops/call，置信度加权）
  - C. Team split（同聚类跨队 → 拆开）
  - D. Orphan absorb（松阈值合并孤儿）
  - E. 命名 `A-9` / `B-10` / `A-GK`
  - **为什么 image-list 不用 montage**：避免邻居穿透（bbox padding 把队友带进 cell）
  - **为什么 Claude 不用 Qwen/Gemini**：拒答率高 → 多 unknown 投票好过多噪声污染
- notes: 强调"反向 verification"——时间共现作为硬约束让 cosine 阈值可以放宽。

---

## P10 — Stage 4 · Event Detection · Possession 状态机派生

- title: 事件检测 · 不再用网络，纯规则状态机
- rhythm: dense
- points:
  - 流水线第一个**不再用神经网络**的 stage
  - 地基：**possession 状态机**——球离最近球员 ≤ 2 米连续若干帧 → possession span
  - 依赖图：`possession → {pass, shot, carry, defensive}`——所有事件从 possession spans **之间或之内**派生
  - 设计哲学：所有事件天然一致（不会出现"传球成功"但"接球者从未控球"）
  - **shot outcome 4 类**：Goal / Saved / Off_Target / Blocked
  - **射手归属**：用 **pre-kick possession** 而不是"球离脚瞬间最近"——防御者瞬间贴近会污染
- notes: 强调"用强先验代替模型"——possession 是非常强的物理约束，比训练 NN 性价比高。

---

## P11 — Agent 层 · tool-use + Code Interpreter

- title: 数据消费层 · LLM 自然语言问比赛
- rhythm: dense
- points:
  - 核心选择：**tool-use 而不是 RAG**——精确数值聚合 + token 经济
  - 5 个固定工具：list_events / get_player_stats / get_team_stats / get_frame_snapshot / **run_python**
  - **run_python = 逃生舱**：在 AgentCore Code Interpreter 沙盒跑任意 Python，画 matplotlib 图返回
  - 覆盖固定工具到不了的开放问题（"画两队跑动距离对比柱状图"）
  - 类似 ChatGPT Advanced Data Analysis，但数据是**这场比赛的**
  - 可选：整个 agent 跑在 AgentCore Runtime 容器（多 session 隔离 + 共享 JSON）
- notes: 强调 Code Interpreter 的"逃生舱"角色——预定义 API 永远列不全，给 LLM 写 Python 是最通用的解。

---

## P12 — 关键 trade-off 表 · 一页通览

- title: 我们选了什么 · 为什么
- rhythm: dense
- points:
  - 标定：**physical** 后端（固定内参 + 7-DOF LM）—— 业余视频内参未知但机型固定
  - 训练数据：**4 帧** finetune —— 固定机位视角分布极窄，少量样本就够
  - ReID：**PRTReID part-based** —— 同队球员区分需要 part 级信号
  - 球衣识别：**Claude image-list** —— 拒答率高 + 邻居穿透避免
  - 身份合并：**5 阶段贪心 + jersey VLM** —— ReID 单独不够，jersey 是绝对信号
  - 事件抽取：**规则状态机** —— 数据稀缺 + possession 是强先验，比 NN 性价比高
  - Chat：**tool-use + run_python** —— 精确聚合 + 任意可视化 + token 经济
- notes: 收尾页。每行只读一两秒——重点在让听众感受到"每个决策都是基于具体场景约束的，不是默认选了最 fancy 的"。问答开始。

---
