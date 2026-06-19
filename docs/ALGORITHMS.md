# GoalInsight 算法概念（sunday_soccer 配置版）

> 这份文档**只讲技术概念和算法思路**——不涉及 CLI 用法、配置 key 名、API 路径或具体文件结构。
> 内容范围严格对照 `configs/sunday_soccer.yaml` 实际启用的算法路径，没启用的 backend / stage 一律不写。
>
> 阅读路径：先 §1（问题定义）→ §2（流水线骨架）→ 之后任挑感兴趣的 §3-§7 看。每节关键术语会带英文。

---

## 1. 问题定义

输入：**一段 4K、60 fps、固定机位的足球比赛视频**（业余比赛常见配置：phone/GoPro 类相机架在中线 × 边线交点附近、约 5 m 高的非专业三脚架机位，整段比赛单机位俯拍）。

希望产出：

1. **结构化的比赛数据**：每一帧每个球员/球的世界坐标、稳定身份、队伍归属
2. **语义事件**：传球（成功/失败）、射门（进球/扑救/偏门/封堵）、带球、抢断、拦截
3. **每个球员的画像**：跑动距离、速度、热力图、前/后照片

核心难点分两部分。

### A. 视频理解（§3-§7 服务这一组难点）

把视频压成结构化数据的过程里几个绕不过的难点：

- **业余比赛**：相机内参未知；机位偏低；广角镜头会缓慢变焦（同一段视频里 fx 从 6000 漂到 9000）
- **没有训练数据**：业余视频没有现成标注，每次接入新机型都要从零开始；几千张帧手工标 keypoint 是几百小时工作量，业务上不可行——所有训练相关的设计都必须能用**极小样本**（个位数标注帧）跑得动
- **球非常小**：典型 4K 上 ~47 px，常 ~20 px；远景 + 运动模糊导致检测器置信度低
- **球员频繁互相遮挡**，IoU 跟踪反复断裂
- **球衣号识别**在远景模糊背景下 OCR 不可靠，但又是恢复稳定身份的关键
- **同队球员的 ReID 区分难**——球衣相同、体型接近，外观特征相似度天然高

### B. Agent 数据消费（§8 服务这一组难点）

视频被压成 JSON 之后，怎么让用户拿到答案是另一类难点：

- **固定工具覆盖不到的开放问题**——"两队跑动距离对比柱状图"、"球员 7 在禁区内接到的传球占比"、"画一下整场控球时长的时间轴"——预定义 API 永远列不全
- **LLM 不擅长精确数值聚合**——把 events.json 直接塞 prompt 让模型自己算控球率，会算错；需要把"取数"和"叙述"分开
- **token 经济**——一场 90 分钟比赛的 events.json 能有几万行；整体 RAG 上下文成本无法持续

---

## 2. 流水线骨架

把"视频 → 结构化数据"拆成**五个独立 stage**，每个 stage 是纯函数：上一个的产物 → 下一个的输入。

```
field_registration   — 相机/球场标定
       │
tracking             — 球员+球检测+短期跟踪
       │
track_consolidation  — 把碎片化 track 合并成稳定球员身份
       │
event_detection      — 在结构化数据上跑事件状态机
       │
player_profile       — 每个球员的画像
```

> **设计原则**：
>
> - **JSON 契约**：每个 stage 通过盘上 JSON/pkl 文件交流，不是 Python 对象。任意 stage 可单跑、可重跑、可换实现
> - **stage 内 → 复杂 / stage 间 → 极简**：stage 内部算法可以很贵（DL 推理、PnP RANSAC、LM 优化、LLM 调用），stage 之间只是文件读写
> - **中后期都跑在世界坐标系**：tracking 之后再没有"像素距离"出现；所有距离/速度/拦截判断都用米
> - **不同 stage 可以用不同 fps**：calibration 在 10 fps 算（相机变化慢），tracking 在 30 fps 算（同帧多球员区分需要更密的样本）。两者通过统一的"采样 stride"和帧索引保持一致

---

## 3. 球场标定（Field Registration）— Physical 后端

**目标**：找出每一帧相机的外参 + 内参，使得视频像素 ↔ 真实球场坐标的映射存在且精确到几个像素以内。所有以米为单位的下游判断都依赖它。

### 3.1 整体策略：固定内参 profile + 几何先验 + LM 优化

`physical` 后端的核心思路是：**不要让网络自己端到端预测相机参数**，而是给一组合理先验，把求解问题约束成有限自由度的非线性最小二乘。具体做法：

1. **从 profile 取一个粗糙的相机内参**（generic_4k：`fx ≈ 3500`、cx/cy ≈ 主点中心、零畸变）
2. **粗略给一个相机位置先验**（手测 / 模糊估计的机位坐标，例如中线 × 边线交点附近、约 5 m 高）
3. **用 HRNet 检测每帧的球场关键点**（57 个 PnLCalib 风格 keypoints，覆盖角旗、点球点、罚球区角等）
4. **PnP RANSAC** 求初始外参 + 焦距
5. **LM 优化**进一步精修：自由度 = rvec(3) + tvec(3) + fx(1) = **7-DOF**

### 3.2 两遍 (two-pass) 解法

相机位置在整段视频里实际不变（机位是三脚架），但 LM 单帧解的相机位置会有抖动。处理：

- **Pass 1**：每帧独立解，相机位置作为**软先验**而不是硬约束（LM 可以调，但偏离先验会受到惩罚）
- **Pass 2**：取 Pass 1 里**残差小的帧**的相机位置中位数 → 全局**硬锁定**这个位置 → 重新逐帧解，自由度降到 **4-DOF**（rvec + fx）

效果：第二遍把抖动消掉了，且因为自由度更小，关键点稀疏（远景帧只有 3-4 个可见 keypoint）的难帧也能稳定收敛。

### 3.3 把 LM 拽回合理可行域

LM 在无约束情况下偶尔会跑到物理上无意义的解（焦距过短/过长、相机几乎水平、俯视过陡）。两类参数用两种**强度不同**的约束拽回来：

- **焦距 fx**：硬 bounds。镜头不是定焦——同一段视频里 fx 从 ~6000 漂到 ~9000——bounds 必须能覆盖整个变焦范围。bounds 用 **HFOV 角度**而不是焦距像素表达（"水平视场 18°-51°"），换算到当前分辨率的焦距像素自动展开；这样同一份配置在 4K 和 1080p 都成立。
- **俯仰角**：**软 barrier**。合理俯仰角范围 [2°, 15°]，超出按"100 px / 度违反"罚到目标函数里。这是个 fixed-length penalty，不会把可行域硬切掉，但能引导 LM 跳出病态解（病态解附近残差表面平坦，硬约束会让 LM 卡在边界；软 barrier 会让它"回流"到合理区间）。

> 经验法则：**强物理先验 → 硬约束**（焦距由相机硬件决定，不会突变）；**弱物理先验 → 软 barrier**（俯仰角依赖架机方式，先验粗糙）。同样的 idea 也用在 cam_pos 上——见 §9.4。

### 3.4 关键点 finetune — **4 帧标注就够了**

通用 PnLCalib HRNet 在自有视频上效果不够（球场视角分布偏离 SoccerNet 训练集）。但**重新标几千张帧不现实**——业余视频没有现成标注，单人每帧约 1-2 分钟手工点 keypoints，几千张是数百小时工作量。

GoalInsight 的 finetune 闭环走**极小样本**路线：

- 在 web UI 上对**精选的 4 帧**手工点 keypoints（覆盖整段视频典型视角：开场、中场、远端 zoom-in、近端宽角各取一帧）
- 跑一次 finetune（几十分钟，单 GPU），得到一份针对**这台相机这种机位**的专用 keypoint head
- 替换通用 head；推理阈值可以同步调低（finetune 后置信度分布右移）

这是一个非常**反直觉的成功案例**——HRNet 通常需要数千张标注才能 finetune 到位，但**视角分布极窄**的场景例外：固定机位单镜头的视频里所有帧都是同一个相机轨迹的小扰动，4 帧已经覆盖了这个分布，模型只需要"稍微调一下空间响应"即可。kids_soccer 用过 7 帧训练集，sunday_soccer 用了 4 帧，验证集 loss 降到 ~3e-5 量级。

这个 trick **直接决定了系统能不能跑**：通用模型在这种业余视频上 reproj_error 经常 30+ px（不可用），finetune 后能稳定在 5-10 px（可用）。

> **注意**：line 检测 head 在这场配置里**不用**——通用 line 模型在这种镜头下噪声大，反而会污染 LM 的目标函数。所有线段相关的残差权重都置零。

### 3.5 SIFT/LightGlue chain gap-fill

逐帧独立标定有两类失败：

- **关键点太少**：远景帧或者部分被裁掉的视角，HRNet 只检出 3-4 个 keypoint，PnP RANSAC 不收敛
- **重投影误差大**：keypoint 检测有外点，LM 收敛到一个 reproj_error 远超阈值的"坏解"

补丁：**把"好帧"的相机位姿沿时间轴 chain 推到坏帧上**：

1. 每若干帧（如每 3 帧）选一帧做 anchor 候选
2. 在 anchor 间 / anchor 到目标帧之间用 **LightGlue + SuperPoint** 做特征匹配
3. 用匹配点估计相邻帧之间的 homography
4. 用 chain 起来的 homography 把已知位姿的 anchor → 未知位姿的目标帧
5. 对 4K 输入先 downscale 到 1920 长边再匹配（特征匹配在 1080p 上够稠密，速度快 4×；坐标再 rescale 回 4K 给下游 PnP）

LightGlue 比经典 SIFT+FLANN 快 ~20×，inlier quality 相当甚至更好。退化时回退到 SIFT。

### 3.6 时间一致性

剩下的小尺度抖动用**滑窗平滑** + drift 检测处理：相邻帧位姿变化超出"机位三脚架可达运动"上限就标记为 outlier 并用前后帧线性插值替代。

---

## 4. 多目标跟踪（Tracking）— StrongSORT + PRTReID

**目标**：每一帧检测出谁是球员、谁是球，并把同一个目标在时间上串起来——给每个目标一个**短期 track_id**。

### 4.1 检测：YOLOv8 单 forward pass

YOLOv8 同时检测人和球两类，**统一推理**（unified detection）。两类共用一次 forward，但用不同后处理：

- 球的置信度阈值更低（小目标本身置信度就低）
- 球员的 NMS IoU 阈值放宽到 0.55（默认 0.45 在两人紧贴重叠时会把一个抑制掉，导致跟丢）

**输入分辨率**：保持 4K **原生分辨率**（长边 round 到 32 倍数）。降到 1920 会把 47 px 的球降到 ~23 px，YOLO 在小目标上的置信度会塌掉。代价：4K 推理时间 ~1920 的 4×。

### 4.2 短期数据关联：四阶段级联匹配

把检测和已有 track 关联起来是这个 stage 最难的部分。直觉：用 **Kalman 滤波**对每个 track 预测下一帧位置，然后想办法把当前帧的检测和这些预测配对。

GoalInsight 跑一个 **StrongSORT 风格的四阶段级联匹配器**：

1. **tentative-IoU**：刚刚出生（< 几帧）的 track 和近邻检测做 IoU 匹配
2. **tentative-pitch**：tentative track 在**世界坐标系**用米作距离单位匹配（处理像素瞬间跳变但实际位置接近的情况）
3. **confirmed-IoU**：已确认（连续命中阈值次数）的 track 走 IoU
4. **confirmed-ReID**：上面没匹上的，最后用**外观特征 ReID** cosine 距离匹配

> **关键创新**：在世界坐标系而不是像素空间用距离做 gating。同样的 1m 实际距离在画面前景和后景对应的像素距离差几倍——以米为门限可以保持几何一致性。

阈值 (`pitch_gate_m`、`reid_pitch_max_m` 等) 都按 fps 自动归一化：sunday_soccer 在 30 fps 下跑，所有阈值自动按 1/3 倍尺寸 scale，物理意义保持一致。

### 4.3 处理遮挡和 Kalman 滑行

- **Kalman coast**：track 被遮挡时不立刻杀死，让 Kalman 先"滑行"几帧；如果在窗口内有 ReID 匹配的检测，就成功续上
- **stationary-track killer**：广告板、看台、护栏经常被检测成"人"且永远不动，单独拉一个静止 track 检查清掉

### 4.4 ReID 特征：PRTReID

每帧每个 bbox 抽一个 **256 维 part-based ReID 特征向量**（PRTReID，HRNet32 backbone，在 SoccerNet 上 finetune）。

为什么不用 OSNet：**同队球员**球衣相同、体型接近，OSNet 的 global feature 会把同队不同人的相似度顶得很高（cosine ≈ 0.9 也可能是不同人）。Part-based 模型把 head / torso / legs 分开提取，对同队区分更敏锐——头发颜色、面部、腿型这些 part-level 信号被独立保留下来。

ReID 特征同时被两个层使用：

1. **当前 stage**：confirmed-ReID 匹配的 cosine 距离
2. **下一个 stage（consolidation）**：把跨长时间断裂的 track 重新粘起来

### 4.5 队伍分类

球员检测出来后还要分队。这场用的是 **KMeans on jersey color histogram**：在每个 bbox 提取主色（裤子+上衣），全场聚类成两类。

加一个轻微的 `position_weight=0.1` 把球员位置作为辅助维度参与聚类——纯颜色聚类在阴影帧 / 红黄相邻颜色上偶尔会翻车，但**同队球员从不会全场都挤在对方半场**，位置维度做软兜底。

> 备选 backend：tracklet voting（每个 track 内部做颜色投票）——同 track 帧间颜色一致性强，理论上更稳定，但 sunday_soccer 这场 KMeans 已经够用，没换。

### 4.6 球的特殊处理

**球非常小**，用普通跟踪器有几个问题：

- IoU 在小框上不稳：用 **center-distance** 替代 IoU
- 漏检比误检常见：除了主检测通道，可启用**两阶段**——第一阶段全图扫，第二阶段在前后帧位置插值出来的小窗口里 enlarge 后再扫

---

## 5. 身份消歧（Track Consolidation）— 五阶段贪心 + Claude VLM 投票

跟踪器吐出来的 `track_id` 在长时间尺度上**完全不可信**：每次遮挡 + 重出现都是新 id。10 分钟视频 22 个球员可以产出 200+ 个 track。下游如果直接用 `track_id`，会出现"同一个球员的所有事件分散在 5 个不同身份下"的灾难。

**目标**：把碎片 `track_id` 聚类成**整场恒定的 player_id**（"A-9"、"B-10"、"A-GK" 这种）。

### 5.1 五阶段贪心管线

整体策略：**先用最强的信号做最严格的合并**，再逐层放松约束捡剩下的。

1. **Stage A — ReID-first 聚类**：在 cosine 相似度高于严格阈值（如 0.95）**且时间不重叠**（不能两个同时出现的 track 算同一人）的前提下，做单链接聚类
2. **Stage B — 球衣号 VLM 投票**：对每个聚类采样若干 crops 喂给 Claude（细节见 §5.2），做**置信度加权**投票得出该聚类的球衣号 + role
3. **Stage C — Team split**：如果某个聚类里同时有蓝队和红队的 track，强制拆开（说明 Stage A 的 ReID 失误了）
4. **Stage D — Orphan absorb**：剩下的孤立 track 在更松的阈值下合并到已有聚类
5. **Stage E — 命名**：得出 `<队伍前缀>-<球衣号>` 形式（守门员 `A-GK`，球衣号未知 `B-unk-01`）

最短观测时间过滤：观测帧数 < 1 秒的 track 在 Stage A 之前直接丢掉（YOLO 一两帧的假阳性）。

### 5.2 球衣号识别：Claude image-list 模式

这场配置用 **Claude Opus** 做球衣号识别，不是 OCR：

- 每个 track 按 stride=8 采样 crops（约 22 个 crops/track）
- 4 个 crops 打包成**一次** LLM 调用（image-list 模式：4 张独立图片，**不是**拼成 montage）
- 模型对每张独立返回一个 `(role, team, jersey_number, confidence)`
- 同一 track 的多次调用结果做 **置信度加权投票**

为什么 image-list 而不是 montage：

- montage（4 个 160×160 cell 拼成一张图）有**邻居穿透**问题——bbox padding 经常把队友的一部分带进 cell，模型容易混淆
- image-list 每张独立的 crop，邻居最多以 bbox 边缘的小切片形式出现，目标球员在中心更明确

为什么 Claude 而不是 Qwen-VL / Gemini：实测在远景球衣上 Claude 比另两家**更愿意拒绝识别**（输出 "unknown"），而后两家更愿意瞎猜单字数字。在投票场景下，**多 unknown** 比 **多噪声** 更好——投票自然忽略 unknown，但被噪声污染就回不来了。

### 5.3 关键技术点：ReID 不能时间共现

如果两个 track 在某帧同时存在，**它们一定不是同一个人**。这条硬约束让 Stage A 的 cosine 阈值可以放宽，因为时间冲突直接否决——是一种"反向 verification"。

---

## 6. 事件检测（Event Detection）— Possession 状态机派生

**目标**：在结构化轨迹上提取语义事件——传球、射门、进球、带球、抢断、拦截。这是流水线中**第一个不再用神经网络**的 stage：纯规则状态机。

### 6.1 一切从 Possession 开始

把"控球"做成**地基**：

- 状态机：当前控球的 `player_id`，连续多少帧
- 触发：球离最近球员距离 ≤ 阈值（约 2 米）连续若干帧
- 释放：球速突增（被踢） / 球距离当前控球球员超出阈值

得到一组 **possession spans**：每个 span 是一个三元组 `(player_id, start_frame, end_frame)`。

> **设计哲学**：所有其他事件都是 possession spans **之间或之内**的事件，不是从原始轨迹直接生成。这让事件之间天然一致——例如不会出现"传球成功"但"接球者从未控球"的矛盾。

### 6.2 五个事件 detector

**依赖图**：

```
possession ─┬→ pass
            ├→ shot
            ├→ carry
            └→ defensive (tackle, interception)
```

- **pass**：相邻 possession spans 之间插了一段球速突增。如果接球者属于同队 → successful；属于对方或没人接到 → failed。**特殊处理**：one-touch（直接踢出，没有 carry）也算 pass，不要漏
- **shot**：可识别为射门的踢球——球速达到阈值 + 朝球门方向。outcome 分四类：
  - **Goal**：球进入球门坐标范围
  - **Saved**：守门员一定阈值内接触球
  - **Off_Target**：飞出球门外
  - **Blocked**：撞到守门员之外的防守球员
- **carry**：在 possession span 内部，控球者前向位移 ≥ 阈值
- **defensive**：
  - **tackle**：possession 易主 + 期间球速被打偏（角度突变）
  - **interception**：上一个 pass 是 failed + 接到球的是对方

### 6.3 一个细节：射手归属

直觉是"球离脚瞬间最近的球员就是射手"。但实测中**防守球员**经常在射门一瞬间贴上来——按这个规则会把进球算到防守球员头上。

GoalInsight 用 **pre-kick possession**：射门事件的 `player_id` 等于"这一脚之前最后一段 possession span 的控球人"。语义上 = "谁带着球射的"，不会受瞬间距离扰动。

---

## 7. 球员画像（Player Profile）

每个 `player_id` 计算一组**整场聚合指标**和一组**视觉 artifact**。

### 7.1 跑动距离（在世界坐标系积分）

不是简单把每帧 `pitch_position` 距离累加——会被定位噪声 + 跟踪 jitter 干掉。处理：

1. **gap-aware**：相邻两帧实际间隔 > 1 秒（断 track）就不连续
2. **speed clipping**：人最快冲刺约 10 m/s，超过 12 m/s 必然是定位错（瞬移），跳过这一段

最终的距离比"逐帧累加"少一截，但是符合实际跑动。

### 7.2 跑动热力图

简单 2D histogram on pitch coords，直接可视化。

### 7.3 前/后照片选择

每个球员选一张**正面**（看到胸前队徽）+ 一张**背面**（看到球衣号）的 crop：

- 用 §5.2 Claude 调用里已经返回的"crops 是否有数字可见"标签做信号
- "看不到数字" → 正面候选；"清晰看到数字" → 背面候选
- 在每类候选里挑 **sharpness score 最高** 的一张

---

## 8. Agent 层（Agent over Match Data）

经过前面 5 个 stage，比赛已经被压缩成**一组 JSON 文件**（events、tracks、ball、team_assignments、players_profile 等），可以被任何上层应用消费。GoalInsight 自带的上层是一个 **LLM chat agent**——把"自然语言问比赛"接到结构化数据上。

### 8.1 Tool-use 而不是 RAG

不是把整份 JSON 扔进 prompt 让 LLM 自己读——量大且 LLM 不擅长精确数值聚合。改用 **tool calling**：

> 用户："team A 第二节有几次射门？"
>
> Agent 调 `list_events(types=["shot"], team_id="team_A", time_range_s=[1500, 3000])` → 拿到结构化结果 → LLM 总结成自然语言。

固定五个工具：

| 工具 | 角色 |
|---|---|
| `list_events` | 通用事件过滤器（type / team / player / 时间） |
| `get_player_stats` | 单球员聚合数值（距离、速度、触球、传球、射门、进球） |
| `get_team_stats` | 单队伍聚合数值（控球率、传球成功率、射门、抢断、拦截） |
| `get_frame_snapshot` | "这一刻屏幕上有谁、球在哪" |
| `run_python` | **逃生舱**：在 Code Interpreter 沙盒里跑任意 Python，访问完整 JSON、画 matplotlib 图返回 |

### 8.2 为什么 run_python 是点睛之笔

固定工具覆盖不到的用户问题（如"画一下两队跑动距离对比柱状图"、"算下球员 7 在禁区内接到的传球占比"）通过 `run_python` 处理：

- LLM 写一段 Python（基于工具返回的数据形态）
- 跑在 **AgentCore Code Interpreter** 沙盒：隔离环境、有 matplotlib / numpy 可用
- 产出的图片（base64 PNG）回流成 markdown 内嵌图，直接在 chat 里显示

效果上类似 ChatGPT Advanced Data Analysis，但数据是**这场比赛的**——而且用户完全没意识到背后是动态生成 Python。

### 8.3 可选：Agent 整体远程化

把整个 chat agent（不止 Code Interpreter sandbox）放到 AWS 托管的 **AgentCore Runtime** 容器里，再用 FastAPI 透传 SSE。容器按 session 常驻：第一次访问从 S3 拉一份 run JSON，之后整个 MicroVM 生命周期内复用，省掉每次 cold-start 解析 JSON 的开销。

适合多用户共享同一份数据、想用 AWS 提供的隔离环境扩缩容的场景。

---

## 9. 一些贯穿所有 stage 的设计哲学

### 8.1 数据契约 > 接口耦合

每个 stage 写的是**普通 JSON / pkl**，下一个 stage 读盘。**不是** Python 类传引用、不是 protobuf、不是 ORM。理由：

- **任何 stage 可单独 rerun**，因为它只看上一个 stage 的盘面快照
- **手工 patch / 离线分析无门槛**——任何文本编辑器都能改 JSON
- **缓存复用零成本**：昂贵步骤（球衣 VLM 调用，几十次 Claude API）独立缓存，重跑下游不用重花钱

### 8.2 中后期都用世界坐标，不用像素

凡是涉及"远 / 近 / 速度 / 距离"的判断，全部在世界坐标系下用米。像素只是采集端格式，进入 tracking 之后就被换算掉了。

好处：算法对**视频分辨率、相机焦距、机位**都不敏感。同一套阈值（"控球距离 ≤ 2m"）在 1080p 和 4K 上行为一致。

### 8.3 fps 归一化

不同 stage 可以跑在不同有效 fps 上（calibration 10 fps、tracking 30 fps）。所有时间相关阈值（max_age、pitch_gate、min_track_seconds 等）都用**秒**表达，stage 启动时根据当前 effective_fps 换算成帧。

好处：换镜头帧率不需要改算法配置——同一份阈值在 30 fps 和 60 fps 上行为一致。

### 8.4 软先验 + 硬 barrier，少用硬约束

field_registration 里相机位置、俯仰角都用 **soft prior + bounds**：模型可以违反先验，但代价递增。比起硬锁，这种处理方式更鲁棒：

- 先验粗糙时（手测 cam_pos 偏几米），LM 仍能找到真解
- 先验靠谱时（Pass 2 锁定中位 cam_pos），先验权重变成约束
- LM 跑飞时（俯仰角去到无意义值），barrier 把它拽回来而不是直接报错

---

## 10. 关键 trade-off 汇总

| 决策 | 选了什么 | 备选 | 为什么 |
|---|---|---|---|
| 标定 backend | physical (固定内参 + 7-DOF LM) | 端到端 NN | 业余视频内参未知但**机型固定**，profile 给定足够先验；可解释、可微调 |
| KP 检测训练 | **4 帧**手工标注 finetune | 通用预训练 / 数千张标注 | 固定机位视角分布极窄，少量样本就能 cover；通用模型 reproj_error 30+ px 不可用 → finetune 后 5-10 px |
| 标定线段约束 | 关闭（line_weight=0） | 启用 line residual | 该镜头下 line 模型噪声 > 信号，污染 LM |
| 标定时间一致性 | LightGlue chain gap-fill | 单帧重试 / 简单插值 | 远景帧 KP 太少 PnP 不收敛，必须从邻帧 chain 过来 |
| 跟踪 gating | 世界坐标米 | 像素 | 跨视角/分辨率一致 |
| ReID backbone | PRTReID (part-based, 256-d) | OSNet (global, 512-d) | 同队球员区分需要 part-level 信号 |
| 球衣识别 | Claude Opus image-list | Qwen / Gemini / OCR / Claude montage | 实测拒答率 + 准确率最高；image-list 避免 montage 邻居穿透 |
| 身份合并 | ReID + jersey VLM 5 阶段 | 单 ReID | ReID 在长时间断裂上不可靠，jersey 是绝对信号 |
| 事件抽取 | 规则状态机 | 端到端 NN | 数据稀缺；规则可解释、可调阈值；possession 是非常强的先验 |
| 射手归属 | pre-kick possession | nearest-at-kick | 防御者瞬间贴近会污染 |
| YOLO 输入分辨率 | native 4K | 1920 长边 | 球太小，下采样会塌掉小目标置信度，代价是 4× 推理时间 |
| Calibration fps | 10 fps（独立） | 跟 tracking 同 30 fps | 相机变化慢，省 3× 时间 |
| Chat 数据消费 | tool-use + run_python | RAG（把 JSON 塞 prompt） | 精确数值聚合 + 任意可视化 + token 经济 |

---

*本文档故意保持语言密集、不深入实现细节。每节涉及的具体类、文件、字段名见 `TECHNICAL_GUIDE.md` §4-§5。如发现描述与代码当前行为不一致以代码为准。*
