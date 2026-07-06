# GoalInsight Architecture

```mermaid
%%{init: {'theme': 'dark', 'flowchart': {'nodeSpacing': 16, 'rankSpacing': 28, 'padding': 12}}}%%
flowchart LR
    VIDEO[/"Input Video"/]

    subgraph S1["S1: Field Registration"]
        direction TB
        S1_KP["KeypointDetector · HRNet"]
        S1_BE["PnLCalib / BroadTrack / Physical / NBJW / Homography"]
        S1_KP --> S1_BE
    end

    subgraph S2["S2: Tracking"]
        direction TB
        S2_DET["UnifiedDetector · YOLO"]
        S2_PT["StrongSORT / BoT-SORT + ReID"]
        S2_BT["BallTracker → Filter → Traj3D"]
        S2_TA["TeamClassifier"]
        S2_DET --> S2_PT --> S2_TA
        S2_DET --> S2_BT
    end

    subgraph S3["S3: Track Consolidation"]
        direction TB
        S3_RC["ReID-first clustering"]
        S3_JV["Jersey vote (Claude / Gemini / Qwen)"]
        S3_NM["Team split → orphan absorb → naming (A-9, B-10, ...)"]
        S3_RC --> S3_JV --> S3_NM
    end

    subgraph S4["S4: Event Detection"]
        direction TB
        S4_OR["EventOrchestrator"]
        S4_PS["Possession"]
        S4_EV["Pass · Shot · Carry · Defensive"]
        S4_OR --> S4_PS --> S4_EV
    end

    subgraph S5["S5: Player Profile"]
        direction TB
        S5_CR["Front/back crops + heatmap + distance"]
        S5_SP["Spotlight follow-cam (opt-in)"]
        S5_CR --> S5_SP
    end

    subgraph S6["S6: Highlights"]
        direction TB
        S6_DT["EventDetector"]
        S6_AN["SceneAnalyzer"]
        S6_CP["ClipComposer"]
        S6_DT --> S6_AN --> S6_CP
    end

    subgraph S7["S7: Annotated Video"]
        direction TB
        S7_HUD["Full-match HUD render (boxes + IDs + ball trail + events)"]
    end

    subgraph S8["S8: Enhancement (inline)"]
        direction TB
        S8_UP["Upscale · Real-ESRGAN / Real-CUGAN / Anime4K"]
        S8_FI["Interpolate · RIFE"]
        S8_UP --> S8_FI
    end

    VIDEO --> S1 --> S2 --> S3 --> S4
    S3 --> S5
    S4 --> S6
    S3 --> S7
    S6 -.upscale + slow-mo replay.-> S8
    S7 -.optional upscale.-> S8
    S5 --> OUTPUT[/"Output"/]
    S6 --> OUTPUT
    S7 --> OUTPUT

    classDef stage fill:#2563eb,stroke:#1e40af,color:#fff,font-weight:bold
    classDef io fill:#059669,stroke:#047857,color:#fff
    class S1,S2,S3,S4,S5,S6,S7,S8 stage
    class VIDEO,OUTPUT io
```

The pipeline runs as a config-driven chain of stages registered in
`goalinsight/pipeline/_adapters.py` via `@register_stage`. The order is
controlled by `pipeline.stages` in the YAML config. `track_consolidation`
must precede `event_detection` so events carry stable `player_id`s
(`A-7`) instead of raw track integers. `player_profile`, `highlights`,
and `annotated_video` are independent leaves that all consume the
consolidated tracks. Video enhancement (`video2x` upscaling + RIFE
interpolation) is invoked inline from the highlights composer and the
annotated-video renderer, not as a standalone stage.

A web app (`goalinsight/web/`) sits on top of the run output: FastAPI
viewer + Bedrock-backed chat with five tools (`list_events`,
`get_player_stats`, `get_team_stats`, `get_frame_snapshot`, `run_python`
in an AgentCore Code Interpreter sandbox). The chat agent can optionally
run inside an AgentCore Runtime container instead of in-process — see
[`deploy/agentcore_runtime/README.md`](deploy/agentcore_runtime/README.md).

For the rendered AWS-architecture views (chat path, local pipeline,
remote SageMaker pipeline) see
[`docs/architecture/`](docs/architecture/README.md).
