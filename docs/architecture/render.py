"""Render the GoalInsight architecture diagrams.

Outputs four PNGs into the same directory as this script:
  - chat_architecture.png         — online viewer (Bedrock + AgentCore)
  - chat_architecture_runtime.png — chat hosted on AgentCore Runtime
  - pipeline_local.png            — offline pipeline default (local stages)
  - pipeline_remote.png           — offline pipeline with --remote-stages

Run:  python docs/architecture/render.py
Deps: pip install diagrams ; apt install graphviz
"""

from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECR
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Bedrock, Sagemaker
from diagrams.aws.security import IAM
from diagrams.aws.storage import S3
from diagrams.generic.storage import Storage
from diagrams.onprem.client import Users
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python

OUT_DIR = Path(__file__).parent
GRAPH_ATTR = {
    "fontsize": "20",
    "labelloc": "t",
    "pad": "0.5",
    "splines": "spline",
    "rankdir": "LR",
    "nodesep": "0.6",
    "ranksep": "1.2",
}


# ---------------------------------------------------------------------------
# Diagram 1: chat / viewer
# ---------------------------------------------------------------------------

def render_chat() -> None:
    # Top-down for chat: star topology around FastAPI looks more
    # natural with the user/UI on top and AWS services below.
    attrs = dict(GRAPH_ATTR)
    attrs["rankdir"] = "TB"
    attrs["ranksep"] = "1.0"
    with Diagram(
        "GoalInsight chat — online (Bedrock + AgentCore)",
        filename=str(OUT_DIR / "chat_architecture"),
        outformat="png",
        show=False,
        graph_attr=attrs,
    ):
        user = Users("Browser")

        with Cluster("Local viewer process"):
            api = Fastapi("FastAPI\n/api/chat/stream")

            with Cluster("ChatEngine"):
                ctx = Python("MatchContext\n(events/tracks/...)")
                tools = Python("4 query tools\nlist_events,\nget_player_stats,\nget_team_stats,\nget_frame_snapshot")
                runpy = Python("run_python tool\n(CodeSandbox)")

            artifacts = Storage("chat_artifacts/\n(per-run PNGs)")
            data_files = Storage("output/<run>/\nevents.json,\ntracks.json,\nball_tracks.json")

        with Cluster("AWS"):
            bedrock = Bedrock("Bedrock Runtime\nClaude Opus 4.7")
            sandbox = Bedrock("AgentCore\nCode Interpreter\n(aws.codeinterpreter.v1)")

        # User flow
        user >> Edge(label="HTML / SSE") >> api
        api >> Edge(style="dashed", label="serve PNG\n/chat_artifacts/*") >> user

        # Chat loop
        api >> Edge(label="invoke_model_with_response_stream\n(tool_use loop)") >> bedrock
        bedrock >> Edge(label="text deltas\n+ tool_use blocks", style="dashed") >> api

        # Tool dispatch
        api >> tools
        tools >> Edge(label="read") >> ctx
        ctx >> Edge(style="dotted") >> data_files

        api >> runpy
        runpy >> Edge(label="writeFiles (data.json once)\nexecuteCode\nreadFiles (PNGs)") >> sandbox
        sandbox >> Edge(label="PNG bytes", style="dashed") >> runpy
        runpy >> Edge(label="save") >> artifacts


# ---------------------------------------------------------------------------
# Diagram 2: offline pipeline — fully local (default)
# ---------------------------------------------------------------------------

def render_pipeline_local() -> None:
    with Diagram(
        "GoalInsight pipeline — offline (default: all local)",
        filename=str(OUT_DIR / "pipeline_local"),
        outformat="png",
        show=False,
        graph_attr=GRAPH_ATTR,
    ):
        cli = Python("goalinsight CLI")
        video = Storage("data/raw_videos/\n*.mp4")

        with Cluster("Local GPU host"):
            with Cluster("goalinsight Pipeline"):
                fr = Python("field_registration\n(HRNet kp+line,\nLM solver)")
                trk = Python("tracking\n(YOLOv8x + OSNet\n+ ByteTrack)")
                evt = Python("event_detection\n(rule-based)")
                tc = Python("track_consolidation\n(jersey VLM + ReID)")
                hl = Python("highlights\n(crop + RIFE)")
                av = Python("annotated_video")

            out = Storage("output/<run>/\n<stage>/...")

        cli >> video >> fr
        fr >> trk >> evt >> tc >> hl >> av
        for stage in (fr, trk, evt, tc, hl, av):
            stage >> Edge(style="dotted", label="write") >> out


# ---------------------------------------------------------------------------
# Diagram 3: offline pipeline — --remote-stages field_registration,tracking
# ---------------------------------------------------------------------------

def render_pipeline_remote() -> None:
    # Top-down works better than LR for this graph because the dataflow
    # is conceptually local→AWS→local; vertical layout makes that
    # round-trip readable instead of forcing graphviz to fold it.
    attrs = dict(GRAPH_ATTR)
    attrs["rankdir"] = "TB"
    attrs["ranksep"] = "1.2"
    attrs["nodesep"] = "0.5"

    with Diagram(
        "GoalInsight pipeline — --remote-stages field_registration,tracking",
        filename=str(OUT_DIR / "pipeline_remote"),
        outformat="png",
        show=False,
        graph_attr=attrs,
    ):
        # Two side-by-side top-level clusters keep the layout horizontal;
        # nested clusters tend to make graphviz collapse into vertical.
        with Cluster("Local"):
            cli = Python("goalinsight CLI\n--remote-stages")
            remote = Python("pipeline/_remote.py")
            local_stages = Python("event_detection,\ntrack_consolidation,\nhighlights,\nannotated_video\n(unchanged)")
            local_out = Storage("output/<run>/\n<stage>/...")

            cli >> remote
            remote >> local_out
            local_out >> local_stages
            local_stages >> Edge(style="dotted") >> local_out

        with Cluster("AWS"):
            with Cluster("S3"):
                s3_inputs = S3("inputs/\nvideo + config\n+ calibration")
                s3_weights = S3("weights/\nyolov8x,\npnlcalib heads")
                s3_outputs = S3("outputs/\n<run>/<stage>/")
            ecr = ECR("ECR image\ngoalinsight-pipeline")
            iam = IAM("Execution role")

            with Cluster("SageMaker Processing Job"):
                job_fr = Sagemaker("field_registration\nml.g5.xlarge")
                job_trk = Sagemaker("tracking\nml.g5.xlarge")

            cw = Cloudwatch("CloudWatch\nlogs")

        # Cross-cluster edges
        remote >> Edge(label="upload") >> s3_inputs
        remote >> Edge(label="create_processing_job") >> job_fr
        remote >> Edge(label="create_processing_job") >> job_trk

        for job in (job_fr, job_trk):
            ecr >> Edge(style="dashed") >> job
            iam >> Edge(style="dotted", label="assume") >> job
            s3_weights >> Edge(label="weights") >> job
            s3_inputs >> Edge(label="ProcessingInputs") >> job
            job >> Edge(label="ProcessingOutputs") >> s3_outputs
            job >> Edge(style="dashed") >> cw

        s3_outputs >> Edge(label="download") >> remote


def render_chat_runtime() -> None:
    """Chat with AgentCore Runtime hosting the agent loop (opt-in).

    Same Bedrock + Code Interpreter under the hood; the difference is
    that ChatEngine runs in an AWS-managed MicroVM container instead of
    in-process. The local FastAPI app proxies turns over
    InvokeAgentRuntime and forwards the SSE stream back to the browser.
    """
    attrs = dict(GRAPH_ATTR)
    attrs["rankdir"] = "TB"
    attrs["ranksep"] = "1.0"
    with Diagram(
        "GoalInsight chat — AgentCore Runtime (opt-in)",
        filename=str(OUT_DIR / "chat_architecture_runtime"),
        outformat="png",
        show=False,
        graph_attr=attrs,
    ):
        user = Users("Browser")

        with Cluster("Local viewer process"):
            api = Fastapi("FastAPI\n/api/chat/stream")
            remote = Python("RemoteChatEngine\n(boto3 invoke_agent_runtime)")

        with Cluster("AWS"):
            with Cluster("S3"):
                run_s3 = S3("runs/<run>/\nevents.json,\ntracks.json,\nball_tracks.json,\n...")

            with Cluster("AgentCore Runtime (MicroVM)"):
                rt_api = Fastapi("/invocations\n/ping")
                rt_engine = Python("ChatEngine\n+ TOOL_DISPATCH")

            bedrock = Bedrock("Bedrock Runtime\nClaude Opus 4.7")
            sandbox = Bedrock("AgentCore\nCode Interpreter\n(aws.codeinterpreter.v1)")

        user >> Edge(label="HTML / SSE") >> api
        api >> Edge(
            label="invoke_agent_runtime\n(SSE passthrough)",
        ) >> rt_api
        rt_api >> rt_engine
        rt_engine >> Edge(
            label="download (1x per session)", style="dashed",
        ) << run_s3
        rt_engine >> Edge(label="invoke_model_with_response_stream") >> bedrock
        rt_engine >> Edge(
            label="run_python (optional)\nstart/invoke session",
        ) >> sandbox


def main() -> None:
    render_chat()
    render_chat_runtime()
    render_pipeline_local()
    render_pipeline_remote()
    print(f"wrote PNGs into {OUT_DIR}")


if __name__ == "__main__":
    main()
