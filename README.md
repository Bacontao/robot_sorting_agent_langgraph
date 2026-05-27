# Robot Sorting Agent LangGraph

一个面向机器人语义分拣场景的多模态 Agent 项目。系统输入一张图片和一句自然语言指令，自动完成图像感知、任务理解、动作规划、计划审核、失败诊断、动态回退和执行命令生成。


## 项目亮点

- **图像 + 自然语言输入**：支持用自然语言描述分拣、摆放、空间关系任务。
- **真实视觉感知链路**：优先使用 GroundingDINO + SAM vit_b 做开放词汇目标检测和分割，YOLO 作为备用后端。
- **结构化机器人计划**：将自然语言任务转成 `inspect / pick / place / skip` 等可执行步骤。
- **空间关系任务支持**：支持 `left_of / right_of / above / below / near` 等相对空间关系。
- **LangGraph 动态流程编排**：把感知、规划、审核、修复、执行适配拆成清晰节点，并支持失败后动态回退。
- **LLM 诊断错误来源**：计划或执行出错时，由 LLM 判断是感知、意图解析、分配、步骤生成还是执行适配出了问题。
- **可观测中间产物**：每次运行都会保存 `segmentation.json`、`object_table.json`、`plan.json`、`execution_commands.json`、`agent_trace.json` 等文件，方便调试。
- **离线评测体系**：提供 100 条测试用例生成和语义指标评测脚本。

## 系统流程

```text
Image + Instruction
        |
        v
Segmentation
GroundingDINO + SAM / YOLO
        |
        v
Perception
build object table
        |
        v
Intent Parsing
understand user task
        |
        v
Rule + Assignment
match task to objects
        |
        v
Step Generation
build inspect/pick/place plan
        |
        v
Plan Review
LLM critic validates the plan
        |
        +---- fail ----> Failure Diagnosis ----> dynamic rollback
        |
        v
Execution Adapter
convert plan to robot commands
        |
        v
Run + Feedback
        |
        v
Pipeline Response
```


## 目录结构

```text
.
├── src/robot_sorting_agent/
│   ├── api.py                  # FastAPI 服务入口
│   ├── cli.py                  # 命令行入口
│   ├── compat_langgraph.py     # LangGraph 兼容层
│   ├── execution.py            # 执行命令生成和 run
│   ├── graph.py                # 主工作流，LangGraph 节点和条件边
│   ├── image_utils.py          # 支持图片路径、URL、base64 处理
│   ├── llm.py                  # LLM/VLM 调用和结构化输出修复
│   ├── observability.py        # 中间文件写入
│   ├── payloads.py             # 给 LLM 的 payload 压缩
│   ├── perception.py           # 从分割结果构建物体表
│   ├── planning.py             # 任务解析、规则、分配、计划、诊断
│   ├── prompts.py              # 各阶段提示词
│   ├── schemas.py              # 全项目数据结构
│   ├── segmentation.py         # GroundingDINO + SAM / YOLO / stub 后端
│   └── settings.py             # 环境变量配置
├── scripts/
│   ├── download_grounded_sam_models.py # 下载 GroundingDINO + SAM 模型
│   ├── generate_eval_cases.py          # 生成评测用例
│   ├── prepublish_check.py             
│   └── run_eval.py                     # 跑离线评测
├── samples/                    # 示例图片和请求
├── tests/                      # 测试用例
├── docs/                       # 中文说明和评测报告
├── .env.example                # 通用环境变量模板
├── .env.siliconflow.example    # 硅基流动配置模板
├── pyproject.toml
└── README.md
```

## 快速开始

### 1. 创建环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[langgraph,vision,grounded-sam,dev]'
```

如果只想跑轻量测试，不下载大模型，可以先安装：

```bash
pip install -e '.[langgraph,dev]'
```

### 2. 配置环境变量


```bash
cp .env.siliconflow.example .env
```

然后只在本地 `.env` 里填写自己的 key：

```env
OPENAI_COMPAT_API_KEY=sk-your-real-key
```


### 3. 下载 GroundingDINO + SAM vit_b 模型

```bash
python scripts/download_grounded_sam_models.py
```

如果本机证书链有问题，可以使用：

```bash
python scripts/download_grounded_sam_models.py --insecure
```

下载完成后，本地会出现：

```text
.models/GroundingDINO_SwinT_OGC.py
.models/groundingdino_swint_ogc.pth
.models/sam_vit_b_01ec64.pth
.models/bert-base-uncased/
```


### 4. 运行一次 CLI 测试

```bash
env PYTHONPATH=src .venv/bin/python -m robot_sorting_agent.cli \
  --request samples/request_test.json \
  --print commands
```

示例请求格式：

```json
{
  "image": {
    "image_path": "samples/1.png"
  },
  "instruction": "把苹果放在香蕉的左边"
}
```

成功时会生成类似命令：

```json
[
  {
    "command_id": "cmd_001",
    "action": "pick",
    "object_id": "obj_001"
  },
  {
    "command_id": "cmd_002",
    "action": "place",
    "object_id": "obj_001",
    "relation": "left_of",
    "reference_object_id": "obj_002"
  }
]
```

### 5. 启动 FastAPI 服务

```bash
env PYTHONPATH=src uvicorn robot_sorting_agent.api:create_app --factory --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

调用 pipeline：

```bash
curl -X POST http://127.0.0.1:8000/pipeline \
  -H 'Content-Type: application/json' \
  -d @samples/request_test.json
```

## 环境变量说明

常用配置在 `.env.example` 和 `.env.siliconflow.example` 中。

### 感知后端

```env
SEGMENTATION_BACKEND=auto
SEGMENTATION_FALLBACK_CHAIN=grounded_sam,yolo
GROUNDING_DINO_CONFIG=.models/GroundingDINO_SwinT_OGC.py
GROUNDING_DINO_CHECKPOINT=.models/groundingdino_swint_ogc.pth
SAM_CHECKPOINT=.models/sam_vit_b_01ec64.pth
SAM_MODEL_TYPE=vit_b
YOLO_MODEL_PATH=yolo11n-seg.pt
```

`auto` 模式会先尝试 GroundingDINO + SAM。如果没有检测结果、置信度过低或后端报错，会按 fallback 链尝试 YOLO。

### 硅基流动模型配置

```env
LLM_BACKEND=openai_compatible
OPENAI_COMPAT_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_COMPAT_API_KEY=sk-your-siliconflow-api-key
OPENAI_COMPAT_RESPONSE_FORMAT=json_object

VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
VLM_OPENAI_COMPAT_RESPONSE_FORMAT=text

PLANNER_MODEL=Qwen/Qwen2.5-14B-Instruct
CRITIC_MODEL=Qwen/Qwen2.5-14B-Instruct
TOOL_POLICY_MODEL=Qwen/Qwen2.5-14B-Instruct
REPLAN_MODEL=Qwen/Qwen2.5-14B-Instruct
```

视觉模型通常更适合 `text` 返回格式，规划和审核模型使用 `json_object` 更稳定。

## 中间文件说明

每次运行会在 `artifacts/<request_id>/` 下保存调试文件：

| 文件 | 含义 |
| --- | --- |
| `segmentation.json` | 图像分割候选物体 |
| `segmentation_meta.json` | 使用了哪个分割后端、是否 fallback |
| `object_table.json` | 结构化物体表 |
| `task_intent.json` | 用户任务意图 |
| `rules.json` | 从任务意图转出的规则 |
| `assignments.json` | 物体和规则的匹配 |
| `plan.json` | 机器人动作计划 |
| `plan_review.json` | LLM 对计划的审核 |
| `diagnosis_*.json` | 失败来源诊断和回退决策 |
| `execution_commands.json` | 最终执行命令 |
| `dry_run.json` | 执行前检查结果 |
| `execution_results.json` | 执行反馈结果 |
| `agent_trace.json` | LangGraph 每个节点的运行轨迹 |
| `workflow_state.json` | 最终工作流状态 |

`artifacts/` 是运行产物，不建议提交到 GitHub。重要评测结论可以整理到 `docs/`。

## 评测

生成 100 条评测用例：

```bash
env PYTHONPATH=src .venv/bin/python scripts/generate_eval_cases.py --count 100
```

运行评测：

```bash
env PYTHONPATH=src .venv/bin/python scripts/run_eval.py \
  --cases samples/eval_cases.jsonl \
  --output artifacts/eval_report.json
```

当前一次完整评测结果：

```text
num_cases: 100
exception_rate: 0.0
semantic_success_rate: 0.95
plan_valid_rate: 1.0
command_valid_rate: 1.0
avg_perception_recall: 0.975
avg_assignment_accuracy: 0.99
avg_relation_accuracy: 0.97
avg_command_accuracy: 0.96
backend_usage: grounded_sam = 100
```

详细中文评测说明见 `docs/EVALUATION_REPORT_CN.md`。


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
