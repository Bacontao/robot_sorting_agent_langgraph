# 项目说明（LangGraph 版）

## 1. 项目定位
这是一个面向机器人语义分拣场景的多模态 Agent 智能中间层。系统输入图像与自然语言任务，输出机械臂可直接消费的执行命令；如果计划评审或执行反馈发现问题，则先用 LLM 诊断错误来源，再由 LangGraph 动态回退到对应节点继续执行。

## 2. 为什么重写成 LangGraph
原项目核心就是一条有状态、可闭环、可重规划的流程链路：
- 感知层
- 规划层
- 计划评审与失败诊断
- 执行适配
- 执行反馈
- 动态回退 / replan

这类问题非常适合用 LangGraph 来表达，因为每个节点都有明确状态输入输出，并且节点之间存在条件跳转。

## 3. 核心流程
1. `segmentation`
   - YOLO 风格分割边界层，或 GroundingDINO + SAM 开放词汇分割
   - 输出 `SegmentationCandidate`
2. `perception`
   - Qwen2.5-VL-7B-Instruct 做对象语义增强
   - 输出 `ObjectTable`
3. `parse_intent`
   - GPT-4.1-mini 做任务意图解析
   - 输出 `TaskIntent`
4. `assignment`
   - 根据 `TaskIntent` 和 `ObjectTable` 生成规则与物体分配
   - 输出 `Rule`、`Assignment`
5. `step_generation`
   - 根据 assignment 生成 inspect / pick / place 等步骤
   - 输出 `Plan`
6. `plan_review` / `validate_plan`
   - GPT-4.1-mini 作为 critic 评审模型
   - 判断计划是否合格、是否需要诊断和回退
7. `diagnose_plan_issue`
   - LLM 判断错误来自感知、意图解析、assignment、步骤生成还是执行适配
   - LangGraph 根据 `FailureDiagnosis.restart_from` 回退到对应节点
8. `repair`
   - 当诊断认为只需要局部修补时，LLM 智能修复计划
   - 可执行 inspect / reassign / skip / reperceive 等动作
9. `execution_adapter`
   - 把计划转成 `ExecutionCommand`
   - 再做 `dry-run`
10. `execution_feedback`
   - 解析执行结果
   - 判断是否需要诊断
11. `diagnose_execution_issue` / `replan`
   - LLM 根据执行反馈判断是命令生成问题还是实际执行问题
   - 如果是执行失败，则根据反馈重规划
   - 回到执行适配节点继续闭环

## 4. 为什么 planner / critic / replan 都用 GPT-4.1-mini
这是一个“同一底层模型、多角色调用”的设计：
- Planner：负责任务意图解析
- Critic：负责计划评审、失败诊断与 repair
- Replan：负责根据执行反馈诊断和重新规划

它们不是一次调用里连续做三件事，而是在不同阶段以不同 prompt、不同输入和不同 schema 被独立调用。这样做有两个好处：
1. 逻辑解耦，便于调试和替换；
2. 成本、延迟和质量平衡更好。

## 5. 为什么不直接用 LangChain / Spring AI 做主干
主干工作流需要强状态管理、条件跳转、闭环和 artifact dump。LangGraph 更适合做这类有状态图式编排。LangChain 在这个项目里更适合作为适配层，把 end-to-end pipeline 封装成 `StructuredTool` 给外部系统调用。

## 6. 关键工程能力
- `AgentTrace`：记录每个节点为什么执行 / 为什么跳过、输入输出摘要、耗时和异常
- `artifact dump`：按 request_id 落盘保存 `segmentation.json / object_table.json / task_intent.json / assignments.json / plan.json / diagnosis_*.json / execution_commands.json / dry_run.json / execution_results.json / workflow_state.json`
- `benchmark`：围绕意图解析有效率、命令结构有效率、repair 成功率、闭环准备度和端到端时延做离线评测

## 7. GitHub 展示建议
建议在仓库首页突出以下几点：
- 使用 LangGraph 表达主干流程
- GPT-4.1-mini 在 plan_review / failure diagnosis / repair / replan 中体现 Agent 智能
- 输出机械臂可直接消费的 `ExecutionCommand`
- 支持 FastAPI / CLI / LangChain 适配
