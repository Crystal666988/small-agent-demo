# mini-agent — 从零实现的最小可用 Agent

[![tests](https://github.com/Crystal666988/small-agent-demo/actions/workflows/tests.yml/badge.svg)](https://github.com/Crystal666988/small-agent-demo/actions/workflows/tests.yml)

一个不依赖任何 Agent 框架（无 langgraph / openhands / PI 等）的最小 Agent Runtime。
核心循环、工具注册、LLM 输出解析、session 管理、context 压缩全部手写实现，跑在**真实 LLM API** 上。
支持两类 provider：**OpenAI 兼容**（DeepSeek / Moonshot / 本地 vLLM，走 `/chat/completions`）
与 **Anthropic 兼容**（走 `/v1/messages`），用 `MINIAGENT_PROVIDER` 切换。实测已在 DeepSeek `deepseek-chat` 上跑通。

- 语言：Python 3.10+（除 `requests` 外仅用标准库）
- 已通过 38 个离线单元测试 + 真实 API（DeepSeek）端到端冒烟测试

---

## 1. 快速运行

```bash
cd mini-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # 默认用 DeepSeek：填 OPENAI_API_KEY 即可
python -m miniagent         # 启动交互式多窗口 CLI
```

CLI 命令：

| 命令 | 作用 |
| --- | --- |
| `/new [title]` | 新建并切换到一个新窗口（session） |
| `/switch <id>` | 切到已有窗口，例如 `/switch win-1` |
| `/sessions` | 列出所有窗口及其状态 |
| `/todos` | 查看当前窗口的待办 |
| `/memory` | 查看当前窗口记住的持久事实 |
| `/trace` | 开关实时 trace 打印 |
| `/quit` | 退出 |

其它任何输入都会作为用户消息发给 Agent。

### 跑测试

```bash
python -m pytest -q                 # 离线单测（用 mock LLM，不联网）
python -m tests.smoke_live          # 真实 API 端到端冒烟（需要 .env）
```

---

## 2. 系统设计

### 2.1 目录结构

```
miniagent/
  config.py       读取 .env / 环境变量
  llm.py          Anthropic 兼容 HTTP 客户端（带重试），可被 mock 替换
  parser.py       解析 LLM 文本 -> Decision（思考 / 工具调用 / 最终答案）
  prompts.py      构造 system prompt（协议 + 工具 schema + 记忆注入）
  session.py      Session / SessionManager，多窗口隔离 + context 组装
  compressor.py   context 超预算时的基础压缩（LLM 摘要 + 离线兜底）
  runtime.py      核心 Agent 循环（本项目的心脏）
  trace.py        逐条 JSONL 执行日志
  cli.py          多窗口交互式命令行
  tools/
    base.py       Tool 抽象 + ToolRegistry（名称/描述/参数 Schema）
    builtin.py    calculator / search(mock) / weather(mock) / todo
tests/            mock LLM + 单测 + 冒烟脚本
logs/             运行时产生的 trace-<session>.jsonl
```

### 2.2 核心循环（runtime.py）

对应题目的 Loop 四步，每个用户输入触发一次：

```
接收用户输入
  └─> 请求 LLM 给出决策（思考）
        ├─ final_answer  ->  返回给用户，结束本轮
        └─ tool_call     ->  执行工具，把结果作为 observation 追加进 context
                              然后回到「请求 LLM」继续 loop
  直到拿到最终答案，或达到 max_steps 上限
```

关键设计点：

- **不用厂商原生 function-calling**：LLM 被要求只输出一个 JSON 对象，工具调用由我们自己的
  `parser.py` 从文本里提取。这样「LLM 输出解析」逻辑真正落在 Runtime 内部，符合题目要求。
- **决策协议**：模型每轮必须返回
  `{"thought": ..., "tool_call": {...}}` 或 `{"thought": ..., "final_answer": ...}`，
  可选带 `"remember": "..."` 写入持久记忆。
- **解析容错**：`parser.py` 能剥离 ```json 代码块、忽略前后废话、用括号配对扫描定位 JSON
  （字符串内的 `{}` 不会干扰）。解析失败不会崩溃，而是把错误作为 observation 反馈给模型让它自我纠正。

### 2.3 工具注册机制（tools/）

每个 `Tool` 声明 `name`、`description`、`parameters`（JSON-Schema 风格）和 `required`。
`ToolRegistry.render_schemas()` 把所有 schema 渲染进 system prompt，LLM **纯粹依据 schema** 决定
调哪个工具、传什么参数。工具执行前会校验必填参数。

内置四个工具：

- `calculator`：AST 安全求值（只允许数字与算术运算符，**不用 eval，无注入面**）
- `search`：mock，返回确定性结果，便于测试复现
- `weather`：mock，支持中英文城市名
- `todo`：**session 级**状态，`add / list / done`

### 2.4 异常处理

- LLM 调用：网络错误 / 5xx / 429 指数退避重试；彻底失败抛 `LLMError`，Runtime 降级返回错误信息而非崩溃。
- 解析失败 / 未知工具 / 工具报错 / 工具意外崩溃：全部转成 observation 喂回模型，loop 继续。
- `max_steps` 兜底：模型陷入死循环时强制结束。

### 2.5 执行 trace / 日志

`trace.py` 把每一步（用户输入、LLM 原始输出、工具调用、工具结果、错误、最终答案、压缩事件）
写成一行 JSON 到 `logs/trace-<session>.jsonl`，`/trace` 可实时打印到 stderr。每个 session 一个文件，便于审计与回放。

---

## 3. Session 与 Context 管理

### 3.1 Session 隔离（对应题目的窗口1 / 窗口2）

`SessionManager` 按 `session_id` 维护多个 `Session`，每个 Session 有**独立**的
`history`、`todos`、`memory`、`summary`。

Runtime 在每一轮用 `build_registry(todo_store=lambda: session.todos)` 把 `todo` 工具**绑定到当前
session 的待办列表**，从根上保证窗口1（查天气记待办）和窗口2（写周报记待办）互不干扰。
`tests/test_runtime.py::test_session_isolation` 对此有断言。

用户可随时 `/switch` 回任一窗口继续聊，历史与状态都在。

### 3.2 塞进 context 的信息 & 追问

每轮发给 LLM 的 context = **system prompt**（协议 + 工具 schema + 记忆事实）+ **history 回放**。
history 里保留三类信息，映射到 API 的两种角色：

| 内部角色 | 内容 | 映射到 API |
| --- | --- | --- |
| `user` | 用户输入 | user |
| `assistant` | Agent 的思考 / 决策 JSON | assistant |
| `observation` | 工具执行结果 | user（加 `[Tool result]` 前缀区分） |

因为每轮都完整回放 history，所以天然支持两种追问：

- **纯对话追问**：模型能看到之前的用户/助手消息（见 `test_followup_remembers_prior_state`）。
- **带工具的追问**：模型能看到之前的工具结果 observation，也能再次调用工具（如「我有哪些待办」会再调 `todo list`）。

选择「Agent 思考过程也入 context」是有意为之：ReAct 里让模型看到自己上一步的 thought，能显著提升多步任务的连贯性。

### 3.3 Context 压缩

`compressor.maybe_compress()`：当 session 估算 token 超过预算（默认 8000，可配）时，
把**最旧的一批** turn 交给 LLM 生成 ≤150 字摘要，折叠进 `session.summary`，只保留最近 6 轮原文。
LLM 调用失败时回退到截断拼接，保证离线/异常下仍可用。摘要作为一条 user 前言注入下一轮 context。
记忆事实（memory）永不丢弃。复杂压缩（语义去重 / 重要性打分）按题目要求不在此实现。

---

## 4. Memory：召回时机与放置方式

本项目的 memory 指**跨轮持久的一句话事实**（如「用户偏好公制单位」），区别于会随压缩而摘要化的对话 history。

- **放置（写入）时机**：模型在返回决策 JSON 时，可附带顶层 `"remember": "<事实>"` 字段。
  Runtime 在 `runtime.py` 里检测到就调用 `session.remember()` 去重后存入 `session.memory`。
  即「由模型自主决定什么值得长期记住」，而非把所有对话都当记忆。
- **召回时机**：`prompts.build_system_prompt()` 在**每一轮**都把该 session 的全部 memory 事实
  以「REMEMBERED FACTS」段落注入 system prompt。因此记忆是**常驻**的，不受 history 压缩影响，
  模型在任何一步都能看到。
- **放置位置**：memory 挂在 `Session` 上 ⇒ 与窗口绑定、天然隔离。`/memory` 命令可查看当前窗口记住的事实。

> 设计取舍：把 memory 放 system prompt 而非 history，是因为 system 段不会被压缩裁剪，
> 保证「用户名字/偏好」这类关键事实的召回稳定性；对应仓库根目录 CLAUDE 记忆机制的理念——
> 一条事实一处存放、召回时判断相关性。

---

## 5. 架构设计题解答

见 [`docs/architecture-answers.md`](docs/architecture-answers.md)。
面试选答第 6 题（滑动窗口限流，有确定解），同时附上其余 8 题的分析作为学习记录。

## 6. AI Prompt 与问题解决记录

见 [`docs/ai-prompt-log.md`](docs/ai-prompt-log.md)。
```
