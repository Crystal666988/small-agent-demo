# AI Prompt 与问题解决记录

题目允许用 AI 工具辅助开发，但核心 Runtime 自行实现。这里记录我在开发中如何使用 AI、
遇到的问题与决策，便于评审了解思路（"用 AI 帮你思考，而非帮你完成任务"）。

## 1. 关键设计决策（人来定，AI 辅助论证）

| 决策 | 选项 | 我的选择与理由 |
| --- | --- | --- |
| 工具调用协议 | 厂商原生 function-calling vs 自定义 JSON 协议 | **自定义 JSON**。题目要求"实现 LLM 输出解析逻辑"，用原生 function-calling 等于把这块交给厂商，就没有解析可写了。自定义协议让 `parser.py` 成为真实实现。 |
| 思考过程是否入 context | 入 / 不入 | **入**。ReAct 里让模型看到自己上一步 thought 能提升多步连贯性，代价是 token，靠压缩控制。 |
| memory 放哪 | history 里 vs system prompt 里 | **system prompt**。压缩会裁剪 history，而关键事实（用户名/偏好）必须稳定召回，放 system 段不被压缩。 |
| 压缩触发 | 每轮压 vs 超阈值才压 | **超阈值**。频繁压缩会不断改写前缀，破坏 KV/prompt cache 命中，得不偿失。 |
| calculator 实现 | `eval` vs AST 白名单 | **AST 白名单**。`eval` 有代码注入风险；AST 只放行数字与算术运算符，安全且够用。 |

## 2. 我给 AI 的代表性 Prompt（节选）

- "帮我评估：Agent 的工具调用用厂商原生 function-calling 还是自己解析 JSON，分别对'实现 LLM 输出解析'这个考点意味着什么？" —— 用于确认协议选型。
- "一个 JSON 提取器，要能处理：模型加了 ```json 代码块、前后有废话、字符串值里本身含 `{}`。给我括号配对扫描的边界条件。" —— 用于打磨 `parser.py`。
- "多窗口 session 隔离，todo 工具怎么绑定到当前 session 才能保证两个窗口互不干扰？" —— 用于 `runtime.py` 里 `todo_store` 闭包的设计。

（以上是"让 AI 帮我思考取舍"，具体代码结构、闭包绑定、状态机由我自己落地。）

## 3. 开发中遇到的问题与解决

1. **模型输出被 ```json 包裹**：真实 LLM（见 smoke 日志）几乎每次都用代码块包裹 JSON。
   → `parser.py` 先 `_strip_fences` 再做括号配对扫描，实测稳定解析。
2. **字符串内花括号干扰 JSON 定位**：朴素找 `}` 会在字符串含 `{}` 时截断。
   → 写了带 in-string / escape 状态的括号配对扫描器，并加单测 `test_parse_nested_braces_in_strings`。
3. **解析失败会不会让 loop 崩**：
   → 不崩。把 ParseError/未知工具/工具异常统一转成 observation 反馈给模型，让它自我纠正，
     并有 `max_steps` 兜底。对应 `test_parse_error_recovery` 等用例。
4. **API 双鉴权头**：兼容代理有的认 `x-api-key` 有的认 `Authorization: Bearer`。
   → `llm.py` 两个头都发。
5. **CJK token 估算**：`estimate_tokens` 对中文按更密的比例计，避免中文对话下压缩触发过晚。
6. **测试不能依赖网络**：抽象出 `llm.complete()` 接口，用 `ScriptedLLM/RoutedLLM` 注入，
   33 个单测全部离线跑通；另有 `smoke_live.py` 做真实 API 端到端验证。

## 4. 验证

- 离线单测：`python -m pytest -q` → 33 passed（解析/工具/运行时/上下文压缩全覆盖）。
- 真实 API 冒烟：`python -m tests.smoke_live` → 多步工具链、跨轮状态、追问召回均正确
  （日志见 `logs/`，含 calculator→final、weather→todo→final、todo list 追问）。
```
