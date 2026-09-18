# 架构设计题解答

> 面试正式选答：**第 6 题（滑动窗口限流）**，有确定解、能落地。
> 其余各题为学习性分析，答案与具体业务强绑定，这里给出我的思路框架。

---

## 第 6 题：滑动窗口限流器（正式作答）

### 需求

- 按 `userId` 限流：任意连续 `windowSeconds` 秒内每个用户最多 `limit` 次。
- `bool allow(String userId)`。
- 必须是**滑动**窗口（不是固定窗口）。
- 不活跃用户的数据要能清理，避免内存无限增长。

### 6.1 数据结构与核心逻辑（单机）

核心矛盾：固定窗口在窗口边界会出现「双倍突发」（比如 1s 前后各打 limit 次）。
滑动窗口要看「当前时刻往前推 windowSeconds」这段真实区间内的请求数。

有两种主流实现，我给出**滑动日志（精确）**和**滑动计数（近似、省内存）**：

#### 方案 A：滑动日志（Sliding Log，精确）

每个用户维护一个存请求时间戳的双端队列，每次请求先把过期时间戳弹掉，再看队列长度。

```text
struct Bucket { Deque<long> timestamps; long lastSeen; }
ConcurrentHashMap<String, Bucket> map

bool allow(userId):
    now = currentMillis()
    bucket = map.computeIfAbsent(userId, new Bucket)
    synchronized(bucket):                     # 单用户串行，锁粒度到 userId
        windowStart = now - windowSeconds*1000
        while (!bucket.timestamps.isEmpty()
               && bucket.timestamps.peekFirst() <= windowStart):
            bucket.timestamps.pollFirst()     # 移除滑出窗口的旧请求
        bucket.lastSeen = now
        if bucket.timestamps.size() < limit:
            bucket.timestamps.addLast(now)
            return true
        return false
```

- 精确，任意连续 windowSeconds 内严格 ≤ limit。
- 每个活跃用户内存 O(limit)（最多存 limit 个时间戳）。

#### 方案 B：滑动窗口计数（近似，省内存）

把窗口切成若干小格（如 windowSeconds 拆 60 格），只存每格计数，用「当前格 + 上一窗口按比例加权」近似。
内存 O(格数)，误差可控。高 QPS、limit 很大时用它。这里主答精确的方案 A。

#### 不活跃用户清理

三种手段（可组合）：

1. **惰性清理**：`allow` 里发现队列清空后为空 bucket，可顺手 `map.remove`。
2. **后台定时扫描**：定时线程遍历，移除 `now - lastSeen > windowSeconds` 的 bucket。
3. **有界缓存**：用带 TTL/LRU 的缓存（如 Caffeine，`expireAfterAccess = windowSeconds`）承载 map，
   自动淘汰不活跃用户，最省心。

### 6.2 复杂度

| | 时间 | 空间 |
| --- | --- | --- |
| 方案 A 滑动日志 | 单次 `allow` 均摊 O(1)（每个时间戳最多进出队各一次），最坏 O(过期条数) | O(活跃用户数 × limit) |
| 方案 B 滑动计数 | O(1) | O(活跃用户数 × 格数) |

### 6.3 四种算法各自适合的场景

- **固定窗口**：实现最简、计数即可；容忍边界双倍突发的场景（粗粒度统计、非严格限流）。
- **滑动窗口**：要求任意区间平滑限流、避免边界突发的接口级限流（本题）。
- **令牌桶**：允许一定**突发**、长期平均速率受限——API 网关最常用（平时攒令牌，偶尔可爆发）。
- **漏桶**：强制**恒定**出水速率、平滑流量、削峰——对下游有严格匀速要求时（如写入受限的存储）。

一句话：要「均匀」用漏桶，要「可突发但均值受限」用令牌桶，要「精确区间计数」用滑动窗口。

### 6.4 Redis 集群限流设计

多机下计数必须集中，放 Redis。关键是**原子性**——判断+累加必须一次完成，否则并发下会超发。

**首选：ZSET 滑动日志 + Lua 原子脚本**

- key：`rl:{userId}`（用 `{}` hash tag，保证 Cluster 下同一 key 落同一 slot，便于脚本原子执行）。
- 结构：ZSET，member = 唯一请求 id（如 `now:随机`），score = 时间戳 ms。
- Lua 脚本里一次做完：`ZREMRANGEBYSCORE` 清过期 → `ZCARD` 计数 → 未超则 `ZADD` → `PEXPIRE` 续期。

```lua
-- KEYS[1]=rl:user  ARGV: now, windowMs, limit, member
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1]-ARGV[2])
local c = redis.call('ZCARD', KEYS[1])
if tonumber(c) < tonumber(ARGV[3]) then
    redis.call('ZADD', KEYS[1], ARGV[1], ARGV[4])
    redis.call('PEXPIRE', KEYS[1], ARGV[2])   -- 空闲自动过期 = 自动清理
    return 1
end
return 0
```

- **原子性**：整段 Lua 在 Redis 单线程里原子执行，天然避免 check-then-act 竞态。
- **自动清理**：`PEXPIRE` 让不活跃 key 到期自动删，无需额外扫描。
- **精度/内存权衡**：ZSET 日志精确但内存大；超大规模可换 Redis 上的分格计数（多个小 key + INCR + EXPIRE）近似。
- **时间基准**：now 由脚本参数从**应用层统一传入**或用 `redis.call('TIME')`，避免多机时钟漂移。
- **热点**：单用户超高频会让单 slot 变热点，可对同一 user 加随机分片 key 再合并阈值。

---

以下为其余各题的学习性分析（非正式作答）。

## 第 1 题：Agent 后端的 Redis 设计

传统后端 Redis 主要给高频读做缓存。Agent 时代后端不再是最后一层，Redis 的职责扩展为：

1. **会话/上下文态**：session 元数据、多轮 history 指针、活跃 session 路由（哪台机器持有）。
2. **Agent 运行态**：正在执行的 task 状态、step 计数、工具调用去重键（幂等，见第 4 题）。
3. **KV Cache 亲和路由**：记录某 session 的 prompt 前缀命中在哪个推理实例，做 sticky 路由复用 KV cache（见第 2 题）。
4. **工具结果缓存**：确定性工具（搜索、汇率）结果缓存，降低重复外部调用。
5. **限流/配额/去重**：用户级/工具级速率限制、幂等键。
6. **异步桥接**：Streams 做工具任务队列，Pub/Sub 推 token 流（见第 3 题）。

要点：Redis 从「只读加速」变成「Agent 运行时的共享状态中枢 + 异步中枢」，需要区分「短命运行态（TTL 短）」和「可缓存结果（TTL 长）」，并为 KV cache 亲和保留路由信息。

## 第 2 题：KV Cache + Context

目标：效果与耗时综合最优。核心是**最大化 KV cache 前缀复用**同时控制 context 长度。

- **稳定前缀**：把 system prompt、工具 schema、长期不变的记忆放在 context **最前面且逐字节稳定**，
  使推理引擎（vLLM/TensorRT-LLM 的 prefix cache 或 Anthropic 的 prompt caching）能命中前缀 KV，首 token 大幅提速。
  本项目 `prompts.py` 有意把协议+schema 放最前，就是为此类复用留出结构。
- **变动内容后置**：随每轮变化的用户输入、工具结果放后面，避免使前缀失效。
- **压缩换命中率**：context 过长既慢又贵，用摘要压缩（本项目 `compressor.py`）把旧history折叠，
  但压缩会改写前缀 → 命中率下降，所以压缩**不宜过频**，要在「省 token」与「保 cache」间权衡：设较高阈值、批量压缩。
- **会话亲和路由**：同一 session 尽量路由到持有其 KV cache 的实例（配合第 1 题 Redis 记录亲和）。

案例：本 mini-agent 的 system prompt 固定前置 + observation 后置，就是让 Anthropic prompt caching 命中——
smoke 测试日志里 `cache_read_input_tokens` 明显大于 `cache_creation`，即前缀被复用。

## 第 3 题：后端 + Agent 异步/同步机制（硬件对话，首 token < 2s）

矛盾：首 token 要 2s 内，但工具耗时长。策略：

- **先说话再干活**：拿到用户意图后**立刻流式**输出一句过渡语（「好的，正在帮你查…」），
  用 SSE/WebSocket 边生成边推，满足首 token；工具异步跑。
- **工具异步化**：慢工具丢进队列（Redis Streams / MQ），Agent 主循环不阻塞，
  结果回来后再续接第二段回复（two-phase / 追加消息）。
- **预测性并行**：能并行的工具并行发起；对高概率要用的工具提前预热。
- **超时与降级**：每个工具设超时，超时给兜底话术，避免用户干等。
- 注意点：过渡语不能承诺未验证结果；异步结果回填要保证顺序与 session 一致性；
  首 token 用小/快模型或缓存问候，重活交给大模型异步。

## 第 4 题：Agent 幂等性与重试

- **副作用工具打幂等键**：下单类工具要求调用方传 `idempotency_key`（如 `session+step+参数hash`），
  服务端用该键去重（Redis SETNX），重试同键直接返回首次结果，不重复下单。
- **框架感知幂等**：在工具注册的 schema/元数据上标注 `idempotent: true/false` 与 `side_effect: read|write`。
  框架据此决定：读工具可安全重试，写工具必须带幂等键且重试走「查询-确认」而非盲目重发。
- **重试策略**：区分「可重试错误（超时/5xx/网络）」与「不可重试（参数错）」；
  超时的写操作先查询下游状态（订单是否已创建）再决定重发。
- 本项目已有雏形：`Tool.run` 前置参数校验、错误转 observation；生产上再加 `idempotent` 标志与幂等键存储即可。

## 第 5 题：Agent 与后端数据一致性

问题：业务逻辑上移到 Agent 后，数据容易 Agent 存一份、后端存一份。

- **单一事实源（SSOT）**：业务权威数据（订单、账户）**只存后端**，Agent 侧只持有会话态/推理态，不做权威存储。
- **Agent 通过后端 API 读写**：Agent 不直接写库，所有落地经后端领域服务，复用其事务与校验。
- **读写分层**：Agent 的「记忆/上下文」是**派生/易失**数据（可重建），后端是**持久权威**数据；明确边界避免双写。
- **一致性手段**：写走后端事务；跨服务用 outbox/事件 + 最终一致；幂等键防重（接第 4 题）。
- **缓存**：Agent 侧对后端数据只读缓存并设 TTL / 失效事件，不作为写入源。

## 第 7 题：硬件语音流接入与 ASR 会话

- **鉴权与会话**：`deviceId + sn + token` 建连时校验（token 可为设备证书/JWT，服务端校签+查设备白名单）；
  一次「说话轮次」用 `utteranceId`（服务端生成、随首帧下发）标识。
- **音频上行帧**：`{utteranceId, seq, timestampMs, codec(opus/pcm), isLast, payload}`。
  乱序→按 seq 重排缓冲；重复→按 seq 去重；丢包→短暂等待+ASR 对小丢包鲁棒，超阈值标记。
- **断网重连**：重连带 `utteranceId + lastAckedSeq`，服务端返回已收到的 seq，设备**只重传未确认帧**；
  已识别出的部分不可重复识别（服务端按 seq 幂等，重复帧丢弃）。
- **并发约束**：同 deviceId 只允许一个进行中轮次——服务端用 Redis `SETNX device:activeUtterance`，
  新请求要么拒绝（忙）要么抢占旧轮次（视产品策略）。
- **A. 数据结构**：`Map<utteranceId, Utterance{deque<Frame> ordered by seq, lastSeq, state}>`；
  插入/去重 O(log n)（有序结构）或 O(1)（数组+seq 索引）。
- **B. 首帧到最终文本流程**：鉴权建连 → 首帧创建 utterance → 帧重排入缓冲 → 流式送 ASR →
  收 ASR 中间态（可下发） → 端点检测判定说完（见第 9 题） → 取最终文本 → 交对话服务 → 下发播报。

## 第 8 题：多机型主板 + 多 ASR 厂商适配

- **模块划分**：设备接入层（连接/鉴权）→ 协议解析层（各机型帧格式→统一事件）→
  ASR 适配层（各厂商 SDK→统一接口）→ 对话调用层 → 指令下发层（统一指令→各机型格式）。
- **设计模式**：
  - **适配器模式**：每个 ASR 厂商/每种机型一个 Adapter，把外部协议转成内部统一模型，解决「异构接入」。
  - **策略模式**：运行时按机型/厂商选择解析/适配策略；也可加**工厂**按配置创建 Adapter。
- **统一内部模型**：
  上行 `AudioEvent{deviceId, utteranceId, seq, ts, pcm/opus, isLast}`；
  下行 `DeviceCommand{deviceId, type(play/led/...), payload}`。外部协议由各 Adapter 映射进/出。
- **新增「机型 X + ASR Y」**：只需新增两个 Adapter（协议解析 + ASR 适配）并注册，
  **主流程、对话服务、内部模型不动**。自动化测试：给每个 Adapter 喂录制的真实帧样本→断言产出统一事件；
  用契约测试保证所有 Adapter 都满足统一接口。

## 第 9 题：端点检测与一轮对话状态机

状态机：`IDLE → LISTENING → ENDPOINTED → DIALOGING → SPEAKING →（回 IDLE 或 LISTENING）`。

- 收到有效语音帧 → 进/留在 `LISTENING`，刷新 `lastVoiceTs`。
- `now - lastVoiceTs >= silenceMs` → 判端点，转 `ENDPOINTED`（一句话说完）。
- `now - startTs >= maxUtteranceMs` → 强制结束，转 `ENDPOINTED`（防止一直不停）。
- `ENDPOINTED` → 取完整文本送对话服务 → `DIALOGING` → 拿到播报 → `SPEAKING`（默认不监听，防回声/自触发）。
- 播报结束的两种策略：
  - **自动续听**：回 `LISTENING`，体验连续、免唤醒，但易被环境音误触发、功耗高。
  - **等下次唤醒**：回 `IDLE`，省电抗误触，但每轮都要唤醒词、体验略繁琐。
- 静音判定来源：设备端 VAD、或云端按帧能量阈值、或 ASR 中间态长时间无新增字 → 三者可融合，
  云端兜底避免设备 VAD 不准。
```
