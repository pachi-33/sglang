# Prefill 侧专家卸载基本框架

本文先划分类的职责边界、对象归属和调用关系，讨论 `mode=prefill_only`、`prefill_resident_ratio > 0` 且取整后每层常驻数 `R > 0` 的每层部分专家常驻路径。代码依据：`simplify` 工作树，基准提交 `5b9f6624c0`；核对日期：2026-11-16。

文中的范围均为当前 worker / rank；启用 PP 时，模型执行和专家目录只覆盖本 rank 所属 stage。当前部分常驻路径要求 ModelSlim W8A8。主调用链以 AscendOps dispatcher 为例，DeepEP 的接入差异单独说明。

## 1. 核心类的职责边界

| 类 | 管理范围与主要职责 | 职责边界 |
|---|---|---|
| `TpModelWorker` | worker 级入口；创建 `ModelRunner`，把调度批次转换为 `ForwardBatch`，调用模型执行并组织返回结果。 | 调度批次进入模型的边界；专家存储和预取细节由下层完成。 |
| `ModelRunner` | 持有模型和 `moe_offloader`；组织模型加载、设备执行及运行资源；用 `forward_scope` 包住一次 `_forward_raw()`。 | 管理一次模型前向的外层生命周期；每层的权重就绪和缓冲区复用由 offloader 子组件处理。 |
| `FusedMoE` | 模型树中的专家模块，接收 hidden states 和上游的 top-k 结果；组织 `dispatch → run_moe_core → combine`，持有量化方法和 dispatcher。 | 这里的 `layer` 指专家模块；整个 DecoderLayer 还包含 attention 等计算。专家路由结果由上游提供。 |
| `BaseOffloader` | 通用 offloader 基类；提供 `wrap_modules()`、`post_init()` 和 `forbid_copy_engine_usage` 的默认行为。 | 基类没有定义 `prepare_layer()`、`prepare_weights()` 或 `forward_scope()`；这些是具体 MoE offloader 扩展出来的接口。 |
| `AscendOpsOffloader` | 当前 rank 的 MoE offload 管理对象，即代码中的 `owner`；持有专家目录、各层 Host 存储、格式对象、NPU 权重池、复制流、执行策略和当前调用上下文。 | 负责资源归属和对外接入；把加载细节、Prefill 供权、流水线时序分别交给 storage、prefill 和 pipeline。 |
| `ExpertLayerStorage` | 每个受管理专家层一个实例；分配 Host 区域、向 `FusedMoE` 注册参数及 `weight_loader` 回调；接收本 rank 的全部专家分片；校验、逐专家转换并封存 Host 主副本。 | 管理单层权重的加载状态及 Host 数据；运行时搬运计划和 NPU 缓冲区轮转由其他组件管理。 |
| `PrefillOnlyExecution` | `prefill_only` 执行策略；校验配置和拓扑、计算常驻数及槽位数；在 `post_init(owner)` 中建立 NPU 执行资源；实现整轮 `forward_scope(owner, batch)`。 | 它由 offloader 持有，通过方法参数接收 `owner`；自身不是 `BaseOffloader` 的子类，也不实现专家矩阵计算。 |
| `AscendPrefillWeights` | 当前 rank 跨层共享的 Prefill 权重供给对象；建立常驻/冷专家视图、填充常驻权重、建立 W13/W2 搬运计划、提供 `quant_info` 和分组信息、通知层使用结束。 | 借用 `owner` 的 Host 主副本和 NPU 大权重池，自己建立视图、元数据、传输对象及 pipeline；实际 GMM 由 runner 执行。 |
| `LayerPrefetchPipeline` | 一个实例跟踪本轮所有受管理层；维护层顺序、当前层序号、双缓冲选择和 W13/W2 完成凭据；安排初始预取与后续复用。 | 通过注入的 `load` 回调提交搬运；不解释权重格式，也不执行路由、矩阵计算或通信算子。 |

源码入口：[TpModelWorker][worker]、[ModelRunner][modelrunner]、[FusedMoE][fused]、[BaseOffloader][baseoffloader]、[AscendOpsOffloader][offloader]、[ExpertLayerStorage][storage]、[PrefillOnlyExecution][strategy]、[AscendPrefillWeights][prefill]、[LayerPrefetchPipeline][pipeline]。

## 2. UML 类图：对象归属与引用

图例：`<|--` 表示继承，`*--` 表示创建并持有的主要组成对象，`-->` 表示保存对象引用，`..>` 表示调用依赖。组合线描述本特性的逻辑归属，不表示 Python 中没有其他引用，也不承诺析构顺序。

`L` 是本 rank / stage 中受管理的 routed MoE 层数。图中省略与本特性无关的基类；`ModelModule` 是模型模块树的概念节点，代码中没有这个同名类。

```mermaid
classDiagram
    direction TB

    class TpModelWorker {
        -_model_runner
        -_init_model_runner()
        +forward_batch_generation()
    }
    class ModelRunner {
        +model
        +moe_offloader
        +load_model()
        +forward()
        -_forward_raw()
    }
    class ModelModule {
        <<conceptual>>
    }
    class FusedMoE {
        +layer_id
        +quant_method
        +dispatcher
        -_ascend_ops_offloader
        -_moe_offload_storage
        +forward_impl()
        +run_moe_core()
    }
    class BaseOffloader {
        +wrap_modules()
        +post_init()
        +forbid_copy_engine_usage
    }
    class AscendOpsOffloader {
        +catalog
        +storages
        +format
        +host_pool
        +execution
        +weight_copy_stream
        +prepare_layer()
        +post_init()
        +forward_scope()
        +prepare_weights()
    }
    class ExpertLayerStorage {
        +layer
        +spec
        +component_capabilities
        -_state
        -_loader()
        +finalize()
        +require_sealed()
    }
    class PrefillOnlyExecution {
        +resident_ratio
        +resident_experts
        +configure(owner, path)
        +post_init(owner)
        +forward_scope(owner, batch)
    }
    class AscendPrefillWeights {
        +owner
        +banks
        +plans
        +quant_info
        +weights()
        +cold_weights()
        +finish_layer()
        -_load()
    }
    class LayerPrefetchPipeline {
        +layer_ids
        +current
        +ready
        +load
        +begin_forward()
        +start()
        +acquire()
        +wait_for_w2()
        +retire()
        +finish_forward()
    }
    class _Context {
        +compute_stream
        +invocation_id
        +resources
        +retain()
        +record_compute_completion()
    }

    TpModelWorker "1" *-- "1" ModelRunner : _model_runner
    ModelRunner "1" *-- "1" ModelModule : model
    ModelModule "1" *-- "L" FusedMoE : 经 DecoderLayer 持有
    ModelRunner "1" *-- "1" AscendOpsOffloader : moe_offloader
    BaseOffloader <|-- AscendOpsOffloader
    AscendOpsOffloader "1" *-- "1" PrefillOnlyExecution : execution_mode
    AscendOpsOffloader "1" *-- "L" ExpertLayerStorage : storages 按 layer_id 索引
    AscendOpsOffloader "1" *-- "1" AscendPrefillWeights : prefill
    AscendPrefillWeights "1" *-- "1" LayerPrefetchPipeline : pipeline
    AscendOpsOffloader "1" *-- "1" _Context : context 每轮替换
    FusedMoE --> AscendOpsOffloader : _ascend_ops_offloader
    FusedMoE "1" --> "1" ExpertLayerStorage : _moe_offload_storage
    LayerPrefetchPipeline ..> AscendPrefillWeights : load 回调到 _load

    note for BaseOffloader "继承 ABC；图中接口均有默认实现。MoE 专用接口在具体类中扩展。"
    note for ModelModule "概念节点：ModelRunner.model 下的模型模块树。PP 下仅覆盖本 stage。"
    note for AscendPrefillWeights "owner 回指同一个 AscendOpsOffloader；banks 是 execution 的视图。"
    note for PrefillOnlyExecution "owner 作为参数传入；初始化创建的资源挂到 owner 上。"
    note for ExpertLayerStorage "每层一个实例；layer 回指对应 FusedMoE；常驻与冷专家都保留 Host 主副本。"
    note for LayerPrefetchPipeline "在同一个实例上：整轮 begin/finish，逐层 start/acquire/retire。"
```

对象关系中的几个关键约定：

- `owner` 就是 `ModelRunner.moe_offloader` 指向的 `AscendOpsOffloader`。各层的 `_ascend_ops_offloader` 和 `AscendPrefillWeights.owner` 都引用这个对象；并非每层创建一个 offloader。
- `set_offloader()` / `get_offloader()` 提供进程内的对象访问入口。创建专家参数时从该入口取得 owner；接管层后，运行时可以直接沿着层保存的引用取得 owner。[全局访问入口][baseoffloader]
- `prepare_layer()` 创建 storage，并登记 `owner.storages[layer_id]` 和 `layer._ascend_ops_offloader`；storage 自己登记 `layer._moe_offload_storage`。[层接管][preparelayer]、[参数注册][storageparams]
- `PrefillOnlyExecution` 通过 `owner` 操作长期资源；`_Context` 在初始化时创建一次，每轮 forward 再替换，保存本轮计算流、唯一调用标识及异步操作所需引用。实际权重复制流使用 `owner.weight_copy_stream`。[运行资源建立][postinit]、[调用上下文][context]
- `AscendPrefillWeights` 持有 pipeline；pipeline 的 `load` 保存的是绑定方法 `AscendPrefillWeights._load`。调度器决定何时搬哪层，回调决定实际提交哪些 W13/W2 复制计划。[回调注入][prefill]

## 3. UML 类图：模型计算与 offloader 的接入点

`FusedMoE` 与 offloader 之间还经过量化方法、通用 MoE runner 和 NPU runner core。下图保留这些桥接类；虚线上的文字表示调用关系。

```mermaid
classDiagram
    direction TB

    class FusedMoE {
        +forward_impl()
        +run_moe_core()
    }
    class ModelSlimFusedMoEMethod {
        +create_weights()
        +process_weights_after_loading()
        +create_moe_runner()
        +apply()
    }
    class MoeRunner {
        +runner_core
        +run()
    }
    class AscendRunnerCore {
        +run()
        -_run_ffn()
        -_run_split_prefill()
    }
    class AscendOpsDispatcher {
        +dispatch()
        +combine()
    }
    class AscendOpsOffloader {
        +prepare_layer()
        +start_prefill_layer()
        +prepare_weights()
        +wait_for_w2()
    }
    class ExpertLayerStorage {
        +finalize()
    }
    class AscendPrefillWeights {
        +weights()
        +cold_weights()
        +finish_layer()
    }
    class LayerPrefetchPipeline {
        +start()
        +acquire()
        +wait_for_w2()
        +retire()
    }

    FusedMoE --> ModelSlimFusedMoEMethod : quant_method
    FusedMoE *-- AscendOpsDispatcher : dispatcher
    ModelSlimFusedMoEMethod *-- MoeRunner : runner
    FusedMoE --> MoeRunner : runner 的别名引用
    MoeRunner *-- AscendRunnerCore : runner_core
    ModelSlimFusedMoEMethod ..> AscendOpsOffloader : 创建参数时 prepare_layer
    ModelSlimFusedMoEMethod ..> ExpertLayerStorage : 后处理时 finalize
    AscendOpsDispatcher ..> AscendOpsOffloader : dispatch 入口 start_prefill_layer
    AscendOpsDispatcher ..> AscendPrefillWeights : combine 提交后 finish_layer
    AscendRunnerCore ..> AscendOpsOffloader : prepare_weights 和 wait_for_w2
    AscendRunnerCore ..> AscendPrefillWeights : split 分支 cold_weights
    AscendOpsOffloader ..> AscendPrefillWeights : prepare_weights 转调 weights
    AscendOpsOffloader ..> LayerPrefetchPipeline : start 和 wait_for_w2
    AscendPrefillWeights ..> LayerPrefetchPipeline : acquire 和 retire

    note for ModelSlimFusedMoEMethod "加载期负责参数接管与后处理桥接；推理期 apply 调用 runner.run。"
    note for MoeRunner "把 dispatcher 输出转换成 runner 输入；调用 core 后再转换成 combine 输入。"
    note for AscendRunnerCore "从 owner 取得 NPU quant_info，执行 W13、激活、W2，并插入权重等待。"
    note for AscendOpsDispatcher "直接在 dispatch/combine 中调用 offloader；这里没有注册 DecoderLayer 的 forward hook。"
```

源码定位：[ModelSlim 接入][modelslim]、[FusedMoE 的 runner/dispatcher 创建][fusedcreate]、[MoeRunner 的前后转换及 core 调用][moerunner]、[AscendRunnerCore][runnercore]、[AscendOpsDispatcher][dispatcher]。

其他资源类的边界如下：

| 类 | 使用关系与职责 |
|---|---|
| `NpuExpertFormat` / `NpuV4W8A8Format` | 由 owner 持有；storage 调用它写入和转换 checkpoint；执行策略调用它分配符合运行布局的 NPU bank。负责格式和物理字节布局。[格式实现][format] |
| `ExpertBank` | `owner.execution` 是根 bank，持有 NPU 张量；`view()` 创建共享底层分配的槽位视图；`quant_info()` 将这些张量组织成计算接口需要的对象。[bank 实现][bank] |
| `AclHostTransferOps` | 由 `AscendPrefillWeights` 创建；依据 `HostCopyPlan` 提交复制流上的字节搬运，返回包含完成事件的 `TransferTicket`。[搬运提交][transferops] |
| `_Context` | 由 owner 持有，标识一次调用并保留异步资源引用；层结束时记录计算完成事件，整轮结束同步后清理这些引用。[上下文实现][context] |

## 4. 初始化与加载期调用关系

下面以常规 `DefaultModelLoader` 路径为例，省略模型架构相关的中间构造函数。

```text
TpModelWorker._init_model_runner()
└─ ModelRunner.__init__()
   ├─ 创建 AscendOpsOffloader
   │  └─ 创建 PrefillOnlyExecution，建立专家目录并计算槽位需求
   ├─ set_offloader(moe_offloader)
   └─ initialize() → load_model()
      ├─ loader 创建模型
      │  └─ FusedMoE.__init__()
      │     ├─ ModelSlimFusedMoEMethod.create_weights()
      │     │  └─ owner.prepare_layer()
      │     │     └─ 创建 ExpertLayerStorage，注册 Host 参数及加载回调
      │     ├─ create_moe_runner() → MoeRunner → AscendRunnerCore
      │     └─ 创建 dispatcher
      ├─ loader 加载 checkpoint
      │  └─ model.load_weights()
      │     └─ param.weight_loader()
      │        └─ storage 的加载回调 → format.write_checkpoint()
      ├─ loader 调用量化后处理
      │  └─ ModelSlimFusedMoEMethod.process_weights_after_loading()
      │     └─ storage.finalize() → 逐专家格式转换、封存 Host 主副本
      └─ owner.post_init()
         ├─ 检查所有受管理层的 storage 已封存
         └─ PrefillOnlyExecution.post_init(owner)
            ├─ 分配 owner.execution，创建权重复制流及 context
            ├─ 创建 AscendPrefillWeights
            │  ├─ 建立视图并复制常驻专家到 NPU
            │  ├─ 建立冷专家 W13/W2 复制计划
            │  └─ 创建 LayerPrefetchPipeline，注入 _load 回调
            └─ 设置 owner.ready = True
```

`ExpertLayerStorage` 加载的是本 rank 负责的全部专家；每层的常驻比例只决定 NPU 放置和后续复制范围。初始化末尾常驻专家已经到位，冷专家计划已准备；每轮冷专家的首次 H2D 由 `begin_forward()` 启动。加载后的通用设备 staging 会跳过这些 storage 管理的模块，转换由 storage 自己控制。[加载器后处理][loaderpost]、[跳过通用 staging][loadingcontext]

### 初始化时序图：A → B → C → D

下面四张图连续展开同一次初始化，各图的编号独立。A 创建管理对象并进入模型加载；B 构造所有模型层；C 加载并封存权重；D 建立可供 Prefill 使用的 NPU 资源。调用栈在图之间延续，并不是四次独立初始化。图中展示成功路径；校验或转换失败会抛出异常，无法推进到 `ready=True`。

图中的实线箭头表示调用、创建或明确标注的状态写入，虚线箭头表示返回；`__init__()` 箭头标明对象真正创建的时刻。一个带“各层”标记的生命线代表循环中每层各自的实例。`BaseOffloader` 是 `AscendOpsOffloader` 的基类，没有额外创建的基类实例，因此不单列生命线；初始化期间调用的是具体类的 `prepare_layer()`、`post_init()` 等实现。

#### A. 创建 worker、runner、offloader 和执行策略

```mermaid
sequenceDiagram
    autonumber
    participant Worker as TpModelWorker
    participant Runner as ModelRunner
    participant Owner as AscendOpsOffloader<br/>owner
    participant Mode as PrefillOnlyExecution
    participant Registry as offloader.py<br/>全局访问入口
    participant Loader as DefaultModelLoader
    participant Model as 模型模块树

    Worker->>Runner: _init_model_runner() 中调用 __init__()
    Runner->>Owner: __init__(hf_config, offload_config)
    Owner->>Owner: 读取 AscendOffloadOptions，确认 prefill_only
    Owner->>Mode: __init__(use_deepep, resident_ratio)
    Owner->>Mode: validate_topology(owner)
    Mode-->>Owner: 拓扑校验通过
    Owner->>Owner: _stage_layer_range()，build_expert_catalog()
    Note over Owner,Mode: 目录限定本 stage 的 routed 层和本 EP rank 的专家<br/>prefill_only 将这些 routed 层全部纳入管理
    Owner->>Mode: configure(owner, path)
    Mode->>Mode: prefill_weight_slots(E, L, ratio, budget)
    Mode-->>Owner: 写入 resident_experts 和 execution_slots
    Owner->>Owner: storages={}；format/host_pool=None；ready=False
    Owner-->>Runner: 返回 owner
    Runner->>Registry: set_offloader(owner)
    Runner->>Runner: initialize() → load_model()
    Runner->>Loader: 经 load_model_with_memory_saver() 调用 load_model()
    Loader->>Model: _initialize_model(...) 构建模型
    Note over Loader,Model: 进入阶段 B：逐层创建专家模块<br/>所有层构造完成后，进入阶段 C 加载与后处理
```

这时已经确定 `R` 和执行池所需槽位数 `S = L × R + 2 × (E - R)`，实际 Host 池及 NPU 执行池尚未创建。`set_offloader(owner)` 必须先于模型构造，因为量化方法创建专家参数时会调用 `get_offloader()`。[worker 创建 runner][init-worker-create]、[runner 创建 owner][modelrunner]、[槽位计算][slots]

#### B. 构建模型：每层参数接管与计算组件创建

```mermaid
sequenceDiagram
    autonumber
    participant Model as 模型模块树
    participant Layer as FusedMoE<br/>各层实例
    participant Quant as ModelSlimFusedMoEMethod<br/>各层实例
    participant Owner as AscendOpsOffloader
    participant Mode as PrefillOnlyExecution
    participant Format as NpuV4W8A8Format<br/>NpuExpertFormat
    participant Host as MemFabricHostPool
    participant Storage as ExpertLayerStorage<br/>各层实例
    participant MoERunner as MoeRunner<br/>各层实例
    participant Core as AscendRunnerCore<br/>各层实例
    participant Dispatch as dispatcher<br/>各层实例

    loop 本 stage 的每个受管理专家层
        Model->>Layer: __init__(layer_id, ...)
        Note over Layer,Quant: 已选定 ModelSlim 量化方法及 W13/W2 scheme
        Layer->>Quant: create_weights(layer, num_local_experts, ...)
        Quant->>Quant: get_offloader() 取得阶段 A 登记的 owner
        Quant->>Owner: prepare_layer(layer, ...)
        Owner->>Mode: validate_quantization(owner, quant_spec)
        Mode-->>Owner: 部分常驻路径要求 W8A8
        opt 首个受管理层：owner.format 尚未创建
            Owner->>Owner: get_native_acl() 初始化原生传输支持
            Owner->>Format: __init__(hidden_size, intermediate_size, device, ...)
            Format->>Format: _probe()：用样本张量探测 W13/W2 的运行布局
            Note right of Format: 探测会使用临时 NPU 张量<br/>只保留 schema/layouts，释放探测张量
            Format-->>Owner: 返回格式对象
            Owner->>Owner: 汇总所有受管理层所需 Host 字节，检查预算
            Owner->>Host: __init__(total_bytes, device_index, physical_bytes, ...)
            Host-->>Owner: 返回可异步读取的 Host 池
        end
        Owner->>Owner: 校验各层量化方式和专家形状一致
        Owner->>Storage: __init__(layer, spec, format, host_pool, attrs)
        loop W13/W2 的 weight、scale、offset 六个组件
            Storage->>Host: allocate(host_extent * E, alignment)
            Host-->>Storage: Host allocation 与访问能力信息
            Storage->>Format: checkpoint_view(allocation, component, E)
            Format-->>Storage: checkpoint 形状和类型的 CPU 视图
            Storage->>Storage: 创建 Parameter，绑定 weight_loader=self._loader(name)
            Storage->>Layer: register_parameter(name, parameter)
            Storage->>Storage: 保存每个专家组件的 Host 地址与运行字节数
        end
        Storage->>Layer: 写入 _moe_offload_storage 引用
        Storage-->>Owner: 返回本层 storage，状态为 loading
        Owner->>Owner: storages[layer_id] = storage
        Owner->>Layer: 写入 _ascend_ops_offloader = owner
        Owner-->>Quant: 返回 True，表示参数已接管
        Quant-->>Layer: create_weights() 返回，跳过普通设备参数创建
        Layer->>Quant: create_moe_runner(layer, config)
        Quant->>MoERunner: __init__(ASCEND, config)
        MoERunner->>Core: __init__(config)
        Core-->>MoERunner: runner_core
        MoERunner-->>Quant: runner
        Quant-->>Layer: runner 配置完成
        Layer->>Dispatch: 经 create_moe_dispatcher() 创建 dispatcher
        Dispatch-->>Layer: dispatcher 对象
        Layer->>Layer: runner = quant_method.runner
        Layer-->>Model: 本层构造完成
    end
    Note over Model,Storage: 所有层构造结束后，才进入阶段 C 的 checkpoint 加载<br/>此时 Host 参数已注册；参数内容尚未完成 checkpoint 填充
```

`Format` 和 Host 池在首层创建后供后续层复用；`ExpertLayerStorage`、量化方法、MoE runner、core 和 dispatcher 则按层创建。图中 dispatcher 在 AscendOps 路径对应 `AscendOpsDispatcher`；DeepEP 使用相应后端对象，hook 在阶段 D 才注册。[层接管][preparelayer]、[参数注册][storageparams]、[计算组件创建][fusedcreate]、[量化方法桥接][modelslim]

#### C. 加载 checkpoint，逐专家转换，封存 Host 主副本

```mermaid
sequenceDiagram
    autonumber
    participant Loader as DefaultModelLoader
    participant Model as 模型 load_weights
    participant Quant as ModelSlimFusedMoEMethod<br/>各层实例
    participant Storage as ExpertLayerStorage<br/>各层实例
    participant Format as NpuV4W8A8Format<br/>NpuExpertFormat
    participant Adapter as _TorchNpuW8A8FormatAdapter<br/>含 TorchNpu 基类方法
    participant Dispatch as 本层 dispatcher
    participant Runner as ModelRunner

    Loader->>Loader: load_weights_and_postprocess() → load_weights_only()
    Loader->>Model: load_weights(weights)
    loop 迭代专家 checkpoint 分片，按参数名和 expert_id 分发
        Model->>Storage: param.weight_loader(...) → _loader(name) 返回的回调
        alt expert_id 属于其他 EP rank
            Storage-->>Model: 跳过该分片
        else expert_id 属于本 rank
            Storage->>Storage: 校验 loading 状态、shard_id、重复加载及 CPU 来源
            Storage->>Format: write_checkpoint(name, raw_component[local], weight, shard_id, metadata)
            Format->>Format: 定位 W1/W3 或 W2 分片，校验并 copy_ 到 Host
            Format-->>Storage: 分片写入完成
            Storage->>Storage: 登记已加载组件分片
            Storage-->>Model: 回调完成
        end
    end
    Model->>Model: 等待已提交加载任务完成，执行模型自身 post_load_weights
    Model-->>Loader: model.load_weights() 返回
    Loader->>Loader: postprocess_weights(model, target_device)
    loop 每个受管理 FusedMoE 的量化后处理
        Loader->>Loader: device_loading_context() 检测 storage，跳过通用整模块 staging
        Loader->>Quant: process_weights_after_loading(layer)
        Quant->>Storage: finalize()
        Storage->>Storage: 校验所有本地专家的全部组件分片齐全
        Storage->>Format: validate_master(raw_components)
        loop 本层的每个本地专家，包括常驻和冷专家
            Storage->>Format: repack_into(raw_components, expert, destinations)
            loop W13 组件组、W2 组件组
                Format->>Adapter: to_device()：本专家本投影的组件临时 H2D
                Adapter-->>Format: 临时 NPU 输入
                Format->>Adapter: prepare(weight, scale, offset)
                Adapter-->>Format: 转置后的 NZ weight、BF16 scale、offset
                Format->>Adapter: synchronize()：等待格式转换完成
                Adapter-->>Format: 转换结果可以被原生 memcpy 读取
                Format->>Format: 校验 shape、stride、dtype、物理字节数符合 probe
                Format->>Adapter: copy_to_host()：原生物理字节 D2H 写回 Host 主副本
                Adapter-->>Format: 本投影写回完成
                Format->>Format: 释放本次转换临时张量
            end
            Format-->>Storage: 本专家转换完成
        end
        Storage->>Dispatch: set_quant_config(schema.dispatch_config)
        Storage->>Storage: 状态改为 sealed，禁止 weight_loader 继续写入
        Storage->>Storage: 清理加载记录和 raw_components 视图集合
        Storage-->>Quant: finalize() 返回
        Quant->>Dispatch: 经 w13_kernel._set_moe_dispatch_dtype() 设置 INT8 约定
        Quant-->>Loader: 本层后处理完成
    end
    Loader->>Loader: model.eval()
    Loader-->>Runner: 经加载包装函数返回已加载的模型
    Note over Storage,Runner: 所有受管理层的 Host 主副本都已封存<br/>NPU 常驻区和冷 bank 的建立接续阶段 D
```

加载回调允许由线程池调度；图中表示每个分片的逻辑调用链，以及“加载全部完成后才后处理”的先后关系，不限定线程池内部的执行顺序。`finalize()` 清理的是加载期记录与视图集合，Host 主副本仍由参数、存储能力对象和池的引用保留。转换后 Host 存的是运行布局的字节，推理时预取无需再做 NZ 转换。[加载器顺序][init-default-loader]、[模型加载回调][init-weight-callback]、[逐专家转换][init-repack]、[W8A8 格式处理][init-w8-adapter]

#### D. 校验封存状态，创建执行池，填充常驻专家，准备冷专家流水线

```mermaid
sequenceDiagram
    autonumber
    participant Runner as ModelRunner
    participant Owner as AscendOpsOffloader
    participant Storage as ExpertLayerStorage<br/>各层实例
    participant Mode as PrefillOnlyExecution
    participant Format as NpuV4W8A8Format<br/>NpuExpertFormat
    participant Bank as ExpertBank<br/>根池和槽位视图
    participant Context as _Context
    participant Prefill as AscendPrefillWeights
    participant Transfer as AclHostTransferOps
    participant Stream as weight_copy_stream<br/>torch.npu.Stream
    participant Pipeline as LayerPrefetchPipeline
    participant Dispatch as 本层 dispatcher

    Runner->>Runner: 保存 model/loader，执行平台 post_load_model()
    Runner->>Owner: get_offloader().post_init()
    Owner->>Owner: 检查 storages 覆盖全部计划中的层
    loop 每个受管理 storage
        Owner->>Storage: require_sealed()
        Storage-->>Owner: 已封存
    end
    Owner->>Mode: post_init(owner)
    Mode->>Owner: 写入 hot=None、state=None 等 Prefill-only 状态
    Mode->>Format: allocate_bank(S, EXECUTION_WEIGHTS, ...)
    Format->>Format: adapter.empty_component() 分配 NZ 权重与 ND 元数据
    Format->>Bank: __init__(format, tensors, S, ...)
    Bank-->>Format: 根执行池
    Format-->>Mode: ExpertBank
    Mode->>Owner: execution = 根执行池，创建设备 error 标志
    Mode->>Stream: __init__(device)
    Stream-->>Mode: 权重复制流
    Mode->>Owner: 保存 weight_copy_stream，并调用 _new_context()
    Owner->>Context: __init__()，记录当前计算流和调用身份
    Context-->>Owner: context
    Owner-->>Mode: 返回新 context
    Mode->>Owner: 保存 owner.context
    Mode->>Prefill: __init__(owner, resident_experts=R)
    Prefill->>Prefill: 分配 counts，计算 resident_slots，按池大小决定 split_gmm
    Note over Prefill,Bank: 本文 R 大于 0；S 大于 1024 时启用 split_gmm
    Prefill->>Bank: view(L*R + b*C, C)，建立两个冷 bank
    Bank-->>Prefill: 共享底层分配的冷 bank 视图
    alt split_gmm=True
        loop 每个受管理层 i
            Prefill->>Bank: view(i*R, R).quant_info()，保存常驻权重视图
        end
        Prefill->>Bank: 两个冷 bank 分别调用 quant_info()
    else type2 路径
        Prefill->>Bank: execution.quant_info()，取得整个池的权重视图
        Prefill->>Prefill: 创建物理槽位分组表 groups
    end
    Prefill->>Transfer: __init__(native_acl, owner._retain)
    Transfer-->>Prefill: host_ops
    loop 每个受管理层 i：_populate_resident()
        Prefill->>Bank: view(i * R, R) 获取本层固定常驻区
        Bank-->>Prefill: 常驻区视图
        Prefill->>Storage: 读取本层 component_capabilities 的前 R 个专家
        Prefill->>Prefill: _projection_copy_plans(..., slice(R))
        loop 常驻 W13 计划、常驻 W2 计划
            Prefill->>Transfer: materialize_into(plan, resident_bank, stream)
            Transfer->>Context: 经 owner._retain() 保留计划、bank 和描述符
            Transfer->>Stream: 排队提交 ACL 复制，记录完成事件
            Transfer-->>Prefill: TransferTicket，表示已提交
        end
    end
    Prefill->>Stream: synchronize()
    Stream-->>Prefill: 所有层的常驻 H2D 完成
    Prefill->>Transfer: clear_cache() 清理常驻填充使用的描述符缓存
    Prefill->>Context: resources.clear() 释放本次填充的临时保留引用
    loop 每个受管理层 i：准备冷专家计划
        Prefill->>Storage: 读取 component_capabilities 的 [R, E) 范围
        Prefill->>Prefill: 建立源 Host 到 cold_bank[i % 2] 的 W13/W2 计划
        Prefill->>Transfer: prepare(w13_plan, device)，prepare(w2_plan, device)
        Transfer-->>Prefill: 已缓存复制描述符，此时没有提交冷专家 H2D
    end
    Prefill->>Pipeline: __init__(layer_ids, load=self._load, copy_stream)
    Pipeline->>Pipeline: reset()：current=0，started=False，ready={}
    Pipeline-->>Prefill: 预取调度器已建立
    Prefill-->>Mode: 返回 Prefill 权重供给对象
    Mode->>Owner: 保存 owner.prefill
    opt 使用 DeepEP
        loop 每个受管理层的 dispatcher
            Mode->>Dispatch: register_pre_dispatch_hook(start)
            Mode->>Dispatch: register_post_combine_hook(finish)
        end
    end
    Mode->>Owner: ready = True
    Mode-->>Owner: Prefill-only post_init 完成
    Owner-->>Runner: 专家卸载资源初始化完成
    Note over Runner,Owner: ModelRunner 继续完成其他初始化，最终返回 TpModelWorker<br/>ready=True 表示 offloader 资源就绪，不代表整套服务已完成启动
    Note over Prefill,Pipeline: 尚未调用 begin_forward()，冷 bank 没有有效的本轮预取内容<br/>首次模型 forward 才通过 _load() 提交前两层冷专家 H2D
```

图 D 中 `C = E - R`，`b` 取 0 或 1。先决定 `split_gmm` 并建立权重视图，再创建 `host_ops` 和填充常驻区。`host_ops.prepare()` 只准备描述符，`materialize_into()` 才提交搬运。[运行资源建立][postinit]、[Prefill 构造与常驻填充][prefill]、[描述符准备][init-copy-prepare]、[复制提交][transferops]、[DeepEP hook][deepephooks]

#### 初始化完成时应具备的状态

| 对象 | 完成状态 | 后续首次 forward 的行为 |
|---|---|---|
| `AscendOpsOffloader` | `ready=True`；目录、storages、format、execution、复制流、prefill 均已建立。 | `forward_scope()` 创建新的本轮 context。 |
| 各层 `ExpertLayerStorage` | `sealed`；全部本地专家的 Host 主副本都已转换为运行布局，加载回调禁止再写。 | 作为异步复制源保留。 |
| `owner.execution` | 固定常驻区已填充；两个冷 bank 已分配。 | 冷 bank 按层交替填充和复用。 |
| `AscendPrefillWeights` | 权重视图、分组元数据、各层冷专家 W13/W2 计划及描述符均已准备。 | `_load()` 将相应计划提交到复制流。 |
| `LayerPrefetchPipeline` | `invocation_id=None`、`current=0`、`started=False`、`ready={}`。 | `begin_forward()` 登记本轮身份并提交前两层预取。 |
| `FusedMoE` 及计算组件 | storage/owner 引用、量化方法、runner/core、dispatcher 已建立；dispatcher 的 INT8 约定已设置。 | 进入 `dispatch → runner → combine`，触发逐层 offload 接入点。 |

## 5. 一次 Prefill forward 的调用关系

模型级 `forward_scope` 和逐层的 `start/acquire/retire` 操作同一个 pipeline、同一个本轮 context。`yield` 期间执行模型并触发逐层接入点。

```text
TpModelWorker.forward_batch_generation()
└─ ModelRunner.forward()
   └─ with owner.forward_scope(forward_batch):
      ├─ PrefillOnlyExecution.forward_scope(owner, batch) 进入
      │  ├─ 创建本轮 context
      │  └─ pipeline.begin_forward(context)
      │     └─ 提交前两个受管理 MoE 层的冷专家搬运
      ├─ yield：执行 ModelRunner._forward_raw()
      │  └─ 模型逐层执行，进入 FusedMoE.forward_impl()
      │     ├─ dispatcher.dispatch()
      │     │  └─ owner.start_prefill_layer() → pipeline.start()
      │     ├─ run_moe_core()
      │     │  └─ ModelSlimFusedMoEMethod.apply() → MoeRunner.run()
      │     │     └─ AscendRunnerCore.run()
      │     │        ├─ owner.prepare_weights() → prefill.weights()
      │     │        └─ _run_ffn() 或 _run_split_prefill()
      │     │           └─ 在使用冷 W13/W2 前建立相应事件依赖
      │     └─ dispatcher.combine()
      │        └─ 合并提交后 prefill.finish_layer()
      │           └─ 记录计算完成事件 → pipeline.retire()
      │              └─ 复制流等待旧读者完成后，在同一 bank 装入第 i+2 层
      ├─ yield 正常返回后：pipeline.finish_forward()、owner._check_error()
      └─ finally：等待计算/复制完成、清理本轮资源引用、pipeline.reset()
```

这里的 `i` 是受管理 MoE 层列表中的序号，可能与模型的绝对 `layer_id` 不同。常驻专家始终使用各层固定槽位；两个冷 bank 按 `i % 2` 轮转。

| 操作 | 触发位置 | 实际作用 |
|---|---|---|
| `begin_forward()` | 模型 forward 入口 | 建立本轮身份与初始流依赖，提交前两层预取。 |
| `start()` | 每层 dispatch 入口 | 校验层顺序和调用身份，设置 `started=True`；此处没有新的 H2D 提交，也不等待权重。 |
| `acquire()` | 使用冷 W13 前 | 返回 bank 序号，并让计算流等待本层 W13 复制完成事件。 |
| `wait_for_w2()` | W13/激活之后、冷 W2 计算之前 | 让计算流等待本层 W2 复制完成事件。 |
| `retire()` | 每层 combine 提交后 | 用计算完成事件约束 bank 复用，提交第 `i+2` 层，推进当前层序号。 |
| `finish_forward()` | 模型 forward 正常返回后 | 检查本轮受管理层是否全部消费；资源排空及重置在外层 `finally` 中完成。 |

`weights()` 的返回与 W13 等待有两种安排：

- `split_gmm=False`：`weights()` 内调用 `pipeline.acquire()`，建立冷 W13 依赖后返回权重视图；随后执行 FFN，并在 W2 前单独等待。
- `split_gmm=True`：`weights()` 先返回常驻权重；runner 先提交有 token 的常驻部分，再通过 `cold_weights() → acquire()` 建立冷 W13 依赖，执行冷专家部分并在冷 W2 前等待。空 token 分组会跳过对应计算。

等待的底层实现是 `compute_stream.wait_event(event)`，用于建立设备流依赖；整轮 `finally` 中的 `synchronize()` 才负责等待排队任务结束后清理本轮引用。Host 主副本、NPU 权重池和常驻权重跨 forward 保留。[权重返回与冷权重接口][prefillweights]、[runner 计算及等待位置][runnercore]、[事件等待][eventwait]、[forward_scope][scope]

### DeepEP 的接入差异

AscendOps 在 `dispatch()` 和 `combine()` 中直接调用 owner。DeepEP 在 `PrefillOnlyExecution.post_init()` 时，为每个受管理专家模块的 dispatcher 注册：

```python
layer.dispatcher.register_pre_dispatch_hook(start)
layer.dispatcher.register_post_combine_hook(finish)
```

回调分别调用 `owner.start_prefill_layer(layer_id)` 和 `owner.prefill.finish_layer(layer_id)`，共同使用上面的流水线。这里注册的是 dispatcher hook；这套 offload 调度没有通过给整个 DecoderLayer 注册 `register_forward_pre_hook/register_forward_hook` 实现。[DeepEP hook 注册][deepephooks]

## 6. 部分常驻对应的资源归属

令每层本地专家数为 `E`，常驻数 `R = int(E × prefill_resident_ratio)`，冷专家数 `C = E - R`，受管理层数为 `L`。当前实现按本地专家编号选取前 `R` 个固定常驻，执行池槽位数为：

```text
S = L × R + 2 × C

owner.storages[layer_id]：每层保留 E 个专家的 Host 主副本

owner.execution：
[第0层常驻 R] ... [第L-1层常驻 R] [冷 bank 0：C] [冷 bank 1：C]

owner.prefill：维护上述槽位的视图、复制计划和计算分组信息
owner.prefill.pipeline：维护本轮哪些层可以读、何时可以复用冷 bank
```

Host 权重的加载、封存属于 `ExpertLayerStorage`；NPU 大权重池的建立由 `PrefillOnlyExecution` 调用 format 完成并挂在 owner 上；视图及搬运计划属于 `AscendPrefillWeights`；复用时序属于 `LayerPrefetchPipeline`。这种分工也区分了“数据存在哪里”和“何时允许读取/覆盖”。[槽位计算][slots]、[资源建立][postinit]、[常驻与冷 bank 视图][prefill]

## 7. type2 前后：token 不换队，改成按槽位找权重

本节比较的是**同一次 forward 中，进入 `owner.prepare_weights()` 之前，与它返回之后**的输入，不是两个版本的程序。范围为部分常驻、`split_gmm=False`，即实际 NPU 权重池槽位数 `S <= 1024` 的 type2 路径。以 AscendOps 的 alltoallv Prefill 和非空接收为例；全空接收会跳过 FFN，超过槽位上限的 split 路径见第 5 节。

大白话理解：dispatch 和 pre-permute 已经让 token 按“交给哪个逻辑专家”排好队。type1 的计数表说“专家 0 吃几行、专家 1 吃几行……”。部分卸载后，本层的常驻专家与冷专家放在大权重池的不同位置，type2 就把这张表改为“接下来几行，去物理槽位几找权重”。**排队顺序保持，找权重的地址变得明确。**

### 7.1 输入到底变了什么

令 `M` 为本 rank 路由后参与当前 MoE 的输入行数，`E` 为每层本地专家数，`S` 为整个执行池的槽位数。`M` 包含 top-k 路由展开后的行，同一个原始 token 可以贡献多行。

| 输入 | `prepare_weights()` 前 | 返回后，进入 GMM | 这一步做了什么 |
|---|---|---|---|
| `hidden_states` | 已按本层逻辑专家分组的 `M` 行 | 同一个字段、同样的行顺序 | 没有因 type2 再搬一次或重排 token。 |
| `hidden_states_scale` | 与输入行对应的 scale（若有） | 保持对应关系 | 没有因 type2 重排 scale。 |
| `expert_tokens` | `[E]` 的每专家 token 数 | `[S, 2]` 的槽位/行数对 | 把逻辑专家映射到物理槽位，按原专家顺序紧凑写入有 token 的分组；其余项补零。 |
| `group_list_type` | `1` | `2` | 告诉 GMM 如何解释上面的表。 |
| `quant_info` | runner 原本收到的、由 `layer` 属性组装的对象 | owner 提供的 NPU **全池**权重视图，各权重组件的专家维度为 `S` | runner 用返回值替换引用；这里没有把常驻权重和冷权重拼成新 tensor。 |
| 激活步骤需要的分组信息 | 逻辑专家计数 | `owner.prefill.counts`，形状仍为 `[E]`，type1 | 独立保留逻辑计数；需要分组信息的激活核不使用物理槽位表。 |

这里的 `quant_info` 是权重及量化参数的引用集合，不是把所有组件装进一个 tensor。W13、W2 及各自的 scale/offset 使用不同组件 tensor，但共享同一套物理槽位编号。

```mermaid
flowchart LR
    subgraph before[进入 prepare_weights 之前]
        X0["hidden_states<br/>已经排好的 M 行 token"]
        Q0["hidden_states_scale<br/>与 token 对应"]
        G0["expert_tokens: E 个计数<br/>group_list_type = 1"]
    end
    P["owner.prepare_weights<br/>→ prefill.weights<br/>→ prepare_prefill_counts"]
    subgraph after[进入 GMM 的输入]
        X1["原来的 token 行<br/>顺序不变"]
        Q1["原来的 scale<br/>对应关系不变"]
        G1["expert_tokens: S × 2<br/>每对是 槽位号, 行数<br/>group_list_type = 2"]
        W1["quant_info<br/>NPU 全池权重视图"]
    end
    Pool["owner.execution<br/>各层常驻区 + 两个冷 bank"]
    Logical["owner.prefill.counts<br/>保留 E 个逻辑计数<br/>供需要分组信息的激活核使用"]
    X0 -->|保持| X1
    Q0 -->|保持| Q1
    G0 --> P
    P -->|写分组元数据| G1
    P --> Logical
    Pool -->|预建视图，经 prepare_weights 返回| W1
    X1 --> GMM["GMM1：按槽位读 W13<br/>后续 GMM2 沿用相同槽位分组"]
    Q1 --> GMM
    G1 --> GMM
    W1 --> GMM
    classDef same fill:#e6f4ea,stroke:#34834b,color:#173b24;
    classDef meta fill:#fff3db,stroke:#b78019,color:#4c3400;
    classDef weight fill:#e7f0ff,stroke:#4775b5,color:#17345e;
    class X0,X1,Q0,Q1 same;
    class G0,G1,P,Logical meta;
    class Pool,W1 weight;
```

图中箭头表示字段、元数据和引用的传递，不是权重 H2D；冷权重的实际搬运见第 8 节。GMM2 接收的是 GMM1/激活产生的新数值及 scale，但沿用相同的槽位分组与行对应顺序。

对应源码的关键语句是：

```python
# AscendRunnerCore.run(): 返回值替换 quant_info；runner_input 的分组字段被更新
quant_info = owner.prepare_weights(self.config.layer.layer_id, runner_input)
x = runner_input.hidden_states
expert_tokens = runner_input.expert_tokens
group_list_type = runner_input.group_list_type

# AscendPrefillWeights.weights(): type2 分支选择并最终写回这两个字段
group_tokens, group_type = self.groups, 2
runner_input.expert_tokens, runner_input.group_list_type = group_tokens, group_type
```

注意：`quant_info` 不是 `prepare_weights()` 的输入参数；它是返回值，由 runner 接住。进入这个方法前的 token 分组来自 [alltoallv pre-permute][type2-input]；原始权重引用来自 [ModelSlim.apply][type2-original-quant]；转换及替换在 [prefill.weights][prefillweights] 和 [AscendRunnerCore.run][type2-runner]；最终参数透传到 [npu_grouped_matmul][type2-gmm]。

### 7.2 一个能逐行对上的小例子

以下是**按代码规则构造的示意规模，不是模型的实际配置或实测数据**：3 个受管理 MoE 层，每层 4 个本地专家，前 2 个常驻、后 2 个为冷专家。于是 `L=3, E=4, R=2, C=2, S=3×2+2×2=10`。当前执行第 2 个受管理层，即序号 `i=1`。

| 物理槽位 | 长期用途 | 当前层 `i=1` 如何使用 |
|---|---|---|
| `0, 1` | 第 0 层的常驻专家 0、1 | 不读。 |
| `2, 3` | 第 1 层的常驻专家 0、1 | 专家 0 读槽位 2；本例专家 1 没有 token。 |
| `4, 5` | 第 2 层的常驻专家 0、1 | 不读。 |
| `6, 7` | 冷 bank 0，轮流放偶数序号层的冷专家 | 本层不读；可用于第 2 层预取。 |
| `8, 9` | 冷 bank 1，轮流放奇数序号层的冷专家 | 本层冷专家 2、3 分别读槽位 8、9。 |

假设本层逻辑计数是 `[2, 0, 3, 1]`，输入的 6 行已经是 `[a, b, c, d, e, f]`：

```mermaid
flowchart LR
    subgraph rows[输入行保持原顺序]
        AB["第 0～1 行：a, b<br/>逻辑专家 0，共 2 行"]
        Empty["逻辑专家 1：0 行<br/>没有输入行"]
        CDE["第 2～4 行：c, d, e<br/>逻辑专家 2，共 3 行"]
        F["第 5 行：f<br/>逻辑专家 3，共 1 行"]
    end
    subgraph pairs[type2 表的有效前缀]
        Pair0["第 0 对：[2, 2]"]
        Pair1["第 1 对：[8, 3]"]
        Pair2["第 2 对：[9, 1]"]
    end
    subgraph slots[NPU 全池中的实际位置]
        Slot2["槽位 2：本层专家 0<br/>固定常驻"]
        Slot8["槽位 8：本层专家 2<br/>冷 bank 1"]
        Slot9["槽位 9：本层专家 3<br/>冷 bank 1"]
    end
    AB --> Pair0 --> Slot2
    CDE --> Pair1 --> Slot8
    F --> Pair2 --> Slot9
    Empty -.-> Skip["不生成有效分组"]
    classDef resident fill:#e6f4ea,stroke:#34834b,color:#173b24;
    classDef cold fill:#e7f0ff,stroke:#4775b5,color:#17345e;
    classDef meta fill:#fff3db,stroke:#b78019,color:#4c3400;
    class Slot2 resident;
    class Slot8,Slot9 cold;
    class Pair0,Pair1,Pair2 meta;
```

完整的分组 tensor 是：

```text
type1: expert_tokens = [2, 0, 3, 1]               # shape [4]

type2: expert_tokens = [[2, 2],                   # 前 2 行用槽位 2
                       [8, 3],                   # 接着 3 行用槽位 8
                       [9, 1],                   # 接着 1 行用槽位 9
                       [0, 0], [0, 0], [0, 0],
                       [0, 0], [0, 0], [0, 0],
                       [0, 0]]                   # shape [10, 2]，剩余 7 对为零

hidden_states       = [a, b, c, d, e, f]          # 前后相同；6 行不变成 10 行
hidden_states_scale = [sa, sb, sc, sd, se, sf]    # 有逐行 scale 时，仍一一对应
```

表的**第几行不是槽位号**，第一列里的数才是槽位号。`[8, 3]` 的意思是“下一段 3 行 token 用槽位 8”，不是“第 8 个 token 用专家 3”。`[0, 0]` 表示零行，不会要求槽位 0 参与当前计算。

源码中的映射公式为：

```text
逻辑专家 e < R：slot = i × R + e
逻辑专家 e ≥ R：slot = L × R + (i % 2) × C + (e - R)

本例：e = [0, 1, 2, 3] → slot = [2, 3, 8, 9]
      count = [2, 0, 3, 1] → 有效分组 = [[2, 2], [8, 3], [9, 1]]
```

`prepare_prefill_counts` 用 `count > 0` 选择有 token 的专家，并按逻辑专家顺序压紧分组表；它没有压紧或重排 token tensor。源码：[槽位映射及分组表生成][type2-mapping]。

`execution_slots` 因而表示 NPU 常驻区和两个冷 bank 的物理槽位容量。每个组件本来就在执行池的规范布局里；type2 让 GMM 直接寻址相隔的槽位，无需再把“本层常驻 + 本层冷专家”复制到一块临时连续权重区。**冷专家从 Host 到冷 bank 的 H2D 仍然存在**，type2 省掉的是这类额外的权重拼装需求。全池视图的建立见 [AscendPrefillWeights.__init__][type2-pool]。

另一个边界是：**稀疏计算不等于按本轮 token 挑选冷专家预取。** 当前复制计划在初始化时就按 `slice(R, None)` 建好，覆盖本层全部 `C` 个冷专家。即使某个冷专家本轮没有 token，它仍在该层的预取计划里，只是不产生有效 GMM 分组。[冷专家计划建立][copy-plans-init]

## 8. 一次 Prefill forward：copy 从哪里发起，在哪里等待

沿用第 7 节的 3 层示例。下图按真实源码调用顺序展开一次正常、非空的 Prefill forward；它是**源码流程示意，不是 NPU profiler 的实测时间线**。层号 `i=0,1,2` 均为受管理层的序号，实际函数参数传的是相应 `layer_id`。

### 8.1 从整轮 forward 到每层 GMM 的控制流

图中的“提交等待”指往设备流排入依赖，不能读成“Python 在这里等到复制完成才返回”。模型自己推进各层，pipeline 只负责权重供给和复用顺序。

```mermaid
flowchart TB
    Worker["TpModelWorker.forward_batch_generation<br/>→ ModelRunner.forward"]
    Scope["with owner.forward_scope(batch)<br/>→ PrefillOnlyExecution.forward_scope<br/>创建本轮 context"]
    Begin["pipeline.begin_forward<br/>① 计算流记录入口事件，复制流等待它<br/>② _submit(0) 和 _submit(1)<br/>提交前两层冷 W13 / W2"]
    Raw["yield → ModelRunner._forward_raw<br/>执行模型；从第 0 个受管理层开始"]
    Layer["模型推进到本层 MoE<br/>此前可以执行 attention / router 等"]
    Dispatch["FusedMoE.forward_impl → dispatcher.dispatch<br/>owner.start_prefill_layer → pipeline.start<br/>校验层顺序；随后路由 / 通信"]
    Runner["run_moe_core → ModelSlim.apply → MoeRunner.run<br/>pre-permute：整理为逻辑专家顺序<br/>→ AscendRunnerCore.run"]
    Prepare["owner.prepare_weights → prefill.weights<br/>写 type2 分组表；准备全池权重引用"]
    Wait13["pipeline.acquire<br/>计算流提交 wait_event：本层冷 W13 就绪"]
    GMM1["_run_ffn → w13_kernel.apply<br/>GMM1 读取本层常驻和冷 W13"]
    Act["SwiGLU / 激活及量化<br/>需要分组时使用逻辑计数"]
    Wait2["owner.wait_for_w2 → pipeline.wait_for_w2<br/>计算流提交 wait_event：本层冷 W2 就绪"]
    GMM2["w2_kernel.apply<br/>GMM2 读取本层常驻和冷 W2"]
    Combine["post-permute → dispatcher.combine<br/>提交返回通信及输出整理"]
    Retire["prefill.finish_layer<br/>context.record_compute_completion：记录 D_i<br/>→ pipeline.retire"]
    Refill{"还有第 i+2 个<br/>受管理层吗？"}
    Submit["_submit(i+2, reader=D_i)<br/>复制流先提交 wait_event(D_i)<br/>再向同一冷 bank 提交未来层 W13 / W2"]
    Advance["retire 完成：删除当前票据<br/>current += 1；started = False"]
    More{"还有受管理层吗？"}
    Rest["模型完成剩余计算；_forward_raw 返回<br/>finish_forward 校验层均已消费<br/>owner._check_error"]
    Drain["finally：计算流 wait_stream(复制流)<br/>记录最终事件并在 CPU synchronize<br/>清理本轮引用；pipeline.reset"]
    Worker --> Scope --> Begin --> Raw --> Layer --> Dispatch --> Runner
    Runner --> Prepare --> Wait13 --> GMM1 --> Act --> Wait2 --> GMM2 --> Combine
    Combine --> Retire --> Refill
    Refill -->|有| Submit --> Advance
    Refill -->|无| Advance
    Advance --> More
    More -->|有，模型继续下一层| Layer
    More -->|无| Rest --> Drain
    classDef copy fill:#e7f0ff,stroke:#4775b5,color:#17345e;
    classDef wait fill:#fff3db,stroke:#b78019,color:#4c3400;
    classDef compute fill:#e6f4ea,stroke:#34834b,color:#173b24;
    class Begin,Submit copy;
    class Wait13,Wait2,Retire,Drain wait;
    class GMM1,Act,GMM2 compute;
```

三个容易混淆的点：

- `begin_forward()` 在模型执行前就提交前两层；`start_prefill_layer()` 在逐层 dispatch 入口只登记和校验状态，不在那里重新搬权重，也不在那里等待权重。
- 当前 type2 分支把本层常驻和冷专家交给同一次 GMM1，因此它在 GMM1 前统一等待本层冷 W13；不会先单独计算常驻专家。先算常驻再算冷专家是 `split_gmm=True` 的另一条路径。
- `finish_layer()` 发生在 combine 已经提交之后。此时 CPU 可以继续提交未来层复制，复制流里的 `wait_event(D_i)` 保证旧层真正执行到该事件以后，才覆盖它刚用过的冷 bank。

对应入口：[ModelRunner.forward_scope 包裹点][copy-model-scope]、[策略 forward_scope][scope]、[dispatcher start / finish 接入][dispatcher]、[runner 的 GMM 与 W2 等待][copy-ffn]、[pipeline 的 begin / acquire / retire][pipeline]。

### 8.2 发起 H2D 的调用栈：计划、批量提交、完成事件

每层有两份复制计划，先 W13，后 W2。每份计划包含这个投影的权重及量化组件。计划的源是该层 Host 主副本，目的地是 `bank = i % 2` 对应的冷 bank；常驻部分在初始化时已填好，本轮不重新搬运。

```mermaid
flowchart TB
    EntryA["forward_scope → pipeline.begin_forward<br/>首次提交 i=0、i=1"]
    EntryB["dispatcher.combine → prefill.finish_layer<br/>→ pipeline.retire<br/>后续提交 i+2，附带旧层完成事件 D_i"]
    Submit["LayerPrefetchPipeline._submit(index, context, reader)<br/>若有 reader：复制流先排入 wait_event"]
    Load["self.load(...)<br/>→ AscendPrefillWeights._load(layer, bank, context)"]
    Phase["依次处理两个 plan：W13 → W2<br/>每份 plan 各自走一遍下面的提交过程"]
    Materialize["AclHostTransferOps.materialize_into<br/>→ _materialize_into<br/>取得已准备好的批次，保留源 / 目标 / 描述符引用"]
    Stream["with torch.npu.stream(weight_copy_stream)<br/>for batch in batches: module.submit(batch, stream_id)"]
    Native["C++ submit<br/>OpCommand::RunOpApi<br/>将 ACL 提交排入 torch_npu 队列"]
    ACL["Batch::submit<br/>→ aclrtMemcpyBatchAsync<br/>提交该批的地址 / 长度数组"]
    Event["当前投影的所有批次提交后<br/>_ticket → _npu_completion<br/>在复制流 record(Event)，返回 TransferTicket"]
    Ready["两份计划都完成提交<br/>返回 PrefillLayerTransfer(w13_ticket, w2_ticket)<br/>保存为 pipeline.ready[index]"]
    EntryA --> Submit
    EntryB --> Submit
    Submit --> Load --> Phase --> Materialize --> Stream --> Native --> ACL --> Event
    Event -->|W13 后，继续提交 W2| Phase
    Event -->|W2 后，两张票据齐全| Ready
    classDef copy fill:#e7f0ff,stroke:#4775b5,color:#17345e;
    classDef event fill:#fff3db,stroke:#b78019,color:#4c3400;
    class Load,Phase,Materialize,Stream,Native,ACL copy;
    class Submit,Event,Ready event;
```

`ready[index]` 保存的是**可供等待的完成票据**；字典里有这个键，不代表 CPU 已确认 H2D 完成。W13 票据只约束 W13 计划，W2 票据约束 W2 计划；由于共享一条 FIFO 复制流，W2 事件也排在本层 W13 之后。实际使用权重前仍要在计算流插入对应的事件等待。

“是不是一个一个专家 copy”要分两层看：

| 层次 | 当前实现 |
|---|---|
| 描述要搬哪些字节 | `_projection_copy_plans()` 按“组件 → 专家”构造 `HostCopyRegion`，因此原始描述确实包含每个冷专家的切片。 |
| 整理描述符 | `AclHostTransferOps._lower()` 校验各切片；只有源和目的地址都相邻，且共享相同源 registration、目的 owner 时才合并。存在 stride/padding 时不能任意合并。 |
| 发起搬运 | 整理后的描述符按批次提交，默认每批至多 256 项；调用 `aclrtMemcpyBatchAsync`，不是 Python 对每个专家逐一 `copy_` 后等待。描述符通常已在初始化 `prepare()` 时缓存。 |
| 标记完成 | 一份投影计划的所有批次提交后，统一记录一个就绪事件。这里没有为每个专家逐个同步。批量 API 也不等于承诺硬件同时搬完所有切片。 |

`OpCommand::RunOpApi` 这层保证实际 ACL 提交遵守前面排入的等待、后面排入的完成事件的顺序；不能从“Python 调用了 submit”推断复制已执行完。源码：[复制计划构造][copy-plan-builder]、[计划预准备][init-copy-prepare]、[描述符合并和分批][copy-lower]、[复制流内提交][copy-submit]、[C++ 队列提交][copy-native-submit]、[ACL 批量调用][copy-native-batch]、[完成事件记录][copy-ready-event]。

### 8.3 把计算流与复制流放到同一张图

下面只展开 3 个受管理层的关键依赖。绿色是计算，蓝色是 H2D，橙色是等待/事件边界；实线为同一流上的先后关系，虚线为跨流必须满足的事件依赖。图不按耗时比例绘制，两个流实际能重叠多少取决于运行时耗时。

`E_begin`、`E13_i`、`E2_i`、`D_i` 是图里的事件简称，分别表示入口计算完成、本层冷 W13 就绪、本层冷 W2 就绪、本层 combine 后的计算完成；不是代码里的变量原名。

```mermaid
flowchart TB
    subgraph compute[计算流：context.compute_stream]
        CBegin["记录 E_begin<br/>开始模型 forward"]
        C0A["层 0：前序计算、dispatch、分组元数据"]
        C013["等待 E13_0<br/>GMM1 → SwiGLU"]
        C02["等待 E2_0<br/>GMM2 → combine → 记录 D_0"]
        C1A["层 1：前序计算、dispatch、分组元数据"]
        C113["等待 E13_1<br/>GMM1 → SwiGLU"]
        C12["等待 E2_1<br/>GMM2 → combine → 记录 D_1"]
        C2A["层 2：前序计算、dispatch、分组元数据"]
        C213["等待 E13_2<br/>GMM1 → SwiGLU"]
        C22["等待 E2_2<br/>GMM2 → combine → 记录 D_2"]
        CEnd["完成模型其余工作<br/>finally：wait_stream(copy)<br/>记录最终事件，CPU synchronize"]
        CBegin --> C0A --> C013 --> C02 --> C1A --> C113 --> C12
        C12 --> C2A --> C213 --> C22 --> CEnd
    end
    subgraph copying[单条复制流：owner.weight_copy_stream]
        HBegin["等待 E_begin"]
        H013["层 0 冷 W13 → bank 0<br/>记录 E13_0"]
        H02["层 0 冷 W2 → bank 0<br/>记录 E2_0"]
        H113["层 1 冷 W13 → bank 1<br/>记录 E13_1"]
        H12["层 1 冷 W2 → bank 1<br/>记录 E2_1"]
        HReuse["等待 D_0<br/>旧层结束后，bank 0 才能覆盖"]
        H213["层 2 冷 W13 → bank 0<br/>记录 E13_2"]
        H22["层 2 冷 W2 → bank 0<br/>记录 E2_2"]
        HBegin --> H013 --> H02 --> H113 --> H12 --> HReuse --> H213 --> H22
    end
    CBegin -.-> HBegin
    H013 -.-> C013
    H02 -.-> C02
    H113 -.-> C113
    H12 -.-> C12
    C02 -.-> HReuse
    H213 -.-> C213
    H22 -.-> C22
    H22 -.-> CEnd
    classDef compute fill:#e6f4ea,stroke:#34834b,color:#173b24;
    classDef copy fill:#e7f0ff,stroke:#4775b5,color:#17345e;
    classDef event fill:#fff3db,stroke:#b78019,color:#4c3400;
    class C0A,C013,C02,C1A,C113,C12,C2A,C213,C22 compute;
    class H013,H02,H113,H12,H213,H22 copy;
    class CBegin,HBegin,HReuse,CEnd event;
```

入口已把层 0、层 1 的复制排进同一条流，所以顺序是 `0.W13 → 0.W2 → 1.W13 → 1.W2`。层 0 的 combine 提交后，`retire(0)` 才追加 `wait(D_0) → 2.W13 → 2.W2`。这不是为每层另起一条 copy stream。

这样有两类重叠机会：本层 GMM1/激活可以与本层 W2 搬运重叠；当前层计算可以与下一层冷权重预取重叠。`W13` 就绪后即可开始 GMM1，不必把本层两份权重都等齐。图里的依赖只保证正确顺序，不保证运行中一定出现某段长度的重叠。

| 本轮阶段 | bank 0（槽位 6、7） | bank 1（槽位 8、9） | 谁触发 |
|---|---|---|---|
| forward 入口 | 提交层 0 冷专家 | 提交层 1 冷专家 | `begin_forward()` 的 `_submit(0/1)`。 |
| 层 0 combine 提交后 | 排入等待 `D_0`，然后装层 2 | 继续保留层 1 | `finish_layer(0) → retire(0) → _submit(2, reader)`。 |
| 层 1 combine 提交后 | 保留/完成层 2 预取 | 层 1 已消费，无层 3 可补 | `retire(1)` 只推进状态。 |
| 层 2 combine 提交后 | 层 2 已消费，无层 4 可补 | 不再使用 | `retire(2)`；最后由 scope 排空本轮任务。 |

上表的“提交”不等于“已经复制完成”，何时能读仍由图中的就绪事件保证。各层常驻槽位 0～5 在整个过程中保持自己的权重，不参加双 bank 轮转。

### 8.4 每个等待点的具体调用栈和源码

| 等待点 | 从哪里一路调用下来 | 最终动作与含义 |
|---|---|---|
| forward 入口，让复制尊重先前计算 | `forward_scope → begin_forward → context.record_compute_completion → completion.wait_on(copy_stream)` | 计算流记录 `E_begin`，复制流 `wait_event(E_begin)`。 |
| GMM1 前等冷 W13 | `AscendRunnerCore.run → owner.prepare_weights → prefill.weights → pipeline.acquire → ready[current].w13.completion.wait_on(compute_stream)` | 计算流 `wait_event(E13_i)`，后续 GMM1 才能读本层冷 W13。 |
| GMM2 前等冷 W2 | `_run_ffn → wait_for_w2 回调 → owner.wait_for_w2 → pipeline.wait_for_w2 → ready[current].w2.completion.wait_on(compute_stream)` | 计算流 `wait_event(E2_i)`，后续 GMM2 才能读本层冷 W2。 |
| 复用旧层冷 bank | `dispatcher.combine → prefill.finish_layer → record_compute_completion → pipeline.retire → _submit(i+2, reader) → reader.completion.wait_on(copy_stream)` | 复制流 `wait_event(D_i)`，禁止提前覆盖旧层还可能读取的字节。 |
| 整轮结束后清理 | `forward_scope.finally → compute_stream.wait_stream(weight_copy_stream) → record_compute_completion().completion.synchronize()` | 先把复制流末尾并入计算流，再由 CPU 等最终事件；完成后清理引用、重置 pipeline。 |

最底层代码非常直接：

```python
# EventCompletion：按需向某条设备流插入依赖
def wait_on(self, stream):
    stream.wait_event(self.event)

# 整轮收尾使用：CPU 等待该事件真正完成
def synchronize(self):
    self.event.synchronize()
```

这区分的是**权重预取链路中的设备依赖和 CPU 等待**，不意味着整个 forward 的其他部分没有 CPU 同步；例如 dispatch 的计数读取、通信及 `_check_error()` 仍有自己的行为。

源码定位：[入口及初始提交][copy-begin]、[W13 acquire 与 W2 wait][copy-waits]、[finish_layer 记录旧读者事件][copy-finish-layer]、[retire 及 bank 复用][copy-retire]、[计算流事件创建][copy-compute-event]、[EventCompletion 的两种等待][eventwait]、[scope 的最终排空][scope]。

## 9. 五个权重管理对象：初始化与 forward 时序

本节把前文分散的调用关系集中到两张图中，范围为 **`prefill_only` + 每层部分常驻 + DeepEP normal + type2**。令 `E` 为每层本地专家数，`R > 0` 为每层固定常驻数，`C = E - R`，`L` 为本 rank 受管理层数，物理槽位数为 `S = L × R + 2 × C`；这里取 `split_gmm=False`，即当前实现中的 `S <= 1024`。`i` 始终表示受管理层的顺序编号，不是模型的绝对 `layer_id`。

| 对象 | 管什么 |
|---|---|
| `MemFabricHostPool` | Host 内存本身、地址映射、注册和生命周期。 |
| `ExpertLayerStorage` | 这块内存中，本层专家权重的加载、布局转换和封存。 |
| `ExpertBank` | NPU 权重内存和物理 slot，包括根池与共享底层内存的视图。 |
| `AscendPrefillWeights` | 从哪个 Storage 搬到哪个 Bank，以及如何把权重交给计算。 |
| `LayerPrefetchPipeline` | 何时提交搬运、何时允许读取和覆盖。 |

两张图都额外画出调用入口和 `AclHostTransferOps`，用于说明谁触发这些对象、谁实际提交复制。图中的 Host / Bank 内存读写箭头表示数据访问；复制和 GMM 由相应调用方提交，内存对象本身不执行这些算子。

### 9.1 模型初始化：先准备 Host 主副本，再建立 NPU 供权资源

“初始化调用方”合并表示 `ModelRunner`、`AscendOpsOffloader`、`PrefillOnlyExecution` 以及格式适配层的相关调用；展开到各入口的时序见第 4 节。

```mermaid
sequenceDiagram
    autonumber
    participant Init as 初始化调用方
    participant Host as MemFabricHostPool
    participant Storage as ExpertLayerStorage<br/>每层一个
    participant Bank as ExpertBank<br/>根池与视图
    participant Prefill as AscendPrefillWeights
    participant Transfer as AclHostTransferOps
    participant Pipeline as LayerPrefetchPipeline

    Init->>Init: 建立专家目录，计算 E、R、C、S
    Init->>Init: 首次 prepare_layer 时探测运行布局和 Host 空间需求
    Init->>Host: __init__(总需求, device, physical_bytes, native_acl)
    Host->>Host: 分配 Host 内存，建立 HVA/DVA 映射，注册 ACL 可读区域
    Note over Host,Storage: Pool 管理底层分配，Storage 从同一个池中取得本层切片
    loop 每个受管理专家层
        Init->>Storage: __init__(layer, spec, format, host_pool)
        Storage->>Host: allocate(本层各 component 所需空间)
        Host-->>Storage: Host tensor 切片与内存 capability
        Storage->>Storage: 建立 checkpoint 视图与逐专家地址描述
        Storage->>Storage: 注册参数及 weight_loader 回调
    end

    Init->>Init: model.load_weights() 遍历 checkpoint
    loop 各层、各专家、各权重分片
        Init->>Storage: 参数的 weight_loader 回调
        Storage->>Storage: global expert 转 local expert，校验形状和分片
        Storage->>Host: 通过已分配的视图写入 checkpoint 字节
        Storage->>Storage: 记录本分片已加载
    end
    loop 每个受管理专家层
        Init->>Storage: process_weights_after_loading() 调用 finalize()
        Storage->>Storage: 校验全部本地专家的必要分片已加载
        Storage->>Storage: 经 format 逐专家、逐投影转换运行布局
        Storage->>Host: 将转换后的 native 字节写回本层 Host 区域
        Storage->>Storage: 标记 sealed，禁止再次加载
    end
    Note over Host,Storage: 常驻专家与冷专家都有 Host 主副本<br/>运行布局在加载期完成，forward H2D 不再转换布局

    Init->>Storage: post_init() 对各层 require_sealed()
    Init->>Bank: 经 format.allocate_bank(S) 创建 NPU 根执行池
    Bank->>Bank: 保存组件 tensor，建立每个物理 slot 的地址描述
    Init->>Init: 创建 weight_copy_stream 与 context
    Init->>Prefill: __init__(owner, resident_experts=R)
    Prefill->>Bank: view(L*R + b*C, C)，建立两个冷 bank 视图
    Bank-->>Prefill: 共享根池内存的视图，b 为 0 或 1
    Prefill->>Bank: execution.quant_info()
    Bank-->>Prefill: 整个 NPU 池的计算权重视图
    Prefill->>Prefill: 建立 counts[E] 和 groups[S, 2]
    Prefill->>Transfer: 创建 host_ops

    loop 每个受管理层 i：填充常驻区
        Prefill->>Bank: view(i*R, R)
        Bank-->>Prefill: 本层固定常驻槽位
        Prefill->>Storage: 读取前 R 个专家的 component_capabilities
        Prefill->>Prefill: 建立常驻 W13、W2 复制计划
        Prefill->>Transfer: materialize_into(计划, 常驻 bank, copy_stream)
        Transfer->>Bank: 复制流排队：Host native 字节搬入常驻槽位
        Transfer-->>Prefill: 完成凭据，表示复制已提交
    end
    Prefill->>Prefill: copy_stream.synchronize()，等待常驻填充完成
    Prefill->>Transfer: clear_cache()，清理常驻填充的描述符缓存
    Prefill->>Prefill: 清理本次填充的临时保留引用

    loop 每个受管理层 i：准备冷专家搬运
        Prefill->>Storage: 读取专家 [R, E) 的 component_capabilities
        Prefill->>Bank: 读取 cold_bank[i % 2] 的 component_regions
        Prefill->>Prefill: 缓存本层 W13、W2 的 Host 到 Bank 计划
        Prefill->>Transfer: prepare(w13_plan)，prepare(w2_plan)
        Transfer-->>Prefill: 地址描述符准备完成，此时未提交冷专家 H2D
    end
    Prefill->>Pipeline: __init__(layer_ids, load=self._load, copy_stream)
    Pipeline->>Pipeline: reset()，当前层为 0，ready 为空
    Prefill-->>Init: owner.prefill 初始化完成
    Init->>Init: 注册 DeepEP pre_dispatch / post_combine hooks
    Init->>Init: owner.ready = True
    Note over Bank,Pipeline: 常驻权重已就位，冷 bank 已分配，复制计划已缓存<br/>首次 begin_forward() 才提交本轮前两层冷专家预取
```

初始化后的 Host 主副本、NPU 根池和常驻权重跨 forward 保留。冷 bank 的“地址已经存在”和“本轮权重已经搬好”是两个不同状态；后者由每轮的复制事件保证。[Host 池][five-host-pool]、[Storage 加载与封存][storage]、[运行资源建立][postinit]、[Prefill 视图与复制计划][prefill]、[DeepEP hook 注册][deepephooks]

### 9.2 一次 Prefill forward：初始预取、按层读取、隔层复用

“Forward 调用方”合并表示 `ModelRunner`、执行策略、DeepEP dispatcher、MoE runner 和 owner 的转发调用。**图的上下顺序表示 CPU 调用与提交顺序，不代表设备上的 H2D、GMM、combine 已经执行完成。** `wait_event` 为后续设备任务建立依赖；标出的 `synchronize` 才是 CPU 等待完成。

```mermaid
sequenceDiagram
    autonumber
    participant Forward as Forward 调用方
    participant Host as MemFabricHostPool
    participant Storage as ExpertLayerStorage<br/>每层一个
    participant Bank as ExpertBank<br/>常驻区与两个冷 bank
    participant Prefill as AscendPrefillWeights
    participant Transfer as AclHostTransferOps
    participant Pipeline as LayerPrefetchPipeline

    Note over Host,Storage: 初始化时已封存 native Host 主副本，整个 forward 保持有效<br/>运行时直接使用缓存计划里的地址，不重新加载 checkpoint 或转换布局
    Forward->>Forward: 进入 forward_scope，创建本轮 context
    Forward->>Pipeline: begin_forward(context)
    Pipeline->>Pipeline: 登记本轮身份，计算流记录入口事件
    Pipeline->>Pipeline: 复制流排入等待入口事件
    loop 首两个受管理层 j（若存在）
        Pipeline->>Pipeline: _submit(j, context)
        Pipeline->>Prefill: _load(layer_ids[j], j % 2, context)
        Prefill->>Transfer: materialize_into(本层 W13 计划, cold_bank, copy_stream)
        Transfer->>Bank: 入队：Host W13 搬入冷 bank，然后记录 E13_j
        Transfer-->>Prefill: W13 TransferTicket
        Prefill->>Transfer: materialize_into(本层 W2 计划, cold_bank, copy_stream)
        Transfer->>Bank: 同一复制流入队：Host W2 搬入冷 bank，然后记录 E2_j
        Transfer-->>Prefill: W2 TransferTicket
        Prefill-->>Pipeline: PrefillLayerTransfer(w13, w2)，保存到 ready[j]
    end
    Pipeline-->>Forward: 初始预取已提交
    Forward->>Forward: yield，进入 ModelRunner._forward_raw()

    loop 按模型顺序执行每个受管理层 i
        Forward->>Pipeline: DeepEP pre_dispatch hook 经 owner 调用 start(layer, context)
        Pipeline->>Pipeline: 校验层顺序，started = True
        Note over Forward,Pipeline: start 只登记本层开始，不提交 H2D，也不等待权重
        Forward->>Forward: DeepEP dispatch，得到按本地逻辑专家分组的 token
        Forward->>Forward: runner 构造 counts[E]，group_list_type = 1
        Forward->>Prefill: owner.prepare_weights() 调用 weights(layer, runner_input)
        Prefill->>Prefill: prepare_prefill_counts() 生成物理 slot 与 token 数配对
        Prefill->>Prefill: 写回 groups[S, 2]，group_list_type = 2
        Note over Forward,Prefill: hidden_states 和 scale 的行顺序保持不变<br/>激活仍使用逻辑专家 counts，GMM 使用物理槽位分组表
        Prefill->>Pipeline: acquire(layer, context)
        Pipeline->>Pipeline: 计算流排入 wait_event(E13_i)
        Pipeline-->>Prefill: 当前冷 bank 编号 i % 2
        Prefill-->>Forward: 返回初始化时缓存的整池 quant_info
        Forward->>Bank: 提交 GMM1，按 type2 读取本层常驻与冷 W13
        Forward->>Forward: 提交 SwiGLU 等激活处理
        Forward->>Pipeline: 经 owner 调用 wait_for_w2(layer, context)
        Pipeline->>Pipeline: 计算流排入 wait_event(E2_i)
        Pipeline-->>Forward: W2 读取依赖已建立
        Forward->>Bank: 提交 GMM2，按 type2 读取本层常驻与冷 W2
        Forward->>Forward: DeepEP combine 提交合并计算与通信

        Forward->>Prefill: post_combine hook 调用 finish_layer(layer)
        Prefill->>Prefill: context.record_compute_completion() 记录 D_i
        Prefill->>Pipeline: retire(layer, reader=D_i 的凭据, context)
        opt 后面还存在第 i+2 个受管理层
            Pipeline->>Pipeline: _submit(i+2, context, reader)
            Pipeline->>Pipeline: 复制流先排入 wait_event(D_i)
            Pipeline->>Prefill: _load(layer_ids[i+2], i % 2, context)
            Prefill->>Transfer: 依次 materialize_into(W13 计划)、materialize_into(W2 计划)
            Transfer->>Bank: 等待 D_i 后才能覆盖同一个冷 bank<br/>分别在 W13、W2 复制后记录新完成事件
            Transfer-->>Prefill: 新层 W13、W2 的 TransferTicket
            Prefill-->>Pipeline: 保存 ready[i+2]
        end
        Pipeline->>Pipeline: 删除 ready[i]，current += 1，started = False
        Pipeline-->>Prefill: 本层已退役，后续复用依赖已提交
        Prefill-->>Forward: 返回 combine 的 hook，继续模型执行
    end

    Forward->>Pipeline: 模型返回后调用 finish_forward(context)
    Pipeline-->>Forward: 确认受管理层已全部消费
    Forward->>Forward: owner._check_error()
    Forward->>Forward: finally：计算流 wait_stream(copy_stream)
    Forward->>Forward: 记录最终事件并 synchronize()，等待两条流排空
    Forward->>Forward: context.resources.clear()
    Forward->>Pipeline: reset()
    Note over Host,Bank: 本轮临时引用与调度状态已清理<br/>Host 主副本、NPU 根池和常驻权重继续保留
```

图中 GMM / 激活展示有 token 的计算路径；空 token 的处理由 runner 决定。冷专家复制计划在初始化时已覆盖本层全部 `C` 个冷专家，不会因为某个专家本轮 token 数为 0 而自动缩减。

这里有两个方向相反的依赖：

- **读取前等搬好**：`acquire()` / `wait_for_w2()` 让计算流等待复制事件，随后 GMM 才能读取相应冷权重。
- **覆盖前等读完**：`finish_layer() → retire() → _submit(reader)` 让复制流等待 combine 提交之后记录的计算事件，随后才能覆盖同一个冷 bank。第 `i+1` 层已经预取，这里补的是第 `i+2` 层。

`finish_layer()` 的触发点是 **combine 已提交之后**；CPU 在这里记录事件和提交等待，不必先等 combine 在设备上实际完成。因此图中后续提交动作可以发生在设备仍执行前面 GMM 的时候。`begin_forward()` 则由 `forward_scope` 直接调用，并非 dispatcher hook。[整轮 scope][scope]、[初始提交][copy-begin]、[W13/W2 等待][copy-waits]、[combine 后退役][copy-finish-layer]、[bank 复用][copy-retire]、[复制提交][copy-submit]、[GMM 与激活][copy-ffn]

[worker]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/managers/tp_worker.py:282
[modelrunner]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/model_executor/model_runner.py:506
[fused]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:339
[baseoffloader]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/utils/offloader.py:33
[offloader]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/ascend_ops.py:76
[storage]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/format.py:331
[strategy]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/execution_mode.py:84
[prefill]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:151
[pipeline]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:65
[preparelayer]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/ascend_ops.py:262
[storageparams]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/format.py:370
[context]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/ascend_ops.py:52
[postinit]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/execution_mode.py:248
[modelslim]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/quantization/modelslim/modelslim.py:544
[fusedcreate]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:597
[moerunner]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/moe/moe_runner/runner.py:240
[runnercore]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/moe/moe_runner/ascend.py:247
[dispatcher]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/moe/token_dispatcher/ascend_ops.py:177
[format]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/format.py:70
[bank]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/format.py:279
[transferops]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/native_acl.py:349
[loaderpost]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/model_loader/loader.py:1085
[loadingcontext]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/model_loader/loader.py:158
[prefillweights]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:301
[eventwait]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/transfer.py:49
[scope]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/execution_mode.py:300
[deepephooks]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/execution_mode.py:284
[slots]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/options.py:55

[init-worker-create]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/managers/tp_worker.py:486
[init-default-loader]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/model_loader/loader.py:1002
[init-weight-callback]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:440
[init-repack]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/format.py:198
[init-w8-adapter]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/w8a8_format.py:17
[init-copy-prepare]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/native_acl.py:239

[type2-input]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/moe/moe_runner/ascend.py:605
[type2-original-quant]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/quantization/modelslim/modelslim.py:620
[type2-runner]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/moe/moe_runner/ascend.py:258
[type2-gmm]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/hardware_backend/npu/moe/matmul.py:35
[type2-mapping]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/small_batch_kernels.py:62
[type2-pool]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:169
[copy-plans-init]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:215
[copy-model-scope]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/model_executor/model_runner.py:1902
[copy-ffn]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/layers/moe/moe_runner/ascend.py:353
[copy-plan-builder]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:25
[copy-lower]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/native_acl.py:307
[copy-submit]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/native_acl.py:381
[copy-native-submit]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/native/acl_batch.cpp:21
[copy-native-batch]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/native/acl_batch.h:95
[copy-ready-event]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/transfer.py:389
[copy-begin]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:89
[copy-waits]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:121
[copy-finish-layer]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:366
[copy-retire]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/prefill.py:132
[copy-compute-event]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/ascend_ops.py:68
[five-host-pool]: /Users/yaobao/Desktop/Workspace-Ascend/sglang-moe-offload-npu/源哥的优化分支/simplify/python/sglang/srt/offloading/moe/npu/storage.py:33
