# 公共基础设施维护

项目正式名称为 EurekaSI，主仓库为 https://github.com/Trump0412/EurekaSI。发行包名 `eurekasi`；同名 CLI 与原 `spatial` 命令等价。内部 Python 模块、资源安装目录与本地配置目录继续使用原名称，避免破坏现有使用方式。多论文和应用的组织约定见 [研究与应用工作流](RESEARCH_WORKFLOW.md)。

当前主仓库维护公共接口；历史方法由 `catalog/sources.yaml` 固定 commit 获取。机器路径、队列脚本、旧 Git bundle、数据和运行产物保留在忽略的 `_archives/`、`trans/` 或外部工作区。最近的合并决策见 [HANDOFF_MERGE](HANDOFF_MERGE.md)。

## 源码、安装与配置

源码 checkout 的资源位于根目录 catalog/configs/patches，默认配置是根目录的 `spatial.local.yaml`。wheel 声明将运行资源安装到 `share/spatial-intelligence`，通过包所在位置与 Python 安装 scheme 定位，不依赖当前工作目录。支持普通 venv、user 与 target 布局的定位逻辑；实际 wheel 构建和隔离安装尚未在本轮验收。

wheel 默认把本地设置写到 `$XDG_CONFIG_HOME/spatial-intelligence/config.yaml`；Windows 可回退到 `%APPDATA%`，其他情况用 `~/.config`。`SPATIAL_CONFIG`、`init --output` 和显式 settings 路径优先，保持已有源码流程。示例 `configs/...` 在当前目录不存在时回退到安装资源；已经存在的本地配置优先。配置内部的数据路径仍相对执行目录，需要可迁移运行时请使用明确的绝对路径或环境变量。

源码维护仍推荐 editable 安装。wheel 只携带运行所需的包、catalog、配置和补丁，不携带 Bash 安装脚本、完整研究文档、实验测试或历史源码。不要把 wheel 的占位配置当作实际机器配置。

## 静态审查与运行验证

```bash
python scripts/static_check.py
python scripts/release_check.py
```

两条命令只解析本地文件；release 检查与归档共用源码选择器，核对版本、依赖声明、资源覆盖、source lock、VSI 文件哈希与发布链接。版本从 AST 字面量读取，不执行包的 `__init__.py`。Python 3.10 使用声明的 tomli；3.11+ 使用标准库 tomllib。

有运行验证任务时，在准备好的 Linux/WSL 环境执行 `bash scripts/check.sh`。解释器优先级为 `SPATIAL_PYTHON` → `SPATIAL_VENV` → 当前 `VIRTUAL_ENV` → 仓库 `.venv`。检测到未完成安装标记时拒绝运行；不悄悄回退到系统 Python。bootstrap/check 清除宿主的 PYTHONPATH/PYTHONHOME 并关闭 user-site 注入，自定义插件应安装在所选环境里。

CI 分开记录静态解析、wheel/源码包和 CPU 优化器/分布式回归。CI 配置存在不等于 CI 已通过；当前合并未触发这些任务。交接包的历史测试结果不并入本库的通过数。

## 来源与发布

`catalog/sources.yaml` 和 `sources.lock.json` 的 URL、完整 SHA、补丁列表保持一致。选择新源码版本时逐项目评审，不能把无法从公共 URL 获取的本地 HEAD 写进公开锁。修改权重、视觉输入、marker、scene split、评分或缓存身份时，同步修订契约和相关回归。

本工作树标识为 `0.2.1`，尚未发布；版本同步到包常量、pyproject、README、CHANGELOG 和 CITATION。保留原 LICENSE 的范围声明，第三方代码与资产按 NOTICE 记录。

未来发布源码使用 `python scripts/package_release.py --output dist/spatial-intelligence-source.zip`。默认 ZIP 时间戳为 1980-01-01，可指定 `SOURCE_DATE_EPOCH` 或 `--source-date-epoch`；同一文件字节与时间参数得到稳定元数据，跨压缩库版本的压缩字节不作保证。文件权限依据源码类型固定，避免 Windows 导出的脚本失去可执行标志。脚本保留原有拒绝覆盖和临时文件原子替换策略。提交、远端地址和发布仍由维护者决定。
