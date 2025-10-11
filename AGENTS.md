# agent.md | Codex 执行代理规范（中英双语版）

> 目标：让 Codex（或同类代码生成模型）在最少追问的前提下自动完成需求、自测并保证可运行，交付可维护成果。<br>> Goal: Enable Codex (or similar code-generation agents) to autonomously deliver maintainable, runnable work with minimal back-and-forth, including self-testing.

---

## 0. 沟通与产出原则 | Communication & Deliverable Principles
1. **语言**：所有交流、注释、提交信息、README 默认使用简体中文，可附英文术语。<br>**Language**: Default to Simplified Chinese for communication, comments, commit messages, and README content; append English terms when helpful.
2. **默认动作**：遇到可合理假设的细节，应直接根据专业判断继续实现，并将假设记录在 `ASSUMPTIONS.md`。<br>**Default action**: When reasonable assumptions are possible, proceed with a professional default and record the assumption in `ASSUMPTIONS.md`.
3. **最小打扰**：除非风险极高，不等待额外确认；在交付物中标注可选项与替代方案。<br>**Minimise back-and-forth**: Avoid pausing for confirmation unless risk is high; document alternatives and options in the deliverables.
4. **可运行性**：生成的代码、脚本、配置必须可本地一键运行并通过测试。<br>**Run-ability**: Produced code, scripts, and configs must run locally via one-liners and pass tests.
5. **可复现性**：输出固定的环境说明与锁定依赖（如 `requirements.txt`、`poetry.lock`、`package-lock.json`）。<br>**Reproducibility**: Provide deterministic environment notes and lock dependencies (e.g., `requirements.txt`, `poetry.lock`, `package-lock.json`).
6. **安全优先**：默认不开启高危权限；对外部调用使用显式白名单和可配置开关。<br>**Security first**: Do not enable dangerous permissions by default; gate external calls behind explicit allow-lists and configurable switches.

---

## 1. 交付物目录结构（通用模板）| Suggested Deliverable Structure (Generic)

```
project-root/
├─ src/                         # 业务源码 | Core source code
├─ tests/                       # 单元/集成测试 | Unit/integration tests
├─ examples/                    # 最小示例 | Minimal runnable examples
├─ scripts/                     # 任务脚本 | Task automation scripts
├─ config/                      # 配置模板 | Configuration templates
├─ docs/                        # 设计/API/运维文档 | Design/API/Ops docs
├─ .github/workflows/           # CI 工作流 | CI workflows
├─ Dockerfile                   # 容器化运行 | Container runtime
├─ docker-compose.yml           # 本地依赖编排 | Local dependency orchestration
├─ pyproject.toml / package.json / requirements.txt
├─ Makefile                     # 一键任务入口 | One-command task entry
├─ README.md                    # 使用说明 | Usage guide
├─ ASSUMPTIONS.md               # 假设与权衡 | Assumptions & trade-offs
├─ CHANGELOG.md                 # 变更日志 | Change log
└─ agent_report.md              # 自动执行报告 | Automation run report
```

> 要求：在可行情况下保持上述结构，为团队提供统一入口。<br>> Requirement: Adopt the structure when feasible to give the team a consistent entry point.

---

## 2. 一键任务（Makefile 约定）| Makefile One-liner Conventions

```
make setup        # 安装依赖、初始化环境 | Install dependencies and bootstrap environment
make fmt          # 代码格式化 | Format code
make lint         # 静态检查 | Run linters / static analysis
make test         # 运行测试 | Execute test suite
make run          # 启动应用或示例 | Run the application/example
make build        # 构建产物 | Build distributables or images
make ci           # 本地模拟 CI：lint + test + build | Local CI: lint + test + build
```

> PR/交付前需确保 `make ci` 全绿；如无需 Docker，可用其他可发布产物替代。<br>> Ensure `make ci` passes before delivery; if Docker isn’t needed, substitute with another distributable artefact.

---

## 3. 需求到实现的自动流程 | Requirement-to-Implementation Workflow
1. **解析需求**：抽取功能点、接口、数据结构、约束、非功能需求，识别歧义并在 `ASSUMPTIONS.md` 中记录默认值。<br>**Requirement analysis**: Extract features, interfaces, data structures, constraints, and non-functional needs; capture ambiguities and assumptions in `ASSUMPTIONS.md`.
2. **制定方案**：输出模块边界、依赖、关键数据流或时序图，并在 `docs/decision_record.md` 说明取舍。<br>**Design**: Define module boundaries, dependencies, key data flows or sequence diagrams, and document trade-offs in `docs/decision_record.md`.
3. **脚手架落地**：搭建目录与基础文件，补齐 `README.md` 与一键运行指引；配置 `config/.env.example` 等模板。<br>**Scaffolding**: Lay out directories and base files, enrich `README.md` and quick-start instructions, and provide `config/.env.example` templates.
4. **实现编码**：遵循整洁代码，使用中文注释并为公共函数编写 docstring/示例。<br>**Implementation**: Write clean code with Chinese comments and docstrings/examples for public functions.
5. **自测优先**：为模块编写单元测试，关键流程补集成测试，目标覆盖率 ≥ 80%。<br>**Self-testing**: Create unit tests per module and integration tests for critical flows, targeting ≥80% coverage.
6. **质量闸门**：执行 `make fmt && make lint && make test`，生成覆盖率报告与构建产物。<br>**Quality gate**: Run `make fmt && make lint && make test`, produce coverage reports and build artefacts.
7. **交付收尾**：更新 `CHANGELOG.md`、扩充 `README` 常见问题，整理 `agent_report.md`。<br>**Delivery wrap-up**: Update `CHANGELOG.md`, expand the README FAQ, and prepare `agent_report.md`.

---

## 4. 代码与文档规范 | Code & Documentation Standards
- Python：使用 `ruff` + `black` + `mypy`（严格模式）。<br>Python: Adopt `ruff`, `black`, and strict `mypy`.
- TS/JS：使用 `eslint`（typescript-eslint）+ `prettier`。<br>TS/JS: Use `eslint` (typescript-eslint) plus `prettier`.
- Go：使用 `gofmt` + `golangci-lint`。<br>Go: Apply `gofmt` and `golangci-lint`.
- 日志需统一封装，默认不打印敏感信息。<br>Log through a unified wrapper and avoid printing sensitive data by default.
- 配置优先 `config.yaml` + `.env`，读取时需提供默认值和类型校验。<br>Prefer `config.yaml` plus `.env`; supply defaults and type validation when loading.
- 错误信息应在外层统一处理，提供中文可读的提示与修复建议。<br>Handle errors at standard boundaries, presenting human-readable Chinese messages with remediation tips.
- 公共 API 必须附 docstring 与使用示例。<br>Public APIs must include docstrings and usage samples.
- 提交信息遵循 Conventional Commits（如 `feat:`、`fix:`、`docs:`）。<br>Follow Conventional Commits (e.g., `feat:`, `fix:`, `docs:`).

---

## 5. 测试策略 | Testing Strategy
1. **单元测试**：覆盖核心算法、数据转换、边界与异常路径。<br>**Unit tests**: Cover core algorithms, data transforms, edge cases, and error paths.
2. **集成测试**：验证端到端关键流程，必要时对外部依赖进行 mock。<br>**Integration tests**: Validate end-to-end flows with mocks for external systems when needed.
3. **回归样例**：为已修复的缺陷补充最小复现测试。<br>**Regression cases**: Add minimal reproductions for fixed bugs.
4. **性能冒烟（可选）**：对关键路径做小规模基准，记录指标与阈值。<br>**Performance smoke (optional)**: Benchmark critical paths lightly, record metrics and thresholds.
5. **覆盖率目标**：`--cov=src --cov-report=xml`，保持行覆盖率 ≥ 80%。<br>**Coverage target**: Run with `--cov=src --cov-report=xml`, aiming for ≥80% line coverage.

---

## 6. CI/CD 最小流程 | Minimal CI/CD Pipeline
- 触发：`push` 与 `pull_request`。<br>Trigger on `push` and `pull_request`.
- 作业顺序：<br>Pipeline stages:
  1. **Setup**：安装依赖并配置缓存。<br>**Setup**: Install dependencies and configure caches.
  2. **Static**：执行 `make fmt && make lint`。<br>**Static**: Run `make fmt && make lint`.
  3. **Test**：执行 `make test` 并上传覆盖率。<br>**Test**: Run `make test` and upload coverage.
  4. **Build**：执行 `make build`。<br>**Build**: Execute `make build`.
- 任一阶段失败即阻断合入。<br>Failing any stage blocks the merge.

---

## 7. 安全与合规 | Security & Compliance
- 禁止硬编码密钥/Token/私钥；改用 `.env` + 密钥管理（如 GitHub Secrets）。<br>Never hardcode secrets; rely on `.env` plus secret managers (e.g., GitHub Secrets).
- 外部请求需设置超时、重试、熔断，且使用白名单域名。<br>External requests must include timeouts, retries, circuit breakers, and domain allow-lists.
- 文件操作限制在 `project-root/` 内，删除/覆盖前请备份。<br>Restrict file operations to `project-root/`; back up before deleting or overwriting.
- 记录关键操作与失败栈，满足审计需求。<br>Log critical actions and failure traces for auditability.

7. **Conda 环境要求（新增）**：所有自动运行的 agent 或脚本在执行时应确认处于名为 `stock_prediction` 的 conda 环境中，或等效地使用已经安装并锁定项目依赖的虚拟环境。
  <br>**Why**: 保证在 agent 自动运行、测试或 CI 中使用一致的依赖和 Python 版本，避免因为全局包差异或者系统 Python 版本差异导致不可复现的失败。
  <br>**How**: 在 agent 的启动脚本或 CI workflow 中显式激活环境（例如 `conda activate stock_prediction`），或使用 `actions/setup-python` 并安装 `requirements.txt`。在 `README.md` 或 `docs/ops.md` 中记录该要求。

---

## 8. 自动执行报告模板 | Automation Execution Report Template

```
# 本次自动执行报告 | Automation Execution Report

## 需求摘要 | Requirement Summary
- 背景与目标 | Background & objectives:
- 核心功能点 | Key features:

## 关键假设 | Key Assumptions
- （详见 ASSUMPTIONS.md）| (See ASSUMPTIONS.md)

## 方案概览 | Solution Overview
- 架构与模块 | Architecture & modules:
- 选型与权衡 | Choices & trade-offs:

## 实现与自测 | Implementation & Self-testing
- 一键命令 | One-liner: `make setup && make ci && make run`
- 覆盖率 | Coverage: xx%
- 主要测试清单 | Major tests: 单元 N 项 / 集成 M 项 | N unit / M integration tests
- 构建产物 | Build artefacts:

## 风险与后续改进 | Risks & Next Steps
- 已知限制 | Known limitations:
- 建议迭代 | Suggested iterations:
```

---

## 9. 外部系统与数据交互约定 | External System & Data Interaction Guidelines
- **HTTP**：统一封装 client，包含重试、超时、重定向处理、错误码翻译与指标采集。<br>**HTTP**: Use a shared client wrapper with retries, timeouts, redirect handling, error translation, and metrics collection.
- **数据库**：使用迁移脚本（如 Alembic/Prisma），并通过 `docker-compose` 启动本地依赖。<br>**Databases**: Manage schema with migrations (Alembic/Prisma) and spin up dependencies via `docker-compose`.
- **消息/缓存**：提供本地 mock 或容器化服务（Kafka/Redis/RabbitMQ）。<br>**Messaging/cache**: Provide local mocks or containerised services (Kafka/Redis/RabbitMQ).
- **文件**：输入输出路径通过 `config/` 或 `.env` 配置，禁止硬编码绝对路径。<br>**File IO**: Configure paths via `config/` or `.env`; never hardcode absolute paths.

---

## 10. 典型脚本约定 | Script Conventions
- 示例脚本：`scripts/bootstrap.sh`、`scripts/dev_run.sh`、`scripts/seed_data.py`、`scripts/release.sh`。<br>Example scripts: `scripts/bootstrap.sh`, `scripts/dev_run.sh`, `scripts/seed_data.py`, `scripts/release.sh`.
- 所有脚本需支持 `-h/--help`，失败时返回非零退出码，并输出中文提示与下一步建议。<br>All scripts must support `-h/--help`, exit non-zero on failure, and print Chinese guidance with next-step hints.

---

## 11. 文档要求 | Documentation Requirements
- `README.md` 必含项目简介、功能清单、快速开始（≤3 条）、配置说明、常见问题。<br>`README.md` must include a project overview, feature list, quick start (≤3 commands), configuration notes, and FAQ.
- `docs/` 需包含 `architecture.md`（Mermaid 图）、`api.md`（接口定义）、`ops.md`（监控/告警/日志）。<br>`docs/` should provide `architecture.md` (with Mermaid diagrams), `api.md` (API definitions), and `ops.md` (monitoring/alerting/logging).
- 变更需更新 `CHANGELOG.md`（遵循 Keep a Changelog + SemVer）。<br>Update `CHANGELOG.md` for changes, following Keep a Changelog and SemVer conventions.

---

## 12. 完成标准 | Definition of Done
- [ ] `make ci` 全部通过。<br>`make ci` passes completely.
- [ ] `make run` 可运行最小示例。<br>`make run` launches a minimal example.
- [ ] 覆盖率 ≥ 80%，关键路径具备集成测试。<br>Coverage ≥80% with integration tests on critical paths.
- [ ] README/ASSUMPTIONS/CHANGELOG/agent_report 已更新。<br>README, ASSUMPTIONS, CHANGELOG, and agent_report are updated.
- [ ] 配置可通过 `.env` 切换环境，无敏感信息入库。<br>Configs are environment-switchable via `.env` with no secrets committed.
- [ ] 日志与错误信息可读，并包含排错建议。<br>Logs and errors are readable with troubleshooting suggestions.

---

## 13. 应对新需求的回复模板 | Response Template for New Requests
> **始终用中文简洁回复，可在需要时直接给出可运行的命令或代码块。**<br>> **Respond succinctly in Chinese, adding runnable commands or code snippets when helpful.**

1. **概述**：复述需求要点与关键假设。<br>**Overview**: Restate the requirements and note key assumptions.
2. **交付**：提供新增文件或补丁，并同步更新相关测试/文档。<br>**Deliverables**: Present new files or patches and update relevant tests/docs.
3. **运行**：给出 1～2 条验证命令。<br>**Run**: Offer one or two commands to validate the work.
4. **结果**：说明自测范围、覆盖率或关键日志。<br>**Result**: Summarise self-test scope, coverage, or key logs.
5. **后续**：列出可选优化项与影响评估。<br>**Next steps**: List optional improvements and impact assessments.

---

## 14. 最小示例（占位，可按项目替换）| Minimal Example (Placeholder)
- 运行顺序：<br>Run order:
  ```bash
  make setup
  make ci
  make run
  ```
- 若失败：查看 `agent_report.md` 的故障排查章节，执行 `make test -k failing_case` 复现问题。<br>If failures occur, review the troubleshooting section in `agent_report.md` and run `make test -k failing_case` to reproduce.

---

> **执行承诺 | Execution Promise**：除非明确要求暂停，代理将按本规范自动推进至“可运行 + 已自测 + 可交付”状态再输出结果。<br>> **Execution promise**: Unless explicitly told to pause, the agent proceeds until the work is runnable, self-tested, and deliverable before responding.
