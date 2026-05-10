# Agent Brief — Project Conventions

You are working in a ROS 2 workspace at `~/ros2_ws/src/ros2/`. Apply the rules below to any Python node or package you touch. This document is the source of truth for style and structural decisions in this repo.

---

## RULE 0 — DO NOT BREAK EXISTING FUNCTIONALITY (read this first, every time)

Every change made under these conventions is a **refactor**, not a rewrite. The runtime behaviour of every node must be identical before and after your changes. Style and structure improve; behaviour does not change.

**Before you change anything:**

1. Read the entire file you intend to modify, plus every file that imports from it.
2. Identify and write down (in the conversation, not in a doc) the observable behaviours that must be preserved:
   - Every published topic and message type.
   - Every subscribed topic and message type.
   - Every service offered or called.
   - Every parameter name and its effective default.
   - Every side effect on disk (files written, directories created).
   - Timer rates and QoS profiles.
3. Confirm with the user before deleting any file, parameter, or topic — even if it looks unused. "Looks unused" is not the same as "is unused"; external nodes, launch files in other packages, bag recordings, and dashboards may depend on it. Search the entire workspace (`grep -r`) before concluding something is dead.

**While you change things:**

4. Make the smallest change that satisfies the rule. Do not bundle unrelated cleanups.
5. If a rule appears to require a behaviour change (e.g. tightening a QoS profile, renaming a topic), STOP and surface it to the user as a question. Do not silently change behaviour under the banner of "consistency."
6. Preserve parameter names and topic names by default. Renaming is a separate, user-approved change.

**After you change things:**

7. Re-read the modified file end-to-end. Walk through the constructor, every callback, and the timer, and confirm each preserved behaviour from step 2 still holds.
8. Cross-check against importers: every symbol you renamed, moved, or removed — `grep -r` the workspace and update or revert.
9. Run `colcon build --packages-select <package>` and `colcon test --packages-select <package>` if available. If the user has a runtime test (a launch file they exercise), report what you changed and ask them to verify before moving on.
10. In your end-of-turn summary, list explicitly: (a) what changed, (b) what behaviours you verified are unchanged, (c) anything you could not verify and need the user to check.

**If in doubt, do less.** A partially-applied convention with preserved behaviour is always better than a fully-applied convention that broke a topic name some other node depended on.

---

## 1. Package metadata

- `setup.py` and `package.xml` agree on `maintainer`, `email`, `description`, `version`, `license`. Update both together.
- Every runtime import has a matching `<depend>` in `package.xml`. Every `<test_depend>` corresponds to a real test.
- `data_files` in `setup.py` only references paths inside the package. Runtime artefacts (logs, weights, recordings) live in `~/.ros/...` or a parameter-configured directory, never in `share/`.
- `__init__.py` re-exports a symbol only if external packages import it. Otherwise leave it empty.
- Empty directories (`config/`, `resource/`) get content or get deleted. No placeholder folders.

## 2. File layout & naming

- Module filenames are `snake_case.py`. No `PascalCase`, no `ALL_CAPS`, no mixed casing.
- One node per file; the file name matches the executable name in `setup.py` `entry_points`.
- Parameter names are `snake_case`, full words preferred over abbreviations. If you abbreviate one (`kp_rows`), abbreviate all peers consistently.
- Subscription callbacks are prefixed `_on_<topic>` workspace-wide. Service handlers are `_handle_<service>`. Timers are `_on_timer` or `_on_<purpose>_timer`. Legacy `cb_*` and `*_cb` schemes are migrated to `_on_*` when their file is touched.
- Topic and service names the node owns are declared as parameters with sensible defaults. Hardcoded topic strings are allowed only for well-known external topics (`/tf`, `/clock`); document why.

## 3. Node constructor structure

`__init__` is a dispatcher, not an implementation. It calls private setup helpers in this order:

```python
def __init__(self):
    super().__init__('node_name')
    self._declare_parameters()
    self._load_parameters()
    self._init_state()
    self._init_subsystems()   # network, computer, etc.
    self._setup_io()          # publishers, subscribers, services
    self._setup_logging()
    self._start_timers()
```

- Target: `__init__` body fits on one screen (~40 lines).
- All `declare_parameter` calls live in one method.
- Parameter reads (`get_parameter(...).value`) happen once in `_load_parameters` and are stored as `self.<name>`.
- Each piece of state is assigned exactly once in setup. No "default → overwrite → overwrite again."

## 4. Parameter discipline

- Every parameter declared must be read. Every parameter read must have been declared. No silent dead parameters.
- **All user-tunable parameters live in `my_ros2_bringup/config/params.yaml` and nowhere else.** This is the single source of truth for the workspace. Do not create per-package `config/*.yaml` files; do not duplicate values in launch files; do not bury defaults in code that contradict the YAML.
- `declare_parameter` defaults inside the node are safety nets only — they must match the YAML, and the YAML wins on disagreement.
- When adding a new parameter: declare it in code, add it to `params.yaml` under the node's section, and (if it is a value shared across nodes) define it as a YAML anchor in `_globals` so all consumers stay in sync automatically.
- Class-level defaults inside reusable algorithm classes are kept in sync with node-level defaults, or the class is changed to require the dict with no fallback.
- Path parameters use `pathlib.Path(...).expanduser()` and `mkdir(parents=True, exist_ok=True)` once. Never duplicate the resolution logic.

## 5. Constants & shared types

- Cross-node enums live in **one** module that all consumers import. The current owners are:
  - **Task states** (`SEARCH_ITEM`, `APPROACH_ITEM`, `SEARCH_DROPOFF`, `APPROACH_DROPOFF`) — owned by `task_manager`/`task_manager_interfaces`.
  - **Grab event codes** (`EVENT_IDLE`, `EVENT_GRABBED`, `EVENT_DROPPED`, `EVENT_BUSY`) — owned by `grab_node` (or promoted to `taskbot_interfaces` when reused).
  - **Action indices and `ACTION_NAMES`** — owned by `python_snn_node`.
- Importers re-import; they do not redefine. Re-declaring an enum with a different ordering is a bug, not a style issue.
- Magic numbers in callbacks (e.g. message field offsets like `data[2]`, `data[6]`, length checks like `len(data) < 14`) are named constants at module top.

## 6. Callback & timer hygiene

- **Two valid node shapes:**
  - *Timer-driven* (`python_snn_node`, `cmd_arbiter`, `motor_control`): subscription callbacks validate, store on `self`, return. The timer does the work. This is the default for any node that produces a continuous output stream.
  - *Event-driven* (`proximity_stop`, `dopamine_reward_node`): no control timer; logic runs in callbacks. Allowed when the node only emits in response to inputs. Each callback must be bounded, idempotent, and free of long-running work.
- Timer callbacks must complete within their period. Never call `get_logger().info(...)` per tick at >1 Hz — use `debug` or a throttled logger. The same applies to event-driven callbacks that fire at high rate.
- Every publisher and subscriber declares an **explicit** `QoSProfile` that matches the publisher on the other end. Default profile is `BEST_EFFORT` + `KEEP_LAST(1)` for high-rate sensor streams and `RELIABLE` + `KEEP_LAST(10)` for commands and state transitions. **Deviate from the default only when the upstream publisher's profile forces it, and add a one-line comment explaining why.** Do not reuse one `qos_sensor` profile object for unrelated topics.
- A timer that consumes inputs checks input freshness against `idle_timeout_sec` and publishes a safe-state command on staleness. The safe state is one helper, not duplicated logic.

## 7. Control flow in actuation

- Action-to-`Twist` mappings are data, not branching: a single table keyed by action index. Mode-specific behaviour (proximity, approach, normal) modifies the looked-up values; it does not re-implement the table.
- Decision branches are mutually exclusive and total. Every `if/elif` chain ends in an `else` that handles the unexpected case explicitly (log + safe stop), not silently.
- Unreachable branches are deleted, not commented out.

## 8. Logging

- Per-tick `info` logs are forbidden. Use `debug` (off by default) or rate-limited helpers. The CSV/file logger is for high-rate data; ROS logging is for events.
- A row builder for structured logs is one function. Mode/branch logic chooses *whether* to log, not *how* to assemble the row.
- CSV columns that are not populated must be removed from the header. No "always-zero" placeholder columns.
- Async loggers are closed in `destroy_node()` and report dropped-row counts on shutdown.

## 9. Comments & documentation

- Module docstring at the top of each node file lists: purpose, subscribed topics, published topics, services, parameters (or points at the YAML).
- Comment style is uniform within a file. Pick one banner style (`# --- Section ---`) and use it everywhere.
- Comments explain **why**, not **what**. Trailing markers like `###` on publish lines are removed.
- Educational / pedagogical content lives in `docs/` or a README, not inside utility module docstrings where it goes stale.
- Commented-out code is deleted. Git remembers.

## 10. Dead-code policy

- A module not imported by any node, launch file, or test is deleted — **after** confirming with the user (see Rule 0).
- Unused imports are removed on every commit that touches the file. Lint (`flake8`) must pass before merge.
- Parameters declared but never read are removed — **after** confirming nothing external sets them.

## 11. Async service flow

- Multi-step service chains use a state machine or a single `async def`, not a ladder of `add_done_callback` handlers. If callback chaining is unavoidable, document the chain in a comment listing the steps in order.
- Every `call_async` future has both a success and an error path, and the `busy` / in-flight flag is cleared in both.

## 12. Tests directory

- `test/` contains only files that `colcon test` should run. Experimental sweeps, plotting scripts, and one-off harnesses live in `scripts/` or `experiments/`.
- Lint stubs (`test_copyright.py`, `test_flake8.py`, `test_pep257.py`) stay only if they actually pass. A failing lint stub is fixed or removed; never silenced.

## 13. Launch files

- Launch files load parameters from `my_ros2_bringup/config/params.yaml` via the `p(node_name)` helper pattern. Do not redeclare defaults inside the launch file, and do not load any other YAML.
- If a launch file needs to override a value for a specific run, do it via a CLI argument (`DeclareLaunchArgument`) that the user can set at launch time — not by adding a second YAML.
- One launch file per logical unit; compose larger systems via `IncludeLaunchDescription`, not by copy-pasting `Node(...)` blocks.

---

## Pre-change checklist

Before opening any file for edit, confirm you can answer:

- [ ] Which other files import from this one? (`grep -r "from <module>" ~/ros2_ws/src/`)
- [ ] Which topics, services, and parameters does this node expose?
- [ ] Are this node's parameters all in `my_ros2_bringup/config/params.yaml`? (Reject any per-package YAML you find — flag it for consolidation.)
- [ ] Are there bag recordings, dashboards, or external scripts that depend on the current names?

## Post-change checklist

Before reporting the work as done:

- [ ] Topic names, types, and QoS unchanged (or change explicitly approved by user).
- [ ] Service names and signatures unchanged (or change explicitly approved).
- [ ] Parameter names and default values unchanged (or change explicitly approved).
- [ ] Files written to disk land at the same paths as before.
- [ ] Timer rates unchanged.
- [ ] `colcon build --packages-select <package>` succeeds.
- [ ] `colcon test --packages-select <package>` succeeds (if tests exist).
- [ ] All importers of any renamed/moved symbol updated.
- [ ] Summary message lists what changed, what was verified unchanged, and what the user should manually verify.

## Operating principles

1. **Refactor, don't rewrite.** Preserve behaviour byte-for-byte where possible.
2. **Smallest viable change.** Do not bundle unrelated cleanups into one edit.
3. **Ask before deleting.** Files, parameters, topics, and services may have invisible consumers.
4. **Verify twice.** Once before editing (what must I preserve?), once after (did I preserve it?).
5. **Surface uncertainty.** If you cannot verify a behaviour is preserved, say so — do not assume.
