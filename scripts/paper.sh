#!/usr/bin/env bash
# 模拟交易/信号推送服务的启停脚本（服务器上由 shell 启动，见设计文档 §8.2）。
#
#   ./scripts/paper.sh start|stop|restart|status|once|logs
#
# 环境变量：
#   PAPER_SYMBOL / PAPER_STRATEGY  只跑指定品种/策略（默认 symbols.yaml 里全部 enabled）
#   PAPER_NO_PUSH=1                只落库不发飞书（演练用）
#   PAPER_NO_PULL=1                不访问交易所，只用本地库
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PID_FILE="run/paper.pid"
OUT_FILE="logs/paper.out"
ARGS=()

# 本机统一用 uv 管理依赖，但已存在的 .venv 必须复用：直接调它的入口，既省掉每次启动的
# uv 解析，也避免 uv 在环境缺失时静默新建一个。
if [[ -x ".venv/bin/quant" ]]; then
  QUANT=(".venv/bin/quant")
elif command -v uv >/dev/null 2>&1; then
  QUANT=(uv run quant)
else
  echo "找不到 .venv/bin/quant 也没有 uv，先执行 uv sync" >&2
  exit 1
fi

build_args() {
  ARGS=()
  [[ -z "${PAPER_SYMBOL:-}" ]] || ARGS+=(--symbol "$PAPER_SYMBOL")
  [[ -z "${PAPER_STRATEGY:-}" ]] || ARGS+=(--strategy "$PAPER_STRATEGY")
  [[ "${PAPER_NO_PUSH:-}" != "1" ]] || ARGS+=(--no-push)
  [[ "${PAPER_NO_PULL:-}" != "1" ]] || ARGS+=(--no-pull)
}

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  # 只认"确实是本服务的进程"：PID 会被复用，拿旧文件去 kill 会打到无关进程
  ps -p "$pid" -o command= 2>/dev/null | grep -q "quant paper" || return 1
  echo "$pid"
}

preflight() {
  local fail=0
  if [[ ! -f .env ]]; then
    echo "[preflight] 缺少 .env（FEISHU_WEBHOOK_URL / QUANT_PROXY_URL 都在里面）" >&2
    fail=1
  elif ! grep -q '^FEISHU_WEBHOOK_URL=..*' .env; then
    # 只报告键是否存在，绝不打印值
    echo "[preflight] .env 里 FEISHU_WEBHOOK_URL 为空" >&2
    fail=1
  fi
  # Binance 的境外域名境内直连不通，出网必须走代理
  if [[ -z "${QUANT_PROXY_URL:-}" ]] \
    && ! grep -q '^QUANT_PROXY_URL=..*' .env 2>/dev/null \
    && ! grep -q 'proxy_url: *http' config/settings.yaml 2>/dev/null; then
    echo "[preflight] 未配置代理，行情拉取会失败（settings.yaml network.proxy_url 或 .env QUANT_PROXY_URL）" >&2
    fail=1
  fi
  mkdir -p logs run
  return "$fail"
}

start() {
  preflight
  build_args
  local pid
  if pid="$(running_pid)"; then
    echo "已在运行（pid=$pid），如需改参数请执行 $0 restart"
    return 0
  fi
  echo "启动 paper 服务：${QUANT[*]} paper run ${ARGS[*]:-}"
  nohup "${QUANT[@]}" paper run "${ARGS[@]+"${ARGS[@]}"}" >> "$OUT_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
  if pid="$(running_pid)"; then
    echo "已启动（pid=$pid），日志 $OUT_FILE"
  else
    echo "启动失败，最后 20 行日志：" >&2
    tail -20 "$OUT_FILE" >&2 || true
    rm -f "$PID_FILE"
    return 1
  fi
}

stop() {
  local pid
  if ! pid="$(running_pid)"; then
    echo "未在运行"
    rm -f "$PID_FILE"
    return 0
  fi
  # 收到 TERM 后服务会在当前轮结束时退出；落库先于推送，留足收尾时间
  kill "$pid"
  for _ in $(seq 1 30); do
    if ! running_pid >/dev/null; then
      echo "已停止（pid=$pid）"
      rm -f "$PID_FILE"
      return 0
    fi
    sleep 1
  done
  echo "30 秒未退出，强制结束" >&2
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
}

status() {
  local pid
  if pid="$(running_pid)"; then
    echo "运行中（pid=$pid）"
  else
    echo "未运行"
  fi
  "${QUANT[@]}" paper status
}

once() {
  preflight
  build_args
  "${QUANT[@]}" paper once "${ARGS[@]+"${ARGS[@]}"}"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  once) once ;;
  logs) tail -n 100 -f "$OUT_FILE" ;;
  *) echo "用法: $0 start|stop|restart|status|once|logs" >&2; exit 2 ;;
esac
