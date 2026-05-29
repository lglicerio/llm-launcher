#!/usr/bin/env bash
# llm-serve-mlx.sh — launch mlx_lm.server with a local MLX model and wire it
# into aider and/or opencode via the OpenAI-compatible API.
#
# MLX vs llama.cpp: MLX is Apple-native and typically 40–80% faster than
# llama.cpp on Apple Silicon. Models are *directories* (config.json +
# safetensors), not single .gguf files, and live under mlx-community/ on HF.
#
# Usage:
#   ./llm-serve-mlx.sh                       # fully interactive
#   ./llm-serve-mlx.sh -m <repo-or-dir>      # skip model picker (path OR HF id)
#   ./llm-serve-mlx.sh -a aider|opencode|both|none
#   ./llm-serve-mlx.sh -c 32768 -p 8080 -H 127.0.0.1
#   ./llm-serve-mlx.sh --search "qwen3"      # search & download MLX model from HF
#   ./llm-serve-mlx.sh --remove              # interactively delete a local MLX model
#   ./llm-serve-mlx.sh --attach -a aider     # attach to already-running server
#   ./llm-serve-mlx.sh --no-launch           # configure + start server only
#   ./llm-serve-mlx.sh --stop                # stop a running mlx_lm.server
#   ./llm-serve-mlx.sh -h                    # help

set -euo pipefail

# ───────────────────────────── defaults ──────────────────────────────
HOST="127.0.0.1"
PORT="8080"
CTX="32768"
CTX_SET="no"
API_KEY="local-mlx"             # mlx_lm.server doesn't auth, but clients need a non-empty key
MODEL_REF=""                    # absolute path to local dir, or HF repo id (org/name)
AGENT=""
LAUNCH_AGENT="yes"
SEARCH_QUERY=""
REMOVE_ONLY="no"
ATTACH="no"
DOWNLOAD_DIR="$HOME/.lmstudio/models"
PID_FILE="${TMPDIR:-/tmp}/mlx-server.pid"
LOG_FILE="${TMPDIR:-/tmp}/mlx-server.log"

# Directories scanned for MLX model dirs. A "model dir" = contains config.json + *.safetensors.
MODEL_SEARCH_DIRS=(
    "$HOME/.lmstudio/models"
    "$HOME/.cache/huggingface/hub"
    "$HOME/models-mlx"
    "$HOME/models"
)

# ───────────────────────────── helpers ───────────────────────────────
log()  { printf "\033[1;36m[mlx-serve]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[mlx-serve]\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m[mlx-serve]\033[0m %s\n" "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"; }

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

# Resolve the right way to invoke the MLX server (script entrypoint or `python -m`).
MLX_SERVER_CMD=""
resolve_mlx_cmd() {
    if command -v mlx_lm.server >/dev/null 2>&1; then
        MLX_SERVER_CMD="mlx_lm.server"
    elif python3 -c "import mlx_lm.server" 2>/dev/null; then
        MLX_SERVER_CMD="python3 -m mlx_lm.server"
    else
        die "mlx-lm not installed. install with one of:
    pipx install mlx-lm        (recommended — isolated)
    pip3 install --user mlx-lm
    uv tool install mlx-lm"
    fi
}

# Stop any running mlx_lm.server and block until it's fully gone (frees GPU mem).
stop_server() {
    local -a pids=()
    local pid

    if [[ -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && pids+=("$pid")
        rm -f "$PID_FILE"
    fi
    while IFS= read -r pid; do
        [[ -n "$pid" ]] && pids+=("$pid")
    done < <(pgrep -f "mlx_lm.server|mlx_lm\.server" 2>/dev/null || true)

    if [[ ${#pids[@]} -eq 0 ]]; then
        log "no running mlx server found"
        return 0
    fi

    local -A seen=(); local -a uniq=()
    for pid in "${pids[@]}"; do
        [[ -z "${seen[$pid]:-}" ]] && { uniq+=("$pid"); seen[$pid]=1; }
    done

    log "unloading previous mlx server: pids=${uniq[*]}"
    for pid in "${uniq[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done

    local i
    for i in $(seq 1 30); do
        local alive="no"
        for pid in "${uniq[@]}"; do
            kill -0 "$pid" 2>/dev/null && { alive="yes"; break; }
        done
        [[ "$alive" == "no" ]] && break
        sleep 0.5
    done
    for pid in "${uniq[@]}"; do
        kill -0 "$pid" 2>/dev/null && { warn "pid $pid did not exit on SIGTERM — sending SIGKILL"; kill -KILL "$pid" 2>/dev/null || true; }
    done
    for i in $(seq 1 20); do
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || { log "previous server unloaded"; return 0; }
        sleep 0.5
    done
    warn "port $PORT still in use after stop — proceeding anyway"
}

# ─────────────────────────── arg parsing ─────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)    MODEL_REF="$2"; shift 2 ;;
        -a|--agent)    AGENT="$2"; shift 2 ;;
        -c|--ctx)      CTX="$2"; CTX_SET="yes"; shift 2 ;;
        -p|--port)     PORT="$2"; shift 2 ;;
        -H|--host)     HOST="$2"; shift 2 ;;
        --no-launch)   LAUNCH_AGENT="no"; shift ;;
        --search)      SEARCH_QUERY="$2"; shift 2 ;;
        --remove)      REMOVE_ONLY="yes"; shift ;;
        --attach)      ATTACH="yes"; shift ;;
        --stop)        stop_server; exit 0 ;;
        -h|--help)     usage ;;
        *)             die "unknown arg: $1 (try --help)" ;;
    esac
done

need jq
need curl
need python3

# ──────────────────────── discover MLX models ────────────────────────
# An MLX model dir contains config.json AND at least one *.safetensors file.
# We hunt under known roots, dedupe, and skip mmproj/projector dirs.
discover_models() {
    local d
    for d in "${MODEL_SEARCH_DIRS[@]}"; do
        [[ -d "$d" ]] || continue
        # find every config.json, then check its dir for safetensors
        while IFS= read -r cfg; do
            local model_dir; model_dir="$(dirname "$cfg")"
            if compgen -G "$model_dir/*.safetensors" >/dev/null; then
                echo "$model_dir"
            fi
        done < <(find "$d" -maxdepth 6 -type f -name config.json 2>/dev/null)
    done | sort -u
}

# Friendly alias from a model dir. For HF cache paths like
# models--mlx-community--Qwen3.6-30B-A3B/snapshots/<hash>/, return "Qwen3.6-30B-A3B".
# For other paths, return the basename.
alias_from_dir() {
    local d="$1"
    local base; base="$(basename "$d")"
    # HF cache: <root>/models--<org>--<repo>/snapshots/<hash>
    if [[ "$d" =~ /models--[^/]+--([^/]+)/snapshots/[^/]+/?$ ]]; then
        base="${BASH_REMATCH[1]}"
    fi
    echo "$base" | tr '[:upper:]' '[:lower:]'
}

# Read max_position_embeddings from a model's config.json, or empty if unavailable.
ctx_from_dir() {
    local d="$1"
    [[ -f "$d/config.json" ]] || { echo ""; return; }
    jq -r '.max_position_embeddings // .max_seq_len // empty' "$d/config.json" 2>/dev/null
}

detect_running_server() {
    local url="http://${HOST}:${PORT}/v1/models"
    local resp
    resp=$(curl -sf --max-time 2 -H "Authorization: Bearer $API_KEY" "$url" 2>/dev/null) \
        || resp=$(curl -sf --max-time 2 "$url" 2>/dev/null) \
        || return 1
    echo "$resp" | jq -r '.data[0].id // empty'
}

human_size() {
    awk -v b="$1" 'BEGIN{
        split("B K M G T",u);
        for(i=5;i>=1;i--){ s=1024^(i-1); if(b>=s){ printf "%.1f%s", b/s, u[i]; exit } }
        printf "%dB", b
    }'
}

dir_size_bytes() {
    # portable byte sum across BSD/GNU du
    du -sk "$1" 2>/dev/null | awk '{print $1 * 1024}'
}

# HF search restricted to MLX models (library filter + mlx-community author bias).
# Then snapshot_download via the huggingface_hub python lib that ships with mlx-lm.
search_and_download() {
    local query="${1:-}"
    if [[ -z "$query" ]]; then
        read -rp "search huggingface (MLX models) for: " query
    fi
    [[ -n "$query" ]] || { warn "empty query"; return 1; }

    log "searching huggingface (library=mlx) for: $query"
    local results
    results=$(curl -sfG \
        --data-urlencode "search=$query" \
        --data "library=mlx" \
        --data "limit=25" \
        --data "sort=downloads" \
        --data "direction=-1" \
        "https://huggingface.co/api/models") || die "huggingface search failed"

    local count; count=$(echo "$results" | jq 'length')
    if (( count == 0 )); then
        warn "no library=mlx results — retrying with author=mlx-community"
        results=$(curl -sfG \
            --data-urlencode "search=$query" \
            --data "author=mlx-community" \
            --data "limit=25" \
            --data "sort=downloads" \
            --data "direction=-1" \
            "https://huggingface.co/api/models") || die "huggingface search failed"
        count=$(echo "$results" | jq 'length')
    fi
    (( count > 0 )) || { warn "no MLX repos found for: $query"; return 1; }

    echo
    log "matching MLX repos (sorted by downloads):"
    local -a repos
    mapfile -t repos < <(echo "$results" | jq -r '.[].id')
    local i=1
    while IFS=$'\t' read -r id downloads; do
        printf "  \033[1;32m%2d)\033[0m %s  \033[2m(%s downloads)\033[0m\n" "$i" "$id" "$downloads"
        i=$((i+1))
    done < <(echo "$results" | jq -r '.[] | "\(.id)\t\(.downloads // 0)"')
    echo

    local choice
    while :; do
        read -rp "select repo [1-${#repos[@]}] (or 'q' to cancel): " choice
        [[ "$choice" == "q" ]] && return 1
        [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#repos[@]} )) && break
        warn "invalid choice"
    done
    local repo="${repos[$((choice-1))]}"

    # Pre-warm download via huggingface_hub. We snapshot into ~/.lmstudio/models/<repo>/
    # so the local picker finds it later.
    local local_dir="$DOWNLOAD_DIR/$repo"
    mkdir -p "$local_dir"
    log "downloading $repo → $local_dir"
    hf_snapshot_download "$repo" "$local_dir" || die "download failed"
    MODEL_REF="$local_dir"
    log "model ready: $MODEL_REF"
}

# Run snapshot_download in whatever Python actually has huggingface_hub.
# Priority: (1) uv ephemeral env, (2) the mlx-lm tool venv, (3) plain python3.
# The user's system python3 (Homebrew 3.14) won't have huggingface_hub installed
# directly — only the mlx-lm install does — so we route through uv when possible.
hf_snapshot_download() {
    local repo="$1" local_dir="$2"
    local script
    script=$(cat <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$repo", local_dir="$local_dir",
                  allow_patterns=["*.json","*.safetensors","*.txt","tokenizer*","*.model","*.tiktoken"])
PY
)

    if command -v uv >/dev/null 2>&1; then
        # --no-project: don't try to resolve a pyproject in $PWD
        # --with: install huggingface_hub into the ephemeral env
        echo "$script" | uv run --no-project --with huggingface_hub --quiet python -
        return $?
    fi

    if python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "$script" | python3 -
        return $?
    fi

    warn "neither uv nor a python with huggingface_hub is available"
    warn "install one of:  uv tool install mlx-lm    |    pip3 install --user huggingface_hub"
    return 1
}

# Interactively delete a local MLX model dir (with 'yes' confirmation).
remove_models() {
    local -a models
    mapfile -t models < <(discover_models)
    [[ ${#models[@]} -gt 0 ]] || { warn "no local MLX models found"; return 1; }

    echo
    log "local MLX models:"
    local i=1
    for m in "${models[@]}"; do
        local sz; sz=$(human_size "$(dir_size_bytes "$m")")
        printf "  \033[1;31m%2d)\033[0m %s  \033[2m(%s)\033[0m\n" "$i" "$m" "$sz"
        i=$((i+1))
    done
    echo

    local choice
    while :; do
        read -rp "select model to delete [1-${#models[@]}] (or 'q' to cancel): " choice
        [[ "$choice" == "q" ]] && { log "cancelled"; return 1; }
        [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#models[@]} )) && break
        warn "invalid choice"
    done
    local target="${models[$((choice-1))]}"

    echo
    warn "about to permanently delete directory:"
    warn "  $target  ($(human_size "$(dir_size_bytes "$target")"))"
    local confirm; read -rp "type 'yes' to confirm: " confirm
    [[ "$confirm" == "yes" ]] || { log "cancelled"; return 1; }

    rm -rf "$target"
    log "deleted: $target"

    # walk up and rmdir empty parents, stopping at search roots
    local parent; parent="$(dirname "$target")"
    while [[ "$parent" != "/" && -n "$parent" ]]; do
        local is_root="no"
        for d in "${MODEL_SEARCH_DIRS[@]}"; do
            [[ "$parent" == "$d" ]] && { is_root="yes"; break; }
        done
        [[ "$is_root" == "yes" ]] && break
        rmdir "$parent" 2>/dev/null || break
        log "removed empty dir: $parent"
        parent="$(dirname "$parent")"
    done
}

pick_model() {
    while :; do
        local -a models
        mapfile -t models < <(discover_models)
        local running_alias; running_alias=$(detect_running_server || true)

        echo
        if [[ ${#models[@]} -eq 0 && -z "$running_alias" ]]; then
            warn "no local MLX models found — entering search/download flow"
            search_and_download || die "model selection cancelled"
            return
        fi

        if [[ -n "$running_alias" ]]; then
            printf "  \033[1;36m a)\033[0m attach to currently running model: \033[1m%s\033[0m  \033[2m(%s)\033[0m\n" \
                "$running_alias" "${HOST}:${PORT}"
        fi

        if [[ ${#models[@]} -gt 0 ]]; then
            log "available local MLX models:"
            local i=1; local m
            for m in "${models[@]}"; do
                local sz; sz=$(human_size "$(dir_size_bytes "$m")")
                printf "  \033[1;32m%2d)\033[0m %s  \033[2m(%s)\033[0m\n" "$i" "$(alias_from_dir "$m")" "$sz"
                i=$((i+1))
            done
            printf "  \033[1;35m s)\033[0m search & download a new MLX model from Hugging Face\n"
            printf "  \033[1;31m r)\033[0m remove a local MLX model\n"
        fi
        echo

        local choice prompt
        if [[ -n "$running_alias" ]]; then
            prompt="select [1-${#models[@]}], 'a' (attach), 's', or 'r': "
        else
            prompt="select model [1-${#models[@]}], 's', or 'r': "
        fi

        while :; do
            read -rp "$prompt" choice
            if [[ "$choice" == "a" || "$choice" == "A" ]] && [[ -n "$running_alias" ]]; then
                ATTACH="yes"; MODEL_ALIAS="$running_alias"; return
            fi
            if [[ "$choice" == "s" || "$choice" == "S" ]]; then
                search_and_download && return
                break
            fi
            if [[ "$choice" == "r" || "$choice" == "R" ]]; then
                remove_models || true
                break
            fi
            if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#models[@]} )); then
                MODEL_REF="${models[$((choice-1))]}"
                return
            fi
            warn "invalid choice"
        done
    done
}

prompt_ctx() {
    [[ "$CTX_SET" == "yes" ]] && return
    # If we have a local dir, use config.json's max_position_embeddings as the default.
    if [[ -n "${MODEL_REF:-}" && -d "$MODEL_REF" ]]; then
        local model_ctx; model_ctx=$(ctx_from_dir "$MODEL_REF")
        [[ -n "$model_ctx" && "$model_ctx" =~ ^[0-9]+$ ]] && CTX="$model_ctx"
    fi
    echo
    log "context window (informational — mlx_lm.server uses the model's native ctx; this is what we tell the agents)"
    log "typical: 8192, 16384, 32768, 65536, 131072"
    local input
    read -rp "context size [default ${CTX}]: " input
    if [[ -n "$input" ]]; then
        [[ "$input" =~ ^[0-9]+$ ]] && (( input >= 512 )) || die "invalid context size: $input"
        CTX="$input"
    fi
}

pick_agent() {
    echo
    log "which agent should I configure/launch?"
    echo "  1) aider"
    echo "  2) opencode"
    echo "  3) both (configure both, launch none)"
    echo "  4) none  (configure both, launch none)"
    echo
    local choice
    while :; do
        read -rp "select agent [1-4]: " choice
        case "$choice" in
            1) AGENT="aider";    return ;;
            2) AGENT="opencode"; return ;;
            3) AGENT="both";     LAUNCH_AGENT="no"; return ;;
            4) AGENT="none";     LAUNCH_AGENT="no"; return ;;
            *) warn "invalid choice" ;;
        esac
    done
}

# ─────────────────────────────── main ────────────────────────────────
if [[ "$REMOVE_ONLY" == "yes" ]]; then
    remove_models; exit $?
fi
if [[ -n "$SEARCH_QUERY" ]]; then
    search_and_download "$SEARCH_QUERY" || die "search/download cancelled"
fi

if [[ "$ATTACH" == "yes" && -z "${MODEL_ALIAS:-}" ]]; then
    detected="$(detect_running_server || true)"
    [[ -n "$detected" ]] || die "--attach: no mlx_lm.server reachable at ${HOST}:${PORT}"
    MODEL_ALIAS="$detected"
fi

if [[ "$ATTACH" != "yes" ]]; then
    [[ -z "$MODEL_REF" ]] && pick_model
fi

if [[ "$ATTACH" != "yes" ]]; then
    # MODEL_REF can be a local dir or an HF repo id. We only validate paths.
    if [[ "$MODEL_REF" == /* || "$MODEL_REF" == ./* || "$MODEL_REF" == ~/* ]]; then
        MODEL_REF="${MODEL_REF/#~/$HOME}"
        [[ -d "$MODEL_REF" ]] || die "model dir not found: $MODEL_REF"
        MODEL_ALIAS="$(alias_from_dir "$MODEL_REF")"
    else
        # treated as HF repo id — alias is the repo name portion
        MODEL_ALIAS="$(basename "$MODEL_REF" | tr '[:upper:]' '[:lower:]')"
    fi
fi

prompt_ctx
[[ -z "$AGENT" ]] && pick_agent

BASE_URL="http://${HOST}:${PORT}/v1"

if [[ "$ATTACH" == "yes" ]]; then
    log "mode:     ATTACH"
    log "alias:    $MODEL_ALIAS"
else
    log "model:    $MODEL_REF"
    log "alias:    $MODEL_ALIAS"
fi
log "endpoint: $BASE_URL"
log "context:  $CTX tokens"
log "agent:    $AGENT (launch=$LAUNCH_AGENT)"

# ────────────────────────── start mlx server ─────────────────────────
if [[ "$ATTACH" == "yes" ]]; then
    log "attach mode — not touching the running server"
else
    resolve_mlx_cmd
    stop_server

    log "starting mlx server: $MLX_SERVER_CMD (log: $LOG_FILE)"
    # mlx_lm.server: --model accepts HF repo id or local dir; no --api-key or -ngl flags.
    # shellcheck disable=SC2086
    nohup $MLX_SERVER_CMD \
        --model "$MODEL_REF" \
        --host "$HOST" \
        --port "$PORT" \
        --log-level INFO \
        >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    # MLX cold loads can take a while; first-time HF downloads even longer.
    log "waiting for server to become ready (this can take a few minutes on first load)..."
    for i in $(seq 1 600); do
        if curl -sf --max-time 2 "$BASE_URL/models" >/dev/null 2>&1; then
            log "server ready (pid $(cat "$PID_FILE"))"
            break
        fi
        sleep 1
        if (( i == 600 )); then
            warn "server did not respond in 10 min; tail of log:"
            tail -30 "$LOG_FILE" >&2
            die "mlx server failed to start"
        fi
    done
fi

# ─────────────────────────── configure aider ─────────────────────────
configure_aider() {
    local conf="$HOME/.aider.conf.yml"
    local meta="$HOME/.aider.model.metadata.json"
    local model_name="openai/${MODEL_ALIAS}"

    log "writing $conf"
    cat > "$conf" <<EOF
# Generated by llm-serve-mlx.sh — points aider at the local mlx_lm.server.
model: ${model_name}
openai-api-base: ${BASE_URL}
openai-api-key: ${API_KEY}
stream: true
auto-commits: false
EOF

    log "writing $meta"
    local out_tokens=$(( CTX / 4 ))
    (( out_tokens > 8192 )) && out_tokens=8192
    local tmp; tmp="$(mktemp)"
    if [[ -f "$meta" ]]; then
        jq --arg k "$model_name" --argjson ctx "$CTX" --argjson out "$out_tokens" \
           '. + {($k): {max_input_tokens: $ctx, max_output_tokens: $out, input_cost_per_token: 0, output_cost_per_token: 0, litellm_provider: "openai"}}' \
           "$meta" > "$tmp"
    else
        jq -n --arg k "$model_name" --argjson ctx "$CTX" --argjson out "$out_tokens" \
              '{($k): {max_input_tokens: $ctx, max_output_tokens: $out, input_cost_per_token: 0, output_cost_per_token: 0, litellm_provider: "openai"}}' \
           > "$tmp"
    fi
    mv "$tmp" "$meta"
}

# ────────────────────────── configure opencode ───────────────────────
configure_opencode() {
    local cfg_dir="$HOME/.config/opencode"
    local cfg="$cfg_dir/opencode.json"
    mkdir -p "$cfg_dir"
    [[ -f "$cfg" ]] || echo '{"$schema":"https://opencode.ai/config.json"}' > "$cfg"

    local out_tokens=$(( CTX / 4 ))
    (( out_tokens > 8192 )) && out_tokens=8192

    log "updating $cfg (provider: mlx, model: $MODEL_ALIAS)"
    local tmp; tmp="$(mktemp)"
    jq --arg base "$BASE_URL" \
       --arg key  "$API_KEY" \
       --arg alias "$MODEL_ALIAS" \
       --argjson ctx "$CTX" \
       --argjson out "$out_tokens" \
       '
       .provider = (.provider // {}) |
       .provider["mlx"] = {
         npm: "@ai-sdk/openai-compatible",
         name: "mlx_lm.server (local)",
         options: { baseURL: $base, apiKey: $key },
         models: (
           ((.provider["mlx"].models) // {}) +
           { ($alias): { name: $alias, limit: { context: $ctx, output: $out } } }
         )
       } |
       .model = ("mlx/" + $alias)
       ' "$cfg" > "$tmp"
    mv "$tmp" "$cfg"
}

case "$AGENT" in
    aider)    configure_aider ;;
    opencode) configure_opencode ;;
    both)     configure_aider; configure_opencode ;;
    none)     log "skipping agent config" ;;
    *)        die "unknown agent: $AGENT" ;;
esac

# ─────────────────── export env for current shell ────────────────────
ENV_FILE="${TMPDIR:-/tmp}/llm-serve-mlx.env"
cat > "$ENV_FILE" <<EOF
export OPENAI_API_BASE="$BASE_URL"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="$API_KEY"
export MLX_MODEL_ALIAS="$MODEL_ALIAS"
EOF
log "env written to $ENV_FILE — source it in your shell:"
echo "    source $ENV_FILE"

# ──────────────────────────── launch agent ───────────────────────────
if [[ "$LAUNCH_AGENT" == "yes" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    case "$AGENT" in
        aider)    log "launching aider";    exec aider ;;
        opencode) log "launching opencode"; exec opencode ;;
    esac
fi

if [[ "$ATTACH" != "yes" ]]; then
    log "done. server running in background (pid $(cat "$PID_FILE"))."
    log "stop it with:  $0 --stop"
    log "tail logs:     tail -f $LOG_FILE"
else
    log "done. agents configured against existing server at $BASE_URL"
fi
