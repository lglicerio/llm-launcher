#!/usr/bin/env bash
# llm-serve.sh — launch llama-server with a local GGUF model and wire it
# into aider and/or opencode via the OpenAI-compatible API.
#
# Usage:
#   ./llm-serve.sh                          # fully interactive
#   ./llm-serve.sh -m <path-to.gguf>        # skip model picker
#   ./llm-serve.sh -a aider|opencode|both|none
#   ./llm-serve.sh -c 32768 -p 8080 -H 127.0.0.1
#   ./llm-serve.sh --no-launch              # only configure + start server
#   ./llm-serve.sh --search "qwen3 coder"   # search & download a model from HF
#   ./llm-serve.sh --remove                 # interactively delete a local model
#   ./llm-serve.sh --attach -a aider        # attach to already-running server
#   ./llm-serve.sh --stop                   # stop a running llama-server
#   ./llm-serve.sh -h                       # help

set -euo pipefail

# ───────────────────────────── defaults ──────────────────────────────
HOST="127.0.0.1"
PORT="8080"
CTX="32768"
CTX_SET="no"                    # whether user supplied -c explicitly
NGL="999"                       # offload all layers to Metal on Apple Silicon
API_KEY="local-llama"           # dummy key; clients require a non-empty value
MODEL_PATH=""
AGENT=""
LAUNCH_AGENT="yes"
SEARCH_QUERY=""                 # if set, jump straight into search-and-download
REMOVE_ONLY="no"                # --remove: run the deletion flow and exit
ATTACH="no"                     # --attach: don't touch the server, just configure + launch agent
DOWNLOAD_DIR="$HOME/.lmstudio/models"   # where new models are saved (kept under lmstudio so the picker finds them)
PID_FILE="${TMPDIR:-/tmp}/llama-server.pid"
LOG_FILE="${TMPDIR:-/tmp}/llama-server.log"

MODEL_SEARCH_DIRS=(
    "$HOME/.lmstudio/models"
    "$HOME/.cache/llama.cpp"
    "$HOME/models"
    "$HOME/.cache/lm-studio/models"
)

# ───────────────────────────── helpers ───────────────────────────────
log()  { printf "\033[1;36m[llm-serve]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[llm-serve]\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m[llm-serve]\033[0m %s\n" "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing dependency: $1"; }

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

# Stop any running llama-server (pidfile-tracked + any strays) and block until
# the process(es) and the port are actually free. Without this wait the GPU
# memory from the previous model can still be held when we try to load the next
# one, causing OOM or "failed to load model" errors.
stop_server() {
    local -a pids=()
    local pid

    if [[ -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && pids+=("$pid")
        rm -f "$PID_FILE"
    fi
    # also catch strays (different port, orphaned pidfile, etc.)
    while IFS= read -r pid; do
        [[ -n "$pid" ]] && pids+=("$pid")
    done < <(pgrep -f "llama-server" 2>/dev/null || true)

    if [[ ${#pids[@]} -eq 0 ]]; then
        log "no running llama-server found"
        return 0
    fi

    # dedupe
    local -A seen=()
    local -a uniq=()
    for pid in "${pids[@]}"; do
        [[ -z "${seen[$pid]:-}" ]] && { uniq+=("$pid"); seen[$pid]=1; }
    done

    log "unloading previous llama-server: pids=${uniq[*]}"
    for pid in "${uniq[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done

    # wait up to 15s for graceful exit (model unload + GPU memory free)
    local i
    for i in $(seq 1 30); do
        local alive="no"
        for pid in "${uniq[@]}"; do
            kill -0 "$pid" 2>/dev/null && { alive="yes"; break; }
        done
        [[ "$alive" == "no" ]] && break
        sleep 0.5
    done

    # escalate to SIGKILL if anything is still alive
    for pid in "${uniq[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            warn "pid $pid did not exit on SIGTERM — sending SIGKILL"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    # also wait for the port to be released
    for i in $(seq 1 20); do
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || { log "previous server unloaded"; return 0; }
        sleep 0.5
    done
    warn "port $PORT still in use after stop — proceeding anyway"
}

# ─────────────────────────── arg parsing ─────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)    MODEL_PATH="$2"; shift 2 ;;
        -a|--agent)    AGENT="$2"; shift 2 ;;
        -c|--ctx)      CTX="$2"; CTX_SET="yes"; shift 2 ;;
        -p|--port)     PORT="$2"; shift 2 ;;
        -H|--host)     HOST="$2"; shift 2 ;;
        --ngl)         NGL="$2"; shift 2 ;;
        --no-launch)   LAUNCH_AGENT="no"; shift ;;
        --search)      SEARCH_QUERY="$2"; shift 2 ;;
        --remove)      REMOVE_ONLY="yes"; shift ;;
        --attach)      ATTACH="yes"; shift ;;
        --stop)        stop_server; exit 0 ;;
        -h|--help)     usage ;;
        *)             die "unknown arg: $1 (try --help)" ;;
    esac
done

need llama-server
need jq
need curl

# ──────────────────────── discover GGUF models ───────────────────────
discover_models() {
    local d
    for d in "${MODEL_SEARCH_DIRS[@]}"; do
        [[ -d "$d" ]] || continue
        find "$d" -type f -name "*.gguf" 2>/dev/null
    done | grep -Ev '/(mmproj|flux|nomic-embed|.*-embed|.*-projector)' | sort -u
}

# Probe the configured HOST:PORT for a running llama-server. Echoes the model
# alias it currently exposes (from /v1/models) on success; returns non-zero
# (and echoes nothing) if no server is reachable.
detect_running_server() {
    local url="http://${HOST}:${PORT}/v1/models"
    local resp
    # try with our API key first, then bare (in case the server was started without one)
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

# Hugging Face search → list repos tagged gguf → pick repo → list .gguf files → download chosen file.
search_and_download() {
    local query="${1:-}"
    if [[ -z "$query" ]]; then
        read -rp "search huggingface for: " query
    fi
    [[ -n "$query" ]] || { warn "empty query"; return 1; }

    local q_enc; q_enc=$(jq -rn --arg q "$query" '$q|@uri')
    log "searching huggingface for: $query"
    local results
    results=$(curl -sfG \
        --data-urlencode "search=$query" \
        --data "filter=gguf" \
        --data "limit=25" \
        --data "sort=downloads" \
        --data "direction=-1" \
        "https://huggingface.co/api/models") || die "huggingface search failed (check network)"

    local count; count=$(echo "$results" | jq 'length')
    (( count > 0 )) || { warn "no results for: $query"; return 1; }

    echo
    log "matching repos (sorted by downloads):"
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

    log "fetching file list for $repo"
    local tree
    tree=$(curl -sf "https://huggingface.co/api/models/$repo/tree/main?recursive=true") \
        || die "failed to list files for $repo"

    local -a files sizes
    mapfile -t files < <(echo "$tree" | jq -r '.[] | select(.type=="file" and (.path|endswith(".gguf"))) | .path')
    mapfile -t sizes < <(echo "$tree" | jq -r '.[] | select(.type=="file" and (.path|endswith(".gguf"))) | .size')
    [[ ${#files[@]} -gt 0 ]] || die "no .gguf files in $repo"

    echo
    log ".gguf files in $repo:"
    i=1
    for f in "${files[@]}"; do
        local hs; hs=$(human_size "${sizes[$((i-1))]}")
        printf "  \033[1;32m%2d)\033[0m %s  \033[2m(%s)\033[0m\n" "$i" "$f" "$hs"
        i=$((i+1))
    done
    echo

    while :; do
        read -rp "select file [1-${#files[@]}] (or 'q' to cancel): " choice
        [[ "$choice" == "q" ]] && return 1
        [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#files[@]} )) && break
        warn "invalid choice"
    done
    local path="${files[$((choice-1))]}"
    local url="https://huggingface.co/$repo/resolve/main/$path"
    local dest_dir="$DOWNLOAD_DIR/$repo"
    local dest="$dest_dir/$(basename "$path")"
    mkdir -p "$dest_dir"

    if [[ -f "$dest" ]]; then
        log "already present: $dest — using existing file"
    else
        log "downloading → $dest"
        log "url: $url"
        # -L: follow redirects (HF redirects to CDN); -C -: resume on retry; --fail: error on 4xx/5xx
        curl -L --fail -C - -o "$dest" "$url" || die "download failed"
    fi
    MODEL_PATH="$dest"
    log "model ready: $MODEL_PATH"
}

# Delete a local .gguf interactively, with a 'yes' confirmation. Cleans up
# now-empty parent directories up to (but not including) any MODEL_SEARCH_DIRS root.
remove_models() {
    local -a models
    mapfile -t models < <(discover_models)
    [[ ${#models[@]} -gt 0 ]] || { warn "no local models found to remove"; return 1; }

    echo
    log "local models:"
    local i=1
    for m in "${models[@]}"; do
        local size; size=$(du -h "$m" 2>/dev/null | awk '{print $1}')
        printf "  \033[1;31m%2d)\033[0m %s  \033[2m(%s)\033[0m\n" "$i" "$m" "$size"
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
    warn "about to permanently delete:"
    warn "  $target  ($(du -h "$target" 2>/dev/null | awk '{print $1}'))"
    local confirm
    read -rp "type 'yes' to confirm: " confirm
    [[ "$confirm" == "yes" ]] || { log "cancelled"; return 1; }

    rm -f "$target"
    log "deleted: $target"

    # walk up and rmdir any now-empty parents, stopping at search-dir roots
    local parent; parent=$(dirname "$target")
    while [[ "$parent" != "/" && -n "$parent" ]]; do
        local is_root="no"
        local d
        for d in "${MODEL_SEARCH_DIRS[@]}"; do
            [[ "$parent" == "$d" ]] && { is_root="yes"; break; }
        done
        [[ "$is_root" == "yes" ]] && break
        rmdir "$parent" 2>/dev/null || break   # rmdir fails on non-empty — exit loop cleanly
        log "removed empty dir: $parent"
        parent=$(dirname "$parent")
    done
}

pick_model() {
    while :; do
        local -a models
        mapfile -t models < <(discover_models)
        local running_alias; running_alias=$(detect_running_server || true)

        echo
        if [[ ${#models[@]} -eq 0 && -z "$running_alias" ]]; then
            warn "no local .gguf models found — entering search/download flow"
            search_and_download || die "model selection cancelled"
            return
        fi

        if [[ -n "$running_alias" ]]; then
            printf "  \033[1;36m a)\033[0m attach to currently running model: \033[1m%s\033[0m  \033[2m(%s)\033[0m\n" \
                "$running_alias" "${HOST}:${PORT}"
        fi

        if [[ ${#models[@]} -gt 0 ]]; then
            log "available local models:"
            local i=1
            local m
            for m in "${models[@]}"; do
                local size; size=$(du -h "$m" 2>/dev/null | awk '{print $1}')
                printf "  \033[1;32m%2d)\033[0m %s  \033[2m(%s)\033[0m\n" "$i" "$(basename "$m")" "$size"
                i=$((i+1))
            done
            printf "  \033[1;35m s)\033[0m search & download a new model from Hugging Face\n"
            printf "  \033[1;31m r)\033[0m remove a local model\n"
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
                ATTACH="yes"
                MODEL_ALIAS="$running_alias"
                return
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
                MODEL_PATH="${models[$((choice-1))]}"
                return
            fi
            warn "invalid choice"
        done
    done
}

prompt_ctx() {
    [[ "$CTX_SET" == "yes" ]] && return
    echo
    log "context window size — larger = more memory used; typical: 8192, 16384, 32768, 65536, 131072"
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

if [[ "$REMOVE_ONLY" == "yes" ]]; then
    remove_models
    exit $?
fi
if [[ -n "$SEARCH_QUERY" ]]; then
    search_and_download "$SEARCH_QUERY" || die "search/download cancelled"
fi

# --attach short-circuit: skip model picker, derive alias from the running server.
if [[ "$ATTACH" == "yes" && -z "${MODEL_ALIAS:-}" ]]; then
    detected="$(detect_running_server || true)"
    [[ -n "$detected" ]] || die "--attach: no llama-server reachable at ${HOST}:${PORT} (start one first, or omit --attach)"
    MODEL_ALIAS="$detected"
fi

if [[ "$ATTACH" != "yes" ]]; then
    [[ -z "$MODEL_PATH" ]] && pick_model
fi

# pick_model may have set ATTACH=yes — re-check before path validation
if [[ "$ATTACH" != "yes" ]]; then
    [[ -f "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"
    # alias = filename without .gguf, lowercased, friendly for API calls
    MODEL_ALIAS="$(basename "$MODEL_PATH" .gguf | tr '[:upper:]' '[:lower:]')"
fi

prompt_ctx
[[ -z "$AGENT" ]] && pick_agent

BASE_URL="http://${HOST}:${PORT}/v1"

if [[ "$ATTACH" == "yes" ]]; then
    log "mode:     ATTACH (using already-running server)"
    log "alias:    $MODEL_ALIAS"
else
    log "model:    $MODEL_PATH"
    log "alias:    $MODEL_ALIAS"
fi
log "endpoint: $BASE_URL"
log "context:  $CTX tokens"
log "agent:    $AGENT (launch=$LAUNCH_AGENT)"

# ────────────────────────── start llama-server ───────────────────────
if [[ "$ATTACH" == "yes" ]]; then
    log "attach mode — not touching the running server"
else
    # Always unload any previous model first — llama-server holds GPU memory and an
    # mmap on the gguf file, so loading a new model on top can OOM or fail.
    stop_server

    log "starting llama-server (log: $LOG_FILE)"
    # --jinja enables proper chat template handling for modern models like Qwen3
    nohup llama-server \
        -m "$MODEL_PATH" \
        --alias "$MODEL_ALIAS" \
        --host "$HOST" \
        --port "$PORT" \
        -c "$CTX" \
        -ngl "$NGL" \
        --jinja \
        --api-key "$API_KEY" \
        >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    # wait for /v1/models to respond (server warm-up can take a while for big models)
    log "waiting for server to become ready..."
    for i in $(seq 1 90); do
        if curl -sf -H "Authorization: Bearer $API_KEY" "$BASE_URL/models" >/dev/null 2>&1; then
            log "server ready (pid $(cat "$PID_FILE"))"
            break
        fi
        sleep 1
        if (( i == 90 )); then
            warn "server did not respond in 90s; tail of log:"
            tail -20 "$LOG_FILE" >&2
            die "llama-server failed to start"
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
# Generated by llm-serve.sh — points aider at the local llama-server.
model: ${model_name}
openai-api-base: ${BASE_URL}
openai-api-key: ${API_KEY}
# disable retries/telemetry for snappier local UX
stream: true
auto-commits: false
EOF

    log "writing $meta"
    local out_tokens=$(( CTX / 4 ))
    (( out_tokens > 8192 )) && out_tokens=8192
    local tmp; tmp="$(mktemp)"
    if [[ -f "$meta" ]]; then
        jq --arg k "$model_name" \
           --argjson ctx "$CTX" \
           --argjson out "$out_tokens" \
           '. + {($k): {max_input_tokens: $ctx, max_output_tokens: $out, input_cost_per_token: 0, output_cost_per_token: 0, litellm_provider: "openai"}}' \
           "$meta" > "$tmp"
    else
        jq -n --arg k "$model_name" \
              --argjson ctx "$CTX" \
              --argjson out "$out_tokens" \
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

    log "updating $cfg (provider: llama-server, model: $MODEL_ALIAS)"
    local tmp; tmp="$(mktemp)"
    # apiKey must be embedded in options — @ai-sdk/openai-compatible defaults to
    # reading process.env.OPENAI_API_KEY, which is unset when opencode launches.
    jq --arg base "$BASE_URL" \
       --arg key  "$API_KEY" \
       --arg alias "$MODEL_ALIAS" \
       --argjson ctx "$CTX" \
       --argjson out "$out_tokens" \
       '
       .provider = (.provider // {}) |
       .provider["llama-server"] = {
         npm: "@ai-sdk/openai-compatible",
         name: "llama.cpp (local)",
         options: { baseURL: $base, apiKey: $key },
         models: (
           ((.provider["llama-server"].models) // {}) +
           { ($alias): { name: $alias, limit: { context: $ctx, output: $out } } }
         )
       } |
       .model = ("llama-server/" + $alias)
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

# ────────────────── export env for current shell (optional) ──────────
ENV_FILE="${TMPDIR:-/tmp}/llm-serve.env"
cat > "$ENV_FILE" <<EOF
export OPENAI_API_BASE="$BASE_URL"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="$API_KEY"
export LLAMA_MODEL_ALIAS="$MODEL_ALIAS"
EOF
log "env written to $ENV_FILE — source it in your shell:"
echo "    source $ENV_FILE"

# ──────────────────────────── launch agent ───────────────────────────
if [[ "$LAUNCH_AGENT" == "yes" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    case "$AGENT" in
        aider)
            log "launching aider"
            exec aider
            ;;
        opencode)
            log "launching opencode"
            exec opencode
            ;;
    esac
fi

log "done. server running in background (pid $(cat "$PID_FILE"))."
log "stop it with:  $0 --stop"
log "tail logs:     tail -f $LOG_FILE"
