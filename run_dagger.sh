#!/bin/bash
# Launch EMAC+ DAgger server + client in tmux.
# Usage:
#   bash run_dagger.sh              # interactive (attaches to tmux)
#   bash run_dagger.sh --background # detached mode for pipeline automation

SESSION="emac"
BACKGROUND=false

if [[ "$1" == "--background" ]]; then
    BACKGROUND=true
fi

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

# Clean up stale ai2thor FIFO pipes from previous crashed runs
rm -f /tmp/ai2thor-fifo/* 2>/dev/null

tmux new-session -d -s "$SESSION"

# Pane 1: DAgger server
tmux send-keys "conda activate emac && xvfb-run -a python dagger_server.py" Enter

# Pane 2: ALFWorld client (wait for server to start)
tmux split-window -h
tmux send-keys "conda activate emac && sleep 15 && xvfb-run -a python alfworld_client.py" Enter

# Pane 3: Log tail
tmux split-window -v
tmux send-keys "sleep 25 && tail -f output/dagger_server_human_desc/*/running_nb01.log" Enter

if [ "$BACKGROUND" = true ]; then
    echo "DAgger session '$SESSION' started in background."
    echo "Use 'tmux attach -t $SESSION' to monitor."
else
    tmux attach -t "$SESSION"
fi
