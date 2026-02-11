#!/usr/bin/env bash
# Create a git worktree for an experiment branch, sharing data/.
#
# Usage:
#   ./scripts/worktree.sh physics_v1
#   ./scripts/worktree.sh physics_v1 /tmp/weatherstat-physics
#
# Creates:
#   - Branch: experiment/{name} (from current HEAD)
#   - Worktree at ../weatherstat-{name} (or custom path)
#   - Symlink: {worktree}/data -> {main}/data (shared snapshots, models, predictions)
#
# The collector and control loop keep running on main. The experiment worktree
# shares the same data directory so it trains on the same snapshots.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment-name> [worktree-path]"
    echo ""
    echo "Examples:"
    echo "  $0 physics_v1"
    echo "  $0 mpc_prototype /tmp/weatherstat-mpc"
    exit 1
fi

NAME="$1"
BRANCH="experiment/${NAME}"
WORKTREE="${2:-${REPO_ROOT}/../weatherstat-${NAME}}"

# Check if branch already exists
if git -C "$REPO_ROOT" rev-parse --verify "$BRANCH" &>/dev/null; then
    echo "Branch $BRANCH already exists"
else
    echo "Creating branch $BRANCH from HEAD..."
    git -C "$REPO_ROOT" branch "$BRANCH"
fi

# Check if worktree already exists
if [ -d "$WORKTREE" ]; then
    echo "Worktree already exists at $WORKTREE"
    exit 1
fi

# Create worktree
echo "Creating worktree at $WORKTREE..."
git -C "$REPO_ROOT" worktree add "$WORKTREE" "$BRANCH"

# Remove the worktree's own data directory (if any) and symlink to main's
if [ -d "$WORKTREE/data" ]; then
    rm -rf "$WORKTREE/data"
fi
ln -s "$REPO_ROOT/data" "$WORKTREE/data"
echo "Symlinked data/ -> $REPO_ROOT/data"

# Install dependencies in the worktree
echo ""
echo "Installing dependencies..."
(cd "$WORKTREE/ha-client" && pnpm install --frozen-lockfile 2>/dev/null) || true
(cd "$WORKTREE/ml" && uv sync 2>/dev/null) || true

echo ""
echo "Done! Experiment worktree ready:"
echo "  Path:   $WORKTREE"
echo "  Branch: $BRANCH"
echo "  Data:   shared via symlink"
echo ""
echo "Workflow:"
echo "  cd $WORKTREE"
echo "  # Edit ml/src/weatherstat/features.py (or wherever)"
echo "  just train-experiment $NAME          # train to data/models/$NAME/"
echo "  just experiment-compare $NAME        # compare vs production"
echo "  # If better: merge branch, just train to promote to production"
