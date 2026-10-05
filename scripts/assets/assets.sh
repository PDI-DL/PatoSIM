#!/bin/bash
set -e

# Configuration
HF_REPO_ID="PDI-DL/PDI_3DUW"
LOCAL_FOLDER="../../assets/models"

usage() {
    echo ""
    echo "Usage: ./assets.sh [command]"
    echo ""
    echo "Commands:"
    echo "  -auth       Authenticate with Hugging Face (run once)"
    echo "  -download   Clone the remote dataset into ${LOCAL_FOLDER}"
    echo "  -upload     Push all local changes to the remote dataset"
    echo "  -status     Show pending changes without uploading"
    echo "  -help       Show this message"
    echo ""
}

get_token() {
    local token
    token=$(cat ~/.cache/huggingface/token 2>/dev/null)
    if [ -z "$token" ]; then
        echo "### Error: no HuggingFace token found. Run ./assets.sh -auth first."
        exit 1
    fi
    echo "$token"
}

check_git_repo() {
    if [ ! -d "${LOCAL_FOLDER}/.git" ]; then
        echo "### Error: '${LOCAL_FOLDER}' is not a git repo. Run ./assets.sh -download first."
        exit 1
    fi
}

check_git_xet() {
    if ! command -v git-xet &>/dev/null; then
        echo "### Error: git-xet is not installed."
        echo "# Install it with:"
        echo "    curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh"
        echo "    git xet install"
        exit 1
    fi
}

cmd_auth() {
    echo ""
    echo "# Authenticating with Hugging Face..."
    hf auth login || echo "### Authentication failed. Check if you have the Hugging Face CLI installed or if you are already logged in. 
    If you are already logged in dismiss this message.
    If you are not logged in install hf-cli with: 
    curl -LsSf https://hf.co/cli/install.sh | bash"
    echo ""
}

cmd_download() {
    check_git_xet

    if [ -d "${LOCAL_FOLDER}/.git" ]; then
        echo "### Error: '${LOCAL_FOLDER}' already exists. Delete it first if you want a fresh clone."
        exit 1
    fi

    echo ""
    echo "### Cloning ${HF_REPO_ID} into ${LOCAL_FOLDER}..."
    echo "# This may take a while for large datasets."
    echo ""

    local token
    token=$(get_token)
    git clone "https://user:${token}@huggingface.co/datasets/${HF_REPO_ID}" "${LOCAL_FOLDER}"

    echo ""
    if [ "$LOCAL_FOLDER" == "../../assets/models" ]; then
        echo "# Done. Files are in assets/models"
    else
        echo "# Done. Files are in ${LOCAL_FOLDER}"
    fi
    echo ""
}

cmd_upload() {
    check_git_xet
    check_git_repo
    cd "${LOCAL_FOLDER}"

    echo ""
    echo "### Staging all changes..."
    echo ""
    git add -A

    # Check for staged changes
    local has_staged=0
    git diff --cached --quiet || has_staged=1

    # Check for committed but unpushed changes
    local has_unpushed=0
    git log @{u}..HEAD --oneline 2>/dev/null | grep -q . && has_unpushed=1 || true

    if [ "$has_staged" -eq 0 ] && [ "$has_unpushed" -eq 0 ]; then
        echo "# Nothing to push — remote is already up to date."
        echo ""
        exit 0
    fi

    if [ "$has_staged" -eq 1 ]; then
        echo ""
        echo "# Changes to be pushed:"
        GIT_PAGER=cat git diff --cached --stat
        echo ""

        read -rp "Proceed with upload? [y/N] " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "# Upload cancelled."
            echo ""
            exit 0
        fi

        echo ""
        echo "# Committing..."
        git commit -m "Update dataset via assets.sh"
    else
        echo "# Found unpushed commits — pushing now..."
    fi

    echo ""
    echo "# Pushing to HuggingFace... (this may take a while)"
    local token
    token=$(get_token)
    git push "https://user:${token}@huggingface.co/datasets/${HF_REPO_ID}"

    echo ""
    echo "# Done."
    echo ""
}

cmd_status() {
    check_git_repo
    cd "${LOCAL_FOLDER}"

    echo ""
    echo "# Fetching remote state..."
    git fetch

    echo ""
    echo "# Local changes not yet uploaded:"
    GIT_PAGER=cat git status

    echo ""
    echo "# Commits ahead of remote:"
    GIT_PAGER=cat git log origin/main..HEAD --oneline 2>/dev/null \
        || GIT_PAGER=cat git log origin/master..HEAD --oneline 2>/dev/null \
        || echo "  (none)"
    echo ""
}

case "$1" in
    -auth)     cmd_auth ;;
    -download) cmd_download ;;
    -upload)   cmd_upload ;;
    -status)   cmd_status ;;
    -help|"")  usage ;;
    *)
        echo "### Unknown command: $1"
        usage
        exit 1
        ;;
esac