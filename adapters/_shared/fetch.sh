# Sourced by adapters/<name>/setup.sh. Helpers to fetch pinned upstream code and files.
set -euo pipefail

# fetch_git URL COMMIT DEST: shallow checkout of exactly COMMIT into DEST.
fetch_git() {
    local url=$1 commit=$2 dest=$3
    if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD)" = "$commit" ]; then
        return 0
    fi
    rm -rf "$dest"
    git init -q "$dest"
    git -C "$dest" remote add origin "$url"
    git -C "$dest" fetch -q --depth 1 origin "$commit"
    git -C "$dest" checkout -q FETCH_HEAD
}

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# fetch_file URL DEST [SHA256]: download once; verify the checksum when given.
fetch_file() {
    local url=$1 dest=$2 sha=${3:-}
    if [ ! -s "$dest" ]; then
        mkdir -p "$(dirname "$dest")"
        curl -fL --retry 3 -o "$dest.part" "$url"
        mv "$dest.part" "$dest"
    fi
    if [ -n "$sha" ]; then
        [ "$(sha256 "$dest")" = "$sha" ] || { echo "checksum mismatch: $dest" >&2; exit 1; }
    else
        echo "sha256 $(sha256 "$dest")  $dest  (no pinned checksum)"
    fi
}
