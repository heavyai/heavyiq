# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/common_fn.sh"

process_args "$@"
prepare_dist
cp -r ./heavyiq ./dist/heavyiq
write_version_to_dist
verify_no_packaged_dependencies
tar -czf dist.tgz dist
