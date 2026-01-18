"""Launch the inference server."""

import os
import sys

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def run_server(server_args):
    """Run the server."""
    # DeepSeek-only build: gRPC mode removed
    if server_args.encoder_only:
        # DeepSeek-only build: encoder_only mode removed (multimodal not supported)
        raise NotImplementedError(
            "encoder_only mode is not supported in this DeepSeek-only build. "
            "This mode was used for multimodal encoder disaggregation."
        )
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
