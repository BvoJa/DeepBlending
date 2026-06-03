"""Gradio entrypoint for the final NST + gradient-loss workflow.

This file intentionally delegates UI interactions and model inference to
final_run.py so running either script uses the same implementation.
"""

from final_run import (
    create_demo,
    gradient_style_transfer,
    main,
    placement_preview,
    run_gradio,
)


if __name__ == "__main__":
    main()
