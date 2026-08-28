import argparse
from importlib import import_module


def _run_training() -> None:
    import_module('training.main').main()


def _run_xai() -> None:
    import_module('xai.main').main()


def _run_xai_evaluation() -> None:
    import_module('xai_evaluation.main').main()


def _run_analyses() -> None:
    import_module('analyses.main').main()


def _run_full() -> None:
    _run_training()
    _run_xai()
    _run_xai_evaluation()
    _run_analyses()


MODES = {
    'training': _run_training,
    'xai': _run_xai,
    'xai_evaluation': _run_xai_evaluation,
    'analyses': _run_analyses,
    # Backward-compatible alias for the previous help text typo.
    'analysis': _run_analyses,
    'full': _run_full,
}


def get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        dest='mode',
        required=True,
        choices=sorted(MODES.keys()),
        help='Modes: training, xai, xai_evaluation, analyses, full.',
        type=str,
    )
    return parser.parse_args()


def main(mode: str):
    MODES[mode]()


if __name__ == '__main__':
    args = get_command_line_arguments()
    main(mode=args.mode)
