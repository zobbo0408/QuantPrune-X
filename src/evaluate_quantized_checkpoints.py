"""Evaluate saved quantized checkpoints without exporting them."""
from evaluation.quantized_checkpoints import argument_parser, run_evaluation


def main(argv=None):
    parser = argument_parser(__doc__)
    args = parser.parse_args(argv)
    try:
        reports = run_evaluation(args.config, output=args.output, checkpoints=args.checkpoints,
                                 run_id=args.run_id, progress=not args.quiet)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError) as exc:
        parser.exit(1, f"{exc}\n")
    for report in reports:
        print(f"{report['model']} INT{report['quantization']['bit_width']}: evaluated {report['output_directory']}")


if __name__ == "__main__":
    main()
