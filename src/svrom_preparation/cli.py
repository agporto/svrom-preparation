"""Command-line entry point for specimen preparation."""
from __future__ import annotations
import argparse

from .workflow import create_manifest, run_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fit geometric reference articulations and propose specimen-specific patches')
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create an ordered manifest from an atlas ZIP/directory and PLY mesh directory')
    init.add_argument('--atlas', required=True)
    init.add_argument('--meshes', required=True)
    init.add_argument('--out', required=True)
    run = commands.add_parser('run', help='Transfer landmarks, fit adjacent pairs, and export reviewable SVROM inputs')
    run.add_argument('manifest')
    run.add_argument('--output', required=True)
    run.add_argument('--resume', action='store_true', help='Reuse completed transfers for identical inputs/settings')
    run.add_argument('--transfer-only', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'init':
        print(create_manifest(args.atlas, args.meshes, args.out))
        print('Check vertebral order and coordinate/units declarations before running.')
        return 0
    report = run_manifest(args.manifest, args.output, resume=args.resume, transfer_only=args.transfer_only)
    # Partial results have a distinct exit status and a complete review report.
    return 0 if report['status'] in {'transfer_complete', 'complete_geometric_reference'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
