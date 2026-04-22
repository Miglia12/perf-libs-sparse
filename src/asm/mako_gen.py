#!/usr/bin/env python3

# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
# affiliates <open-source-office@arm.com></text>
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# vim: set et sw=4 sts=4 fileencoding=utf-8:

import argparse
import os
import sys

from mako.template import Template, exceptions
from mako.lookup import TemplateLookup


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the template file to render")
    parser.add_argument("-o", "--output", help="the output file", required=True)
    parser.add_argument("--target-os", help="the target operating system to generate kernels for", required=True)

    args, extras = parser.parse_known_args(argv[1:])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bp_file = os.path.join(script_dir, "assembly_boilerplate.py.mako")
    if not os.path.isfile(bp_file):
        print(f"ERROR: expected boilerplate file not found: {bp_file}", file=sys.stderr)
        return 2

    with open(args.path, "r", encoding="utf-8") as f:
        template_lookup = TemplateLookup(directories=[os.path.dirname(args.path), script_dir])
        template = Template(f.read(), strict_undefined=True, lookup=template_lookup)

    try:
        output = template.render(target_os=args.target_os, args=extras)
    except Exception:
        print(exceptions.text_error_template().render(), file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
