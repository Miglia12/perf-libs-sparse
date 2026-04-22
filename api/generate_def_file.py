# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or its
# affiliates <open-source-office@arm.com></text>
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

import json
import os

from argparse import ArgumentParser

def generate_def_files(json_files, output_file, lib_name):
    with open(output_file, "w") as o:
        o.write(f"LIBRARY {lib_name} \n")
        o.write(f"EXPORTS\n")
        for j in json_files:
            with open(j) as f:
                funcs = json.load(f)
            for func in funcs:
                o.write(f"{func['name']}\n")


def main():
    parser = ArgumentParser(description="Generate a module-definition (.def) file exporting "
                                        "public functions from JSON representation.")
    parser.add_argument("json_file", help="JSON files containing list of public functions", nargs="+")
    parser.add_argument("--output_file", help="Output filename", default="exports.def")
    parser.add_argument("--lib_name", help="DLL library name")

    args = parser.parse_args()

    generate_def_files(args.json_file, args.output_file, args.lib_name)


if __name__ == "__main__":
    main()
