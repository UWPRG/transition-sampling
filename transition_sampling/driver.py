import argparse
import os

from colvar import PlumedDriver
from engines import CP2KEngine, AbstractEngine
from algo import AimlessShootingDriver, MultiBasinAcceptor

import yaml
import sys

from schema import Schema, And, Optional, Use, Or

master_schema = Schema({Optional("md_inputs"): dict,
                        Optional("colvar_inputs"): dict,
                        Optional("likelihood_inputs"): dict})

name_schema = Schema({"md_inputs": {"aimless_inputs": {"output_name": str}}}, ignore_extra_keys=True)


def run_aimless(md_inputs: dict):
    md_schema = Schema({"engine_inputs": dict, "aimless_inputs": dict})
    md_schema.validate(md_inputs)
    engine = parse_engine(md_inputs["engine_inputs"])
    algo = parse_aimless(md_inputs["aimless_inputs"], engine)

    print("Starting aimless shooting")
    algo.run(4, **md_inputs["aimless_inputs"])


def parse_engine(engine_inputs: dict) -> AbstractEngine:
    # other engine parsing should be handled by the engine
    if engine_inputs["engine"] == "cp2k":
        engine = CP2KEngine(engine_inputs, engine_inputs["engine_dir"])
    else:
        sys.exit(f"Unsupported engine type: {engine_inputs['engine']}")

    return engine


def parse_aimless(aimless_inputs: dict, engine: AbstractEngine) -> AimlessShootingDriver:
    def check_is_dir(path: str):
        if not os.path.isdir(path):
            raise Exception(f"{path} is not a directory")
        return path

    aimless_schema = Schema({"starts_dir": And(str, Use(check_is_dir)),
                             "output_name": str,
                             "n_parallel": And(int, lambda x: x >= 1, error="n_parallel must be >= 1"),
                             "n_points": And(int, lambda x: x >= 1, error="n_points must be >= 1"),
                             "n_state_tries": And(int, lambda x: x >= 1, error="n_state_tries must be >= 1"),
                             "n_vel_tries": And(int, lambda x: x >= 1, error="n_vel_tries must be >= 1"),
                             Optional("acceptor"): Or(None, dict)})

    aimless_schema.validate(aimless_inputs)

    acceptor = None
    if "acceptor" in aimless_inputs:
        if aimless_inputs["acceptor"] is not None:
            Schema({"type": str}, ignore_extra_keys=True).validate(aimless_inputs["acceptor"])
            if aimless_inputs["acceptor"]["type"] == "multibasin":
                multi_schema = Schema({"type": str,
                                       "reactants": And(list, lambda x: all(isinstance(n, int) for n in x),
                                                        error="reactants should be all ints for multibasin"),
                                       "products": And(list, lambda x: all(isinstance(n, int) for n in x),
                                                       error="products should be all ints for multibasin")})

                multi_schema.validate(aimless_inputs["acceptor"])
                acceptor = MultiBasinAcceptor(set(aimless_inputs["acceptor"]["reactants"]),
                                              set(aimless_inputs["acceptor"]["products"]))
            else:
                sys.exit(f"Unknown acceptor type: {aimless_inputs['acceptor']['type']}")

    if engine is not None:
        return AimlessShootingDriver(engine, aimless_inputs["starts_dir"],
                                     aimless_inputs["output_name"], acceptor)


def run_colvar(inputs: dict):
    parse_colvar(inputs)
    colvar_inputs = inputs["colvar_inputs"]
    plumed_driver = PlumedDriver(colvar_inputs["plumed_cmd"])
    plumed_driver.run(colvar_inputs["plumed_file"], colvar_inputs["xyz_input"],
                      colvar_inputs["csv_file"], colvar_inputs["output_name"])


def parse_colvar(inputs: dict):
    def check_is_file(path: str):
        open(path).close()

    colvar_schema = Schema({"plumed_cmd": str,
                            "plumed_file": And(str, check_is_file),
                            "output_name": str,
                            Optional("csv_input"): Or(None, str),
                            Optional("xyz_input"): Or(None, str)})

    colvar_inputs = inputs["colvar_inputs"]
    colvar_schema.validate(colvar_inputs)

    if "csv_input" not in colvar_inputs or colvar_inputs["csv_input"] is None:
        if not name_schema.is_valid(inputs):
            sys.exit("If not providing csv_input for colvar_inputs, output_name"
                     " must be be given in aimless_inputs")
        colvar_inputs["cvs_input"] = inputs["md_inputs"]["aimless_inputs"]["output_name"] + ".csv"

    if "xyz_input" not in colvar_inputs or colvar_inputs["xyz_input"] is None:
        if not name_schema.is_valid(inputs):
            sys.exit("If not providing xyz_input for colvar_inputs, output_name"
                     " must be be given in aimless_inputs")
        colvar_inputs["xyz_input"] = inputs["md_inputs"]["aimless_inputs"]["output_name"] + ".xyz"

    file = "csv_input"
    try:
        check_is_file(colvar_inputs[file])
        file = "xyz_input"
        check_is_file(colvar_inputs[file])
    except:
        sys.exit(f"{file} cannot be opened")


def execute(inputs: dict):
    assert master_schema.validate(inputs)
    if "md_inputs" in inputs:
        run_aimless(inputs["md_inputs"])

    if "colvar_inputs" in inputs:
        run_colvar(inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="YAML file with all required inputs")
    args = parser.parse_args()

    try:
        with open(args.input, 'r') as file:
            inputs = yaml.safe_load(file)
    except (IOError, FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error parsing YAML file: {args.input}")
        print(os.curdir)
        if type(e) is IOError or type(e) is FileNotFoundError:
            print(f"File could not be read: {e}")
        elif hasattr(e, "problem_mark"):
            print(f"  parser says\n{e.problem_mark}\n  {e.problem}")
            if e.context is not None:
                print(f" {e.context}")
            print("Please correct and retry.")
        else:
            print(f"Something went wrong while parsing yaml file: {e}")

        sys.exit("File could not be parsed")

    execute(inputs)
