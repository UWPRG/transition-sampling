import argparse
import os

from .colvar import PlumedDriver
from .engines import CP2KEngine, AbstractEngine
from .algo import AimlessShootingDriver, MultiBasinAcceptor
from .likelihood import Maximizer

import yaml
import sys

from schema import Schema, And, Optional, Use, Or

master_schema = Schema({Optional("md_inputs"): dict,
                        Optional("colvar_inputs"): dict,
                        Optional("likelihood_inputs"): dict})

name_schema = Schema({"md_inputs": {"aimless_inputs": {"output_name": str}}}, ignore_extra_keys=True)

# GLOBALS to be overwritten as they are parsed
csv_file = None
xyz_file = None
colvar_file = None


def run_aimless(md_inputs: dict) -> None:
    """
    Validate and run the aimless shooting section

    Parameters
    ----------
    md_inputs
        The "md_inputs" dictionary of the input file
    """
    md_schema = Schema({"engine_inputs": dict, "aimless_inputs": dict})
    md_schema.validate(md_inputs)
    engine = parse_engine(md_inputs["engine_inputs"])
    algo = parse_aimless(md_inputs["aimless_inputs"], engine)

    print("Starting aimless shooting")
    algo.run(**md_inputs["aimless_inputs"])


def parse_engine(engine_inputs: dict) -> AbstractEngine:
    """
    Validate engine_inputs section of input file and create the engine.
    Parameters
    ----------
    engine_inputs
        The "engine_inputs" dictionary of the input file

    Returns
    -------
    The AbstractEngine representation of this input section
    """
    # other engine parsing should be handled by the engine
    if engine_inputs["engine"] == "cp2k":
        engine = CP2KEngine(engine_inputs, engine_inputs["engine_dir"])
    else:
        sys.exit(f"Unsupported engine type: {engine_inputs['engine']}")

    return engine


def parse_aimless(aimless_inputs: dict, engine: AbstractEngine) -> AimlessShootingDriver:
    """
    Validate aimless_inputs section of input file and create the Driver.

    Defines global variables for xyz and csv to be used in later sections if
    not explicitly set.

    Parameters
    ----------
    aimless_inputs
        The "aimless_inputs" dictionary of the input file
    engine
        The corresponding AbstractEngine created from the "engine_inputs" section

    Returns
    -------
    The AimlessShootingDriver that is ready to be run.
    """

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

    # Handle creating acceptors
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

            elif aimless_inputs["acceptor"]["type"] != "default":
                sys.exit(f"Unknown acceptor type: {aimless_inputs['acceptor']['type']}")

    # globals being overwritten
    global csv_file, xyz_file
    csv_file = f"{aimless_inputs['output_name']}.csv"
    xyz_file = f"{aimless_inputs['output_name']}.xyz"

    return AimlessShootingDriver(engine, aimless_inputs["starts_dir"],
                                 aimless_inputs["output_name"], acceptor)


def run_colvar(colvar_inputs: dict) -> None:
    """
    Validate and run the colvar section

    Parameters
    ----------
    colvar_inputs
        The "colvar_inputs" dictionary of the input file
    """
    parse_colvar(colvar_inputs)
    plumed_driver = PlumedDriver(colvar_inputs["plumed_cmd"])
    plumed_driver.run(colvar_inputs["plumed_file"], colvar_inputs["xyz_input"],
                      colvar_inputs["csv_input"], colvar_inputs["output_name"])


def parse_colvar(colvar_inputs: dict) -> None:
    """
    Validate colvar_inputs section of input file.

    If xyz and csv files are not explicitly in the inputs, checks the globals
    set above, then errors if those are not set.

    Parameters
    ----------
    colvar_inputs
        The "engine_inputs" dictionary of the input file
    """

    def check_is_file(path: str):
        open(path).close()
        return path

    colvar_schema = Schema({"plumed_cmd": str,
                            "plumed_file": And(str, Use(check_is_file)),
                            "output_name": str,
                            Optional("csv_input"): Or(None, And(str, Use(check_is_file))),
                            Optional("xyz_input"): Or(None, And(str, Use(check_is_file)))})

    colvar_schema.validate(colvar_inputs)

    if "csv_input" not in colvar_inputs or colvar_inputs["csv_input"] is None:
        if csv_file is None:
            sys.exit("If not providing csv_input for colvar_inputs, output_name"
                     " must be be given in aimless_inputs")
        colvar_inputs["csv_input"] = csv_file

    if "xyz_input" not in colvar_inputs or colvar_inputs["xyz_input"] is None:
        if xyz_file is None:
            sys.exit("If not providing xyz_input for colvar_inputs, output_name"
                     " must be be given in aimless_inputs")
        colvar_inputs["xyz_input"] = xyz_file

    cur_file = "csv_input"
    try:
        check_is_file(colvar_inputs[cur_file])
        cur_file = "xyz_input"
        check_is_file(colvar_inputs[cur_file])
    except (IOError, FileNotFoundError):
        sys.exit(f"{cur_file} file {colvar_inputs[cur_file]} cannot be opened")

    # Setting globals
    global colvar_file
    colvar_file = colvar_inputs["output_name"]


def run_likelihood(likelihood_inputs: dict) -> None:
    """
    Validate and run the likelihood section

    Parameters
    ----------
    likelihood_inputs
        The "likelihood_inputs" dictionary of the input file
    """
    parse_likelihood(likelihood_inputs)
    maximizer = Maximizer(likelihood_inputs["colvar_input"], likelihood_inputs["csv_input"],
                          likelihood_inputs["n_iter"], likelihood_inputs["use_jac"])
    solution = maximizer.maximize(likelihood_inputs["max_cvs"])
    solution.to_csv(likelihood_inputs["output_name"])


def parse_likelihood(likelihood_inputs: dict) -> None:
    """
    Validate likelihood_inputs section of input file.

    If colvar and csv files are not explicitly in the inputs, checks the globals
    set above, then errors if those are not set.

    Parameters
    ----------
    likelihood_inputs
        The "likelihood_inputs" dictionary of the input file
    """

    def check_is_file(path: str):
        open(path).close()
        return path

    likelihood_schema = Schema({Optional("max_cvs"): Or(None,
                                                        And(int, lambda x: x >= 1,
                                                            error="max_cvs must be null or >= 1")),
                                "output_name": str,
                                Optional("csv_input"): Or(None, And(str, Use(check_is_file))),
                                Optional("colvar_input"): Or(None, And(str, Use(check_is_file))),
                                Optional("n_iter"): And(int, lambda x: x >= 1,
                                                        error="n_iter must be >= 1"),
                                Optional("use_jac"): bool})

    likelihood_schema.validate(likelihood_inputs)

    if "csv_input" not in likelihood_inputs or likelihood_inputs["csv_input"] is None:
        if csv_file is None:
            sys.exit("If not providing csv_input for colvar_inputs, output_name"
                     " must be be given in aimless_inputs")
        likelihood_inputs["csv_input"] = csv_file

    if "colvar_input" not in likelihood_inputs or likelihood_inputs["colvar_input"] is None:
        if colvar_file is None:
            sys.exit("If not providing colvar_input for likelihood_inputs, output_name"
                     " must be be given in colvar_inputs")
        likelihood_inputs["colvar_input"] = colvar_file

    cur_file = "csv_input"
    try:
        check_is_file(likelihood_inputs[cur_file])
        cur_file = "colvar_input"
        check_is_file(likelihood_inputs[cur_file])
    except (IOError, FileNotFoundError):
        sys.exit(f"{cur_file} file {likelihood_inputs[cur_file]} cannot be opened")


def execute(inputs: dict):
    """
    Run from the yml input parsed as a dictionary

    Parameters
    ----------
    inputs
        yml file in dictionary format
    """
    assert master_schema.validate(inputs)
    if "md_inputs" in inputs:
        run_aimless(inputs["md_inputs"])

    if "colvar_inputs" in inputs:
        run_colvar(inputs["colvar_inputs"])

    if "likelihood_inputs" in inputs:
        run_likelihood(inputs["likelihood_inputs"])


def read_and_run(input_yml: str):
    """
    Parse the yml input and run

    Parameters
    ----------
    input_yml
        Path to the yml file to parse
    """
    try:
        with open(input_yml, 'r') as file:
            inputs = yaml.safe_load(file)
    except (IOError, FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error parsing YAML file: {input_yml}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="YAML file with all required inputs")
    args = parser.parse_args()
    
    read_and_run(args.input)


if __name__ == "__main__":
    main()
