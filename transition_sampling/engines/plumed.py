"""
Collection of classes and methods for handling the committor sections of plumed
used during the shooting point generation.
"""
from __future__ import annotations

import os
import re


class PlumedInputHandler:
    """
    Handles copying and modifying a template plumed file.

    This class is used to read a template plumed file that has exactly one
    COMMITTOR section, add a given FILE argument to it, and write it to a new
    location. This can be invoked repeatedly.

    Will not modify the original file. The original file is read once at init
    and no modifications to it after this object is constructed will carry over.

    Parameters
    ----------
    plumed_file
        path to the plumed file. This file will not be modified.

    Attributes
    ----------
    before : str
        The string to be written before FILE=arg
    after : str
        The string to be written after FILE=arg

    Raises
    ------
    ValueError
        If `plumed_file` is not a file.
    """
    def __init__(self, plumed_file: str):
        if not os.path.isfile(plumed_file):
            raise ValueError("plumed file must a valid file")

        # TODO: more validation - ensure that no-stop is false

        with open(plumed_file) as f:
            # Save the entire string in memory so we can modify repeatedly
            plumed_in_str = f.read()

        self.before, self.after = self._split_file(plumed_in_str)

    def write_plumed(self, new_location: str, out_name: str) -> None:
        """Copy the plumed file and set the committor output file.

        The plumed file is object is tied to is copied to `new_location`.
        Within the COMMITTOR section, the FILE arguments is set to `out_name`

        Parameters
        ----------
        new_location
            Location to copy the plumed file to
        out_name
            Name to be set for the COMMITTOR out file
        """
        with open(new_location, "w") as f:
            f.write(self.before)
            f.write(f"FILE={out_name}")
            f.write(self.after)

    @staticmethod
    def _split_file(raw_str: str) -> tuple[str, str]:
        """Take the raw plumed string and split it to allow arg insertion.

        Given the full string of the plumed file, split it into two strings
        where the break is at a location where an argument could be inserted
        into the COMMITTOR section.
        Parameters
        ----------
        raw_str
            The string of the plumed

        Returns
        -------
        `before`, `after` - the string before the argument insertion and the
        string that follows it

        Examples
        --------
        plumed_string = "....." # raw plumed string that contains COMMITTOR
        before, after = _split_file(plumed_string)
        inserted_arg = before + "FILE=myfile" + after
        """
        # regex to match the start of the committor section. This is where
        # we will insert the FILE arg. Handle the optional ... block format
        # allowed by PLUMED. If \2 matches, we know is format was used.
        # The first section is to state that COMMITTOR must be the first
        # non-horizontal whitespace in a line, thus ignoring any lines that have
        # preceding characters (such as comments) and ignoring multi-line breaks
        # that \s matches.
        pattern = re.compile(r"^[^\S\r\n]*(COMMITTOR (\.\.\.\s*\n)?)", re.MULTILINE)

        # Match is a list of all the matching patterns. Each entry is a tuple,
        # with one entry for each group
        match = pattern.findall(raw_str)
        if len(match) == 0:
            raise ValueError("COMMITTOR section was not found")

        if len(match) > 1:
            raise ValueError("Multiple COMMITTOR sections found")

        # We are guaranteed for `before` to be formatted correctly for insertion
        # due to the regex including a trailing space or new line, but may
        split = pattern.split(raw_str)

        # Set to set the 2nd matched group to be "" to avoid repeat with 1st
        # group when joined
        split[2] = ""

        if match[0][1] == "":
            # The 2nd group did not match, so we just need to add a leading
            # space after insertion to ensure the arg can be inserted properly
            after = f" {split[-1]}"
        else:
            # The 2nd group matched so the extended ... format is used. We need
            # to add a newline after the insertion
            after = f"\n{split[-1]}"

        return ''.join(split[:-1]), after


class PlumedOutputHandler:
    """Reads the output of a plumed committor file.

    Used for parsing which basin the trajectory committed to.

    Parameters
    ----------
    plumed_out_file
        Path of the plumed file that will be output by the committor

    Raises
    ------
    ValueError
        If the passed string is not a file
    """

    def __init__(self, plumed_out_file: str):
        if not os.path.isfile(plumed_out_file):
            raise ValueError("plumed out file must a valid file")

        self.plumed_out_file = plumed_out_file

    def check_basin(self) -> int:
        """
        Read the basin of the attached plumed committor output.

        Looks at the plumed output file that this object was created with and
        returns the basin it committed to.

        Returns
        -------
        The basin the attached plumed file committed to. None if did not commit.
        """
        # Plumed output has this line followed by the basin number.

        # In all plumed versions 2.6.x and earlier this has been a typo
        # "COMMITED". However it looks like 2.7 has a patch for this,
        # so we will keep an extra optional 'T' here to match both versions.
        pattern = re.compile(r"SET COMMITT?ED TO BASIN (?P<basin>\d+)")
        with open(self.plumed_out_file) as f:
            match = pattern.search(f.read())

        if not match:
            return None
        else:
            return int(match.group("basin"))
