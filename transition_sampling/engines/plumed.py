"""
Collection of classes and methods for handling the committor sections of plumed
used during the shooting point generation.
"""

import os
import re


class PlumedInputHandler:
    """
    Handles copying and modifying the base plumed file the user has given us.
    """

    def __init__(self, plumed_file: str):
        """
        Sets the source plumed file that will be used throughout.

        Parameters
        ----------
        plumed_file
            path to the plumed file. This file will not be modified.

        Raises
        ------
        ValueError
            If `plumed_file` is not a file.
        """
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
        pattern = re.compile(r"(COMMITTOR (\.\.\.\s*\n)?)")

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
