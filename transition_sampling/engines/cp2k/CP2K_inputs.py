import numpy as np
from cp2k_input_tools.generator import CP2KInputGenerator
from cp2k_input_tools.parser import CP2KInputParser


class CP2KInputsHandler:
    """Handles manipulating the raw CP2K Inputs data structure.

    This class parses a CP2K inputs file into a memory, then supplies methods
    for altering it. The original file is not modified. Once the in-memory
    inputs have been changed, the current state can be written out into an
    equivalent CP2K inputs file.

    Parameters
    ----------
    cp2k_inputs_file
        The cp2k inputs file that serves as a template for this class

    Attributes
    ----------
    cp2k_dict : dict
        The in-memory data structure representing the current inputs
    """

    def __init__(self, cp2k_inputs_file: str):
        with open(cp2k_inputs_file) as f:
            parser = CP2KInputParser()
            self.cp2k_dict = parser.parse(f)

        self._atoms = None
        self._init_free_energy_section()

    @property
    def atoms(self) -> list[str]:
        """Get the atoms in this input

        Gets the name of the atoms in the inputs as represented by CP2K.

        Returns
        -------
        An ordered sequence of atoms in the inputs.
        """
        if self._atoms is None:
            # TODO: How does this handle coordinates linked in a separate file?
            # Return the first two places for each coordinate entry
            self._atoms = [entry[0:2] for entry in self._get_coord()]

        return self._atoms

    def set_positions(self, positions: np.ndarray) -> None:
        """Set the positions of atoms in the inputs.

        Positions are ordered for n atoms, in shape (n, 3). Rows represent atoms
        and columns represent (x, y, z) dimensions.

        Parameters
        ----------
        positions : np.ndarray with shape (n, 3)
            The positions for atoms to be set to.
        """
        # coords stored as list of "El x y z" strings, same as CP2K .inp file
        coords = self._get_coord()

        for i in range(positions.shape[0]):
            # Create the space separated string and append it to the atom
            pos_str = ' '.join([str(p) for p in positions[i, :]])
            coords[i] = f"{self.atoms[i]} {pos_str}"

    def set_velocities(self, velocities: np.ndarray) -> None:
        """Set the velocities of atoms in the inputs.

        Velocities are ordered for n atoms, in shape (n, 3). Rows represent
        atoms and columns represent (x, y, z) dimensions.

        Parameters
        ----------
        velocities : np.ndarray with shape (n, 3)
            The positions for atoms to be set to.
        """
        vel = self._get_velocity()

        # Assign all the velocities
        for i in range(velocities.shape[0]):
            for j in range(3):
                vel[i][j] = velocities[i, j]

    def flip_velocity(self) -> None:
        """Modify state by multiplying every velocity component by -1
        """
        vel = self._get_velocity()
        for i in range(len(vel)):
            for j in range(3):
                vel[i][j] = -1 * vel[i][j]

    def set_project_name(self, projname: str) -> None:
        """Set the CP2K project name of the inputs

        Parameters
        ----------
        projname
            The project name to be set
        """
        self.cp2k_dict["+global"]["project_name"] = projname

    def set_plumed_file(self, plumed_file_path: str) -> None:
        """Set the plumed input file to the passed path

        Parameters
        ----------
        plumed_file_path
            The full path of the plumed input file
        """
        metadyn = self._get_metadyn()
        metadyn["plumed_input_file"] = plumed_file_path

    def write_cp2k_inputs(self, filename: str) -> None:
        """Write the current state of the inputs to the passed file name.

        Creates the standard cp2k input format. Overwrites anything present.

        Parameters
        ----------
        filename
            The file to write the input to
        """
        with open(filename, 'w') as f:
            cp2k_gen = CP2KInputGenerator()
            for line in cp2k_gen.line_iter(self.cp2k_dict):
                f.write(f"{line}\n")

    def _get_subsys(self) -> dict:
        """Gets the subsys section of the stored cp2k inputs

        This is a direct reference that can be used to modify the state.

        Returns
        -------
        subsys dictionary
        """
        return self.cp2k_dict["+force_eval"][0]["+subsys"]

    def _get_coord(self) -> list[str]:
        """Gets the coord section of the stored cp2k inputs.

        Coordinates are represented as a list of strings, where each string
        follows the .xyz format of "El x y z".

        This is a direct reference that can be used to modify the state.

        Returns
        -------
            Coord list of strings
        """
        return self._get_subsys()["+coord"]["*"]

    def _get_velocity(self) -> list[list[float, float, float]]:
        """Gets the velocity section of the stored cp2k inputs.

        Velocities are represented as a list of lists, where the outer index
        is the atom and the inner index is the floats of x, y, z. If this
        section hasn't been initialized in subsys, it is created with the
        correct length and zeros for all entries.

        This is a direct reference that can be used to modify the state.

        Returns
        -------
        Velocities as a list of lists
        """
        subsys = self._get_subsys()
        if "+velocity" not in subsys:
            subsys["+velocity"] = {
                "*": [[0, 0, 0] for _ in range(len(self.atoms))]}

        return subsys["+velocity"]["*"]

    def _get_metadyn(self) -> dict:
        """Gets the metadyn section of the stored cp2k inputs

        This is a direct reference that can be used to modify the state. Note
        that `_init_free_energy_section()` should be called before this to
        ensure the needed dictionaries are set up.

        Returns
        -------
        metadyn dictionary
        """
        # Try to read this section. If it wasn't setup, catch the key error, set
        # it up, and return again.
        try:
            return self.cp2k_dict["+motion"]["+free_energy"]["+metadyn"]
        except KeyError as e:
            self._init_free_energy_section()
            return self.cp2k_dict["+motion"]["+free_energy"]["+metadyn"]

    def _init_free_energy_section(self) -> None:
        """Set up the free energy section of the cp2k input to use plumed.

        The plumed file path needs to be under motion/freeenergy/metadyn, so if
        this section was not present in the input it needs to be created and set
        to use plumed. Performs a merge, the only setting that will be
        explicitly overwritten is a "used_plumed"
        """
        motion_sect = self.cp2k_dict["+motion"]

        if "+free_energy" in motion_sect:
            # If metadynamics section isn't present, create it
            if "+metadyn" not in motion_sect["+free_energy"]:
                motion_sect["+free_energy"]["+metadyn"] = {}

            # Ensure that use plumed is set to true no matter what.
            motion_sect["+free_energy"]["+metadyn"]["use_plumed"] = True

        else:
            motion_sect["+free_energy"] = {"+metadyn": {"use_plumed": True}
                                           }
