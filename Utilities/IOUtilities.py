from dolfin  import XDMFFile
from pathlib import Path

# Class to handle the writing of the solution on XDMF file
class OutputManager:
    def __init__(self, mesh, base_path, file_name="concentration"):
        self.mesh = mesh

        # Build the path and check the existence of it
        self.output_path = Path(base_path) / mesh.name()
        self.output_path.mkdir(parents=True, exist_ok=True)

        # File name
        self.file_name = file_name
        self.file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self):
        full_path = self.output_path / f"{self.file_name}.xdmf"
        self.file = XDMFFile(self.mesh.mpi_comm(), str(full_path))
        
        # Parameters
        self.file.parameters["flush_output"]          = True
        self.file.parameters["rewrite_function_mesh"] = False
        self.file.parameters["functions_share_mesh"]  = True

    def save(self, function, t, label="concentration"):
        if not self.file:
            raise RuntimeError("Output file is not open. Call open() first.")
        
        # Ensuring the function has the correct name for the XDMF file
        function.rename(label, label)
        self.file.write(function, t)

    def close(self):
        if self.file:
            self.file.close()


class NullOutputManager:
    """Drop-in no-op replacement for OutputManager. Used when output is not needed."""
    def __enter__(self):  return self
    def __exit__(self, *args): pass
    def open(self):  pass
    def save(self, function, t, label="concentration"): pass
    def close(self): pass


def make_output_manager(mesh, output_dir, file_name="concentration"):
    """
    Factory for output managers.

    Returns a NullOutputManager if output_dir is None, otherwise a real OutputManager.
    """
    if output_dir is None:
        return NullOutputManager()
    return OutputManager(mesh, output_dir, file_name)