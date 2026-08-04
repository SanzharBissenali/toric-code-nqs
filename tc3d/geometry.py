"""
Module for handling the lattice geometry for 3D Toric Code, constructing stabilizers, 
and building the connectivity for the toric code model.
"""

import numpy as np
import netket as nk
from typing import List, Tuple, Dict, Any, Optional, Union

class ThreeD_ToricCodeGeometry:
    """
    Class to handle the geometry and topology of the toric code model.
    
    Attributes:
        Lx (int): Number of vertices in the x direction
        Ly (int): Number of vertices in the y direction
        Lz (int): Number of vertices in the z direction
        bc (str): Boundary conditions, either 'OBC' or 'PBC'
        N (int): Total number of qubits in the system
        arr_coord (np.ndarray): Array of qubit coordinates
        dg_v (nk.graph.Graph): Dual lattice for vertex stabilizers
        dg_p (nk.graph.Graph): Dual lattice for plaquette stabilizers
        vertex_all (List): List of all vertex stabilizers
        plaq_all (List): List of all plaquette stabilizers
        bonds (List): List of nearest-neighbor bonds
    """
    
    def __init__(self, Lx: int, Ly: int, Lz: int, bc: str = 'OBC'):
        """
        Initialize the geometry for a toric code model.
        
        Args:
            Lx: Number of vertices in the x direction
            Ly: Number of vertices in the y direction
            Lz: Number of vertices in the z direction
            bc: Boundary conditions, 'OBC' for open or 'PBC' for periodic
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.bc = bc
        
        # Calculate the number of qubits in the system
        if bc == "OBC":
            self.N = 3 * Lx * Ly *Lz - (Lx * Ly + Lx * Lz + Ly * Lz)  # OBC
        else:
            self.N = 3 * Lx * Ly *Lz  # PBC
            
        # Create the lattice
        self._setup_lattice()
        
        # Generate stabilizers
        self.vertex_all = self._generate_stabilizer_vert()
        # plaq_all is parallel to plaq_centers (geometric centre) and plaq_orient
        # (normal axis c); _build_plaq_index() turns the centres into a
        # coordinate->index map the geometry-exact CNN gathers against.
        self.plaq_all, self.plaq_centers, self.plaq_orient = \
            self._generate_stabilizer_plaqs()
        self._build_plaq_index()
        
        # Generate nearest-neighbor bonds
        self.bonds = self._generate_bonds()
        self.Nbonds = len(self.bonds)
        
        # Extract non-boundary vertex stabilizers
        self.vertex_bulk_hetero, self.vertex_edge_hetero = self._separate_vertex_stabilizers()
        
        
    def _setup_lattice(self):
        """Setup the lattice and calculate atomic coordinates."""
        # Lattice basis
        basis = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        # Atom basis for toric code
        basis_atoms = [[1/2, 0, 0], [0, 1/2, 0], [0, 0, 1/2]]
        
        # Generate lattice coordinates
        lattice_coord = [self._map1Dto3D(j)[0] * basis[0] + 
                         self._map1Dto3D(j)[1] * basis[1] +
                         self._map1Dto3D(j)[2] * basis[2]
                         for j in range(0, self.Lx * self.Ly * self.Lz)]
        
        # Generate qubit coordinates (they're at half-offsets between edges)
        atom_coord = []
        for j in range(0, len(basis_atoms)):
            atom_coord += (lattice_coord + np.array(basis_atoms[j])).tolist()
        atom_coord = np.array(atom_coord)
        
        # Sort atomic coordinates
        atom_coord = atom_coord[np.lexsort((atom_coord[:, 0], atom_coord[:, 1], atom_coord[:, 2]))]
        
        # Handle open boundary conditions
        if self.bc == "OBC":
            self.arr_coord = self._select_obc_subset(atom_coord)
            self._coord_to_idx = {tuple((2 * np.asarray(c)).round().astype(int)): i
                                  for i, c in enumerate(self.arr_coord)}
            self.dg_v = nk.graph.Lattice(basis_vectors=basis, pbc=False, extent=[self.Lx, self.Ly, self.Lz])
            self.dg_p = nk.graph.Lattice(basis_vectors=basis, pbc=False, 
                                         site_offsets=[[1/2, 1/2, 0],[0, 1/2, 1/2],[1/2, 0, 1/2]], 
                                         extent=[self.Lx-1, self.Ly-1, self.Lz-1])
        else:
            self.arr_coord = atom_coord
            self._coord_to_idx = {tuple((2 * np.asarray(c)).round().astype(int)): i
                                  for i, c in enumerate(self.arr_coord)}
            self.dg_v = nk.graph.Lattice(basis_vectors=basis, pbc=True, extent=[self.Lx, self.Ly, self.Lz])
            self.dg_p = nk.graph.Lattice(basis_vectors=basis, pbc=True, 
                                        site_offsets=[[1/2, 1/2, 0],[0, 1/2, 1/2],[1/2, 0, 1/2]],
                                        extent=[self.Lx, self.Ly, self.Lz])
    
    def _map1Dto3D(self, n: int) -> Tuple[int, int, int]:
        """Map a 1D index to 3D coordinates as in (x, y, z)."""
        quotient, remainder = divmod(n, (self.Ly * self.Lx))
        return remainder // self.Ly, remainder % self.Ly , quotient
    
    def _select_obc_subset(self, arr: np.ndarray) -> np.ndarray:
        """Keep only qubit coords lying inside the open 3D box
        [0, Lx-1] x [0, Ly-1] x [0, Lz-1]."""
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        in_x = (x >= 0) & (x <= self.Lx - 1)
        in_y = (y >= 0) & (y <= self.Ly - 1)
        in_z = (z >= 0) & (z <= self.Lz - 1)
        return arr[in_x & in_y & in_z]
    
    def _mapping3Dto1D(self, entry: np.ndarray) -> int:
        """Map 3D coordinates to 1D qubit index. Returns -1 if not found.

        Keys are integer 2x-coords to avoid float-equality pitfalls
        (e.g. 1.9999... vs 2.0 after a float modulo).
        """
        key = tuple((2 * np.asarray(entry)).round().astype(int))
        return self._coord_to_idx.get(key, -1)
    
    def _generate_stabilizer_vert(self) -> List[List[int]]:
        """
        Generate Vertex operators with 6 (!!!) neigbors.
            
        Returns:
            List of lists, where each inner list contains the qubit indices for a vertex stabilizer
        """
        dg = self.dg_v
        if self.bc == "OBC":
            neighbors = np.array([
                dg.positions + np.array([1/2, 0, 0]),
                dg.positions + np.array([-1/2, 0, 0]),
                dg.positions + np.array([0, 1/2, 0]),
                dg.positions + np.array([0, -1/2, 0]),
                dg.positions + np.array([0, 0, 1/2]),
                dg.positions + np.array([0, 0, -1/2]),
            ])  # right, left, further, backward, up, and down neigbors
        else:
            neighbors = np.array([
                (dg.positions + np.array([1/2, 0, 0])) % self.Lx ,
                (dg.positions + np.array([-1/2, 0, 0])) % self.Lx,
                (dg.positions + np.array([0, 1/2, 0])) % self.Ly,
                (dg.positions + np.array([0, -1/2, 0])) % self.Ly,
                (dg.positions + np.array([0, 0, 1/2])) % self.Lz,
                (dg.positions + np.array([0, 0, -1/2])) % self.Lz,
            ])
            
        return [[self._mapping3Dto1D(neighbors[i, k]) for i in range(6)]
        for k in range(neighbors.shape[1])]
    
    def _generate_stabilizer_plaqs(self):
        """Plaquette (B_p) stabilizers, parallel with their centres and normals.

        Returns (plaq_all, plaq_centers, plaq_orient):
            plaq_all[p]     -> 4 edge qubit indices of plaquette p,
            plaq_centers[p] -> its geometric centre (np.array, length 3),
            plaq_orient[p]  -> its normal axis c in {0,1,2}.

        PBC: every face is complete (modulo wrapping) -> 3·Lx·Ly·Lz plaquettes.
        OBC: a boundary face with a missing edge is **dropped** — only complete
        4-edge faces are kept, so plaq_all carries no -1 padding (which would
        otherwise corrupt the ED Z-strings and the Wilson product in the CNN).
        """
        L = (self.Lx, self.Ly, self.Lz)
        plaq_all, plaq_centers, plaq_orient = [], [], []
        # iterate over the 3 plane orientations: c is the normal axis
        for c in range(3):
            a, b = [i for i in range(3) if i != c]
            e = np.eye(3)
            # plaquette center = corner + ½ê_a + ½ê_b
            center_offset = 0.5 * e[a] + 0.5 * e[b]
            # 4 edges around it
            edge_offsets = [ +0.5*e[a], -0.5*e[a], +0.5*e[b], -0.5*e[b] ]
            for ix in range(self.Lx):
                for iy in range(self.Ly):
                    for iz in range(self.Lz):
                        corner = np.array([ix, iy, iz], dtype=float)
                        center = corner + center_offset
                        edges = []
                        for off in edge_offsets:
                            coord = center + off
                            if self.bc == "PBC":
                                coord = coord % np.array(L)
                            edges.append(self._mapping3Dto1D(coord))
                        if self.bc == "OBC" and -1 in edges:
                            continue  # incomplete boundary face -> not a stabilizer
                        plaq_all.append(edges)
                        plaq_centers.append(center)
                        plaq_orient.append(c)
        return plaq_all, plaq_centers, plaq_orient

    def _build_plaq_index(self):
        """Map a plaquette centre to its index in plaq_all (2x-integer keys).

        Centres are unique across orientations (the two half-integer coordinates
        identify the normal axis), so the centre alone keys the plaquette. Used by
        KernelManager3D to gather neighbouring plaquettes by coordinate under both
        boundary conditions.
        """
        self._plaq_to_idx = {tuple((2 * np.asarray(c)).round().astype(int)): i
                             for i, c in enumerate(self.plaq_centers)}

    def _plaq_center_to_idx(self, center: np.ndarray) -> int:
        """Plaquette index at geometric centre `center`. Returns -1 if not found.

        PBC wraps the centre into the cell before lookup (2x-integer keys, mod
        2L per axis); OBC does no wrapping so out-of-box centres return -1.
        """
        c = np.asarray(center, dtype=float)
        key = (2 * c).round().astype(int)
        if self.bc == "PBC":
            key = key % (2 * np.array([self.Lx, self.Ly, self.Lz]))
        return self._plaq_to_idx.get(tuple(key.astype(int)), -1)

    def _generate_bonds(self) -> List[List[int]]:
        bonds = []
        e = np.eye(3)
        L = np.array([self.Lx, self.Ly, self.Lz])
        for vx in range(self.Lx):
            for vy in range(self.Ly):
                for vz in range(self.Lz):
                    v = np.array([vx, vy, vz], dtype=float)
                    # 6 edges at this vertex: (axis, sign)
                    incident = []
                    for axis in range(3):
                        for sign in (+1, -1):
                            coord = v + 0.5 * sign * e[axis]
                            if self.bc == "PBC":
                                coord = coord % L
                            idx = self._mapping3Dto1D(coord)
                            incident.append((axis, idx))
                    # perpendicular pairs only, each pair once
                    for i in range(6):
                        for j in range(i + 1, 6):
                            if incident[i][0] != incident[j][0]:   # different axes
                                bonds.append([incident[i][1], incident[j][1]])
        return bonds
    
    def _separate_vertex_stabilizers(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Separate vertex stabilizers into bulk and edge operators."""
        vertex_bulk_hetero = []
        vertex_edge_hetero = []
        
        for v in self.vertex_all:
            lst = [el for el in v if el != -1]
            if len(lst) == 6:
                vertex_bulk_hetero.append(lst)
            else:
                vertex_edge_hetero.append(lst)
                
        return vertex_bulk_hetero, vertex_edge_hetero
    
    def get_vertex_all_hetero(self) -> List[List[int]]:
        """Get all vertex stabilizers with -1 entries removed."""
        return [[el for el in v if el != -1] for v in self.vertex_all]
    
    def construct_Wilson_generators(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct generators for the Z2xZ2xZ2x...=(Z2)^(n) group.
        
        Returns:
            Tuple containing:
                - generators_mat: Matrix representation of generators
                - generators_lst: List representation of generators
        """
        generators_lst = []
        generators_mat = []
        vertex_all_hetero = self.get_vertex_all_hetero()
        
        for v in range(0, len(vertex_all_hetero)):
            mat = np.ones(self.N)  # vector with the size of the qubits
            mat[np.array(vertex_all_hetero[v])] = -1  # to act with a vertex operator simply swap 1 to -1
            generators_lst.append(mat)
            generators_mat.append(np.diag(mat))
            
        return np.array(generators_mat), np.array(generators_lst)
    
    def find_generators(self, vertex_lst: List[List[int]]) -> np.ndarray:
        """
        Find generators for a subset of vertex operators.
        
        Args:
            vertex_lst: List of vertex operators
            
        Returns:
            Array of generators
        """
        generators_lst = []
        vertex_all_hetero = self.get_vertex_all_hetero()
        
        for v in range(0, len(vertex_lst)):
            mat = np.ones(self.N)
            mat[np.array(vertex_all_hetero[v])] = -1
            generators_lst.append(np.diag(mat))
            
        return np.array(generators_lst)
    
    def select_subset(self, pos: Tuple[float, float], radius: float) -> np.ndarray:
        """
        Select subset of qubits bounded by pos-radius and pos+radius points.
        
        Args:
            pos: Center position (x, y)
            radius: Radius around the center
            
        Returns:
            Subset of qubit coordinates
        """
        pos_x, pos_y = pos
        xmax = pos_x + radius
        xmin = pos_x - radius
        ymax = pos_y + radius
        ymin = pos_y - radius
        
        return self.arr_coord[np.logical_and(
            np.logical_and(self.arr_coord[:, 0] <= xmax, self.arr_coord[:, 1] <= ymax),
            np.logical_and(self.arr_coord[:, 0] >= xmin, self.arr_coord[:, 1] >= ymin)
        )]
    
    def qubit_select(self, selected_locs: np.ndarray) -> List[int]:
        """
        Map 2D coordinates to 1D qubit indices.
        
        Args:
            selected_locs: Array of 2D coordinates
            
        Returns:
            List of 1D qubit indices
        """
        return [self._mapping2Dto1D(self.arr_coord, en)[0][0] for en in selected_locs]
    
    def select_bulk(self) -> np.ndarray:
        """
        Select only bulk qubits from the set of qubits.
        
        Returns:
            Array of bulk qubit coordinates
        """
        boundary_locs = np.vstack((
            self.arr_coord[(self.arr_coord == np.max(self.arr_coord)).any(axis=1)],
            self.arr_coord[(self.arr_coord == np.min(self.arr_coord)).any(axis=1)]
        ))
        
        set1 = set(map(tuple, self.arr_coord))
        set2 = set(map(tuple, boundary_locs))
        bulk_locs = np.array(list(set1 - set2))
        
        return bulk_locs 