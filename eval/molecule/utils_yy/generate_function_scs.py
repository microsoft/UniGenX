# -*- coding: utf-8 -*-
import numpy as np

# from scipy.optimize import fsolve
from rdkit import Chem


# generated from chatgpt
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


# generated from chatgpt
def transform_back_to_original(A, B, C, r, theta, phi):
    # Step 1: Convert spherical coordinates to Cartesian coordinates in the new coordinate system
    x_double_prime, y_double_prime, z_double_prime = spherical_to_cartesian(
        r, theta, phi
    )

    # Combine into a single vector
    D_double_prime = np.array([x_double_prime, y_double_prime, z_double_prime])

    # Translate points so that A is the origin
    B_prime = B - A
    C_prime = C - A

    # Normalize B' to find the unit vector along the new x-axis
    u = B_prime / np.linalg.norm(B_prime)

    # Compute the vector from A to C and its component orthogonal to u
    AC_prime = C_prime
    AC_prime_parallel_u = np.dot(AC_prime, u) * u
    v = AC_prime - AC_prime_parallel_u
    v /= np.linalg.norm(v)  # Normalize v to find the unit vector along the new y-axis

    # Compute the unit vector along the new z-axis
    w = np.cross(u, v)

    # Construct the rotation matrix R
    R = np.column_stack((u, v, w))

    # Step 2: Rotate the coordinates back to the original orientation
    D_prime = np.dot(R, D_double_prime)

    # Step 3: Translate the coordinates back to the original position
    D_original = D_prime + A

    return D_original


def generate_coords(feats, mol, choose_c2):
    reference_node_idx = []
    coords = []

    for i, (r, theta, phi, sign) in enumerate(feats):
        if i == 0:
            reference_node_idx.append((0, 0, 0))
            coords.append(
                np.array([0, 0, 0])
            )  # the first point is always at the origin
        elif i == 1:
            reference_node_idx.append((0, 0, 0))
            coords.append(
                np.array([r, 0, 0])
            )  # the second point is always on the positive x-axis
        elif i == 2:
            reference_node_idx.append((1, 0, -1))
            rel_x = np.cos(phi) * r
            rel_y = np.sin(phi) * r
            coords.append(
                np.array([rel_x, rel_y, 0])
            )  # the third point is on the xy-plane
        else:
            for j in range(i)[::-1]:
                # if exist bond between i and j
                if mol.GetBondBetweenAtoms(i, j) is not None:
                    focus_atom_idx = j
                    if choose_c2 == "recurrent-index":
                        if focus_atom_idx == 1:
                            focus_c1_atom_idx = 0
                            focus_c2_atom_idx = 2
                        else:
                            focus_c1_atom_idx = reference_node_idx[j][0]
                            focus_c2_atom_idx = reference_node_idx[j][1]
                    elif choose_c2 == "c1-closest":
                        distance_array = np.sqrt(
                            (
                                (np.array(coords) - np.array(coords)[focus_atom_idx])
                                ** 2
                            ).sum(axis=-1)
                        )
                        sorted_indices = np.argsort(distance_array)
                        nearest_candidates = sorted_indices[sorted_indices < i][
                            1:
                        ]  # select generated and closest atom index
                        focus_c1_atom_idx = nearest_candidates[0]
                        focus_c2_atom_idx = nearest_candidates[1]

                    reference_node_idx.append(
                        (focus_atom_idx, focus_c1_atom_idx, focus_c2_atom_idx)
                    )
                    break
            assert (
                focus_c2_atom_idx >= 0
                and focus_c1_atom_idx >= 0
                and focus_atom_idx >= 0
            ), f"focus_c2_atom_idx: {focus_c2_atom_idx}, focus_c1_atom_idx: {focus_c1_atom_idx}, focus_atom_idx: {focus_atom_idx}"

            focus_atom_positions = coords[focus_atom_idx]
            focus_c1_atom_positions = coords[focus_c1_atom_idx]
            focus_c2_atom_positions = coords[focus_c2_atom_idx]

            new_coords = transform_back_to_original(
                focus_atom_positions,
                focus_c1_atom_positions,
                focus_c2_atom_positions,
                r,
                theta,
                phi * sign,
            )
            # if i == 6:
            #     print(f'original features : {dist}, {angle}, {torsion}, {sign}')
            #     new_local_coords = feat_to_local_coords(dist, angle, torsion, sign, i)
            #     distances, angles, torsions, signs = check_local_coords(new_local_coords, focus_atom_positions, focus_c1_atom_positions, focus_c2_atom_positions)
            #     print(f'coords to features: {distances}, {angles}, {torsions}, {signs}')
            #     print(new_local_coords)
            #     new_coords = local_to_global(new_local_coords, focus_atom_positions, focus_c1_atom_positions, focus_c2_atom_positions)
            #     print(new_coords)
            # else:
            #     new_local_coords = feat_to_local_coords(dist, angle, torsion, sign, i)
            #     new_coords = local_to_global(new_local_coords, focus_atom_positions, focus_c1_atom_positions, focus_c2_atom_positions)

            # exit()
            coords.append(new_coords)

    return np.array(coords)
