# -*- coding: utf-8 -*-
import numpy as np

# from scipy.optimize import fsolve
from rdkit import Chem

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "Br": 194,
        "I": 214,
    },
    "N": {
        "H": 101,
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
        "Cl": 175,
        "Br": 214,
        "S": 168,
        "I": 222,
        "P": 177,
    },
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Br": 172,
        "S": 151,
        "P": 163,
        "Si": 163,
        "Cl": 164,
        "I": 194,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "S": 158,
        "Si": 160,
        "Cl": 166,
        "Br": 178,
        "P": 156,
        "I": 187,
    },
    "B": {"H": 119, "Cl": 175},
    "Si": {
        "Si": 233,
        "H": 148,
        "C": 185,
        "O": 163,
        "S": 200,
        "F": 160,
        "Cl": 202,
        "Br": 215,
        "I": 243,
    },
    "Cl": {
        "Cl": 199,
        "H": 127,
        "C": 177,
        "N": 175,
        "O": 164,
        "P": 203,
        "S": 207,
        "B": 175,
        "Si": 202,
        "F": 166,
        "Br": 214,
    },
    "S": {
        "H": 134,
        "C": 182,
        "N": 168,
        "O": 151,
        "S": 204,
        "F": 158,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {
        "Br": 228,
        "H": 141,
        "C": 194,
        "O": 172,
        "N": 214,
        "Si": 215,
        "S": 225,
        "F": 178,
        "Cl": 214,
        "P": 222,
    },
    "P": {
        "P": 221,
        "H": 144,
        "C": 184,
        "O": 163,
        "Cl": 203,
        "S": 210,
        "F": 156,
        "N": 177,
        "Br": 222,
    },
    "I": {
        "H": 161,
        "C": 214,
        "Si": 243,
        "N": 222,
        "O": 194,
        "S": 234,
        "F": 187,
        "I": 266,
    },
    "As": {"H": 152},
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}


bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}


def calc_opposite_side(dis1, dis2, ang):
    return np.sqrt(dis1**2 + dis2**2 - np.cos(ang) * 2 * dis1 * dis2)


def calc_angle(dis1, dis2, dis3, ori_angle):
    return np.abs(
        np.arccos((dis1**2 + dis2**2 - dis3**2) / (2 * dis1 * dis2))
    ) * (-1 if ori_angle < 0 else 1)


def feats2DM(feats):
    def local_DM(local_feat):
        DM = np.zeros((4, 4))
        rij1, rij2, rij3, theta_ij1j2, theta_ij1j3, theta_ij2j3 = local_feat

        assert theta_ij1j3 * theta_ij2j3 > 0

        rj1j2 = calc_opposite_side(rij1, rij2, theta_ij1j2)
        rj1j3 = calc_opposite_side(rij1, rij3, theta_ij1j3)
        rj2j3 = calc_opposite_side(rij2, rij3, theta_ij2j3)
        DM[0][1] = DM[1][0] = rij1
        DM[0][2] = DM[2][0] = rj1j2
        DM[0][3] = DM[3][0] = rj1j3
        DM[1][2] = DM[2][1] = rij2
        DM[1][3] = DM[3][1] = rij3
        DM[2][3] = DM[3][2] = rj2j3
        return DM

    DM_list = []
    for local_feat in feats:
        DM_list.append(local_DM(local_feat))

    sparse_DM = np.zeros((len(feats) + 3, len(feats) + 3))
    update_count = np.zeros((len(feats) + 3, len(feats) + 3))

    for i in range(len(feats)):
        sparse_DM[i : (i + 4), i : (i + 4)] += DM_list[i]
        update_count[i : (i + 4), i : (i + 4)] += 1

    average_DM = np.divide(sparse_DM, update_count, where=update_count != 0)

    return average_DM


def DM2feats(DM, sign):
    def local_feat(local_DM, sign):
        rij1, rij2, rij3, rj1j2, rj1j3, rj2j3 = (
            local_DM[0, 1],
            local_DM[1, 2],
            local_DM[1, 3],
            local_DM[0, 2],
            local_DM[0, 3],
            local_DM[2, 3],
        )
        theta_ij1j2 = calc_angle(rij1, rij2, rj1j2, 1)
        theta_ij1j3 = calc_angle(rij1, rij3, rj1j3, sign)
        theta_ij2j3 = calc_angle(rij2, rij3, rj2j3, sign)
        feat = [rij1, rij2, rij3, theta_ij1j2, theta_ij1j3, theta_ij2j3]
        return feat

    feats = []

    for i in range(len(DM) - 3):
        feats.append(local_feat(DM[i : (i + 4), i : (i + 4)], sign[i]))

    return feats


def build_local_coords(
    local_dm, sign
):  # local_dm: [rij1, rij2, rij3, theta_ij1j2, theta_ij1j3, theta_ij2j3]
    local_coords = np.zeros((4, 3))  # i1, i2, i3, i4

    r12, r13, r14, r23, r24, r34 = (
        local_dm[0][1],
        local_dm[0][2],
        local_dm[0][3],
        local_dm[1][2],
        local_dm[1][3],
        local_dm[2][3],
    )

    # for i2
    local_coords[1][0] = r12
    # for i3
    if r23**2 > r12**2 + r13**2:  # obtuse
        local_coords[2][0] = -np.abs(r12 / 2 - (r23**2 - r13**2) / (2 * r12))
    else:
        local_coords[2][0] = np.abs(r12 / 2 - (r23**2 - r13**2) / (2 * r12))
    local_coords[2][1] = np.sqrt(np.abs(r13**2 - local_coords[2][0] ** 2))
    # for i4
    local_coords[3][0] = (r12**2 + r14**2 - r24**2) / (2 * r12)
    local_coords[3][1] = (
        local_coords[2][1] ** 2
        - (
            r34**2
            - (local_coords[2][0] - local_coords[3][0]) ** 2
            - r24**2
            + (local_coords[3][0] - r12) ** 2
        )
    ) / (2 * local_coords[2][1])
    local_coords[3][2] = np.sqrt(
        np.abs(r14**2 - local_coords[3][0] ** 2 - local_coords[3][1] ** 2)
    ) * np.sign(sign)
    # import torch
    # if torch.isnan(torch.tensor(q)).any():
    #     import ipdb; ipdb.set_trace()
    #     import time
    #     time.sleep(1000)
    return local_coords


def kabsch(P, Q):
    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    # V, S, W = np.linalg.svd(C)

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def infer(pos_global, dm, sign):
    local_pos = build_local_coords(dm, sign)
    pos_global_centroid = pos_global - pos_global[0]

    U = kabsch(local_pos[:3], pos_global_centroid)

    pos_j3_ = np.dot(local_pos[-1], U)

    pos_j3 = pos_j3_ + pos_global[0]

    return pos_j3


def build_coords(dm, signs):
    pos0, pos1, pos2, pos3 = build_local_coords(dm[0:4, 0:4], signs[0])
    coords = [pos0, pos1, pos2, pos3]
    for i in range(dm.shape[0] - 4):
        pos = infer(
            np.array(coords[-3:]), dm[i + 1 : i + 5, i + 1 : i + 5], signs[i + 1]
        )
        coords.append(pos)

    return coords


def get_distances(p1):
    p1 = np.atleast_2d(p1)
    np1 = len(p1)
    ind1, ind2 = np.triu_indices(np1, k=1)
    D = p1[ind2] - p1[ind1]

    (D_len,) = np.linalg.norm([D], axis=2)

    Dout_len = np.zeros((np1, np1))
    Dout_len[(ind1, ind2)] = D_len
    Dout_len += Dout_len.T
    return Dout_len


def get_LAS_mask(mol):
    def binarize(x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))

    # adj - > n_hops connections adj
    def n_hops_adj(adj, n_hops):
        adj_mats = [
            np.eye(adj.shape[0], dtype=adj.dtype),
            binarize(adj + np.eye(adj.shape[0], dtype=adj.dtype)),
        ]

        for i in range(2, n_hops + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        extend_mat = np.zeros_like(adj)

        for i in range(1, n_hops + 1):
            extend_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return extend_mat

    adj = Chem.GetAdjacencyMatrix(mol)
    extend_adj = n_hops_adj(adj, 2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            for i in ring:
                for j in ring:
                    if i == j:
                        continue
                    else:
                        extend_adj[i][j] += 1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask


def correct_DM(dm, mol):
    dm_mask = get_LAS_mask(mol)
    positions = np.array(mol.GetConformer().GetPositions())
    rdkit_dm = get_distances(positions)
    dm[dm_mask > 0] = rdkit_dm[dm_mask > 0]

    # ddd = rdkit_dm
    # ddd[dm_mask<=0] = 0
    # print(ddd)
    # print()

    return dm


def gradient_descent_optimization(
    dm_corrected, mol, coords=None, iterations=10, learning_rate=0.01, n_components=3
):
    np.random.seed(42)
    N = dm_corrected.shape[0]
    if coords is None:
        coords = np.random.rand(N, n_components)  # Random initialization

    for _ in range(iterations):
        grad = np.zeros_like(coords)
        for i in range(N):
            for j in range(N):
                if i != j:
                    vector_diff = coords[i] - coords[j]
                    distance = np.linalg.norm(vector_diff)
                    delta = (distance - dm_corrected[i, j]) / (distance + 1e-5)
                    grad[i] += delta * vector_diff
                    grad[j] -= delta * vector_diff
        coords -= learning_rate * grad
        # Update distance matrix and correct it
        condensed_dm = get_distances(coords)
        dm_corrected = correct_DM(condensed_dm, mol)
    return coords


def generate_coords(dm, signs, mol, method="MDS", iteration=10, lr=0.01):
    raise NotImplementedError
    # # dm = feats2DM(feats)
    # coords = np.array(build_coords(dm.detach().numpy(), signs.detach().numpy()))
    # condensed_dm = get_distances(coords)
    # # print()
    # # print(coords)
    # # print()
    # # print(condensed_dm)
    # # print()
    # # dm_corrected = correct_DM(condensed_dm, mol)

    # if method == "MDS":
    #     from sklearn.manifold import MDS

    #     for _ in range(iteration):
    #         mds = MDS(n_components=3, dissimilarity="precomputed", eps=1e-5)
    #         coords = mds.fit_transform(dm_corrected)
    #         condensed_dm = get_distances(coords)
    #         dm_corrected = correct_DM(condensed_dm, mol)
    #     # mds = MDS(n_components=3, dissimilarity='precomputed', eps=1e-5)
    #     # coords = mds.fit_transform(dm_corrected)
    # elif method == "iteration":
    #     feats_corrected = DM2feats(dm_corrected, feats)
    #     coords = build_coords(feats_corrected)
    # elif method == "GD":
    #     coords = gradient_descent_optimization(
    #         dm_corrected,
    #         mol,
    #         coords=coords,
    #         iterations=iteration,
    #         learning_rate=lr,
    #         n_components=3,
    #     )
    # elif method == "none":
    #     pass

    # return np.array(coords)
