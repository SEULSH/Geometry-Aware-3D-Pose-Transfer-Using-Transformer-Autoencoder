import torch.utils.data as data
import torch
import numpy as np
import pymesh

class SMPL_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point=False):
        self.train = train
        self.npoints = npoints
        self.shuffle_point = shuffle_point 
        self.path = '/media/lsh/datasets/npt-data/'

        self.datapath = []
        for i in range(16):
            identity_i = i
            for j in range(200, 600):
                identity_p = j
                data_in = [identity_i, identity_p]
                self.datapath.append(data_in)

    def __getitem__(self, index):

        np.random.seed()
        mesh_set = self.datapath[index]
        identity_mesh_i = mesh_set[0]
        identity_mesh_p = mesh_set[1]
        pose_mesh_i = np.random.randint(0, 16)
        pose_mesh_p = np.random.randint(200, 600)

        identity_mesh = pymesh.load_mesh(self.path + 'id' + str(identity_mesh_i) + '_' + str(identity_mesh_p) + '.obj')
        pose_mesh = pymesh.load_mesh(self.path + 'id' + str(pose_mesh_i) + '_' + str(pose_mesh_p) + '.obj')
        gt_mesh = pymesh.load_mesh(self.path + 'id' + str(identity_mesh_i) + '_' + str(pose_mesh_p) + '.obj')  # groundtruth

        pose_points = pose_mesh.vertices
        identity_points = identity_mesh.vertices
        identity_faces = identity_mesh.faces
        gt_points = gt_mesh.vertices

        # normalization
        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))
        
        identity_points = identity_points - (identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points = torch.from_numpy(identity_points.astype(np.float32))

        gt_points = gt_points - (gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        random_sample = np.random.choice(self.npoints, size=self.npoints, replace=False)
        random_sample2 = np.random.choice(self.npoints, size=self.npoints, replace=False)

        new_face = identity_faces
        # shuffle orders
        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points = identity_points[random_sample]
            gt_points = gt_points[random_sample]
            
            face_dict = {}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]] = i
            new_f = []
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]], face_dict[identity_faces[i][1]], face_dict[identity_faces[i][2]]])
            new_face = np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face
        

    def __len__(self):
        return len(self.datapath)

class SMAL_DATA(data.Dataset):
    def __init__(self, train, npoints=3889, shuffle_point=False):
        self.train = train
        self.npoints = npoints
        self.shuffle_point = shuffle_point
        self.path = '/media/lsh/datasets/smal-data/'
        self.datapath = []
        self.id_num = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 38, 39])
        for i in range(len(self.id_num)):
            identity_i = self.id_num[i]
            for j in range(400):
                identity_p = j
                data_in = [identity_i, identity_p]
                self.datapath.append(data_in)

    def __getitem__(self, index):

        np.random.seed()
        mesh_set = self.datapath[index]
        identity_mesh_i = mesh_set[0]
        identity_mesh_p = mesh_set[1]
        pose_mesh_i = np.random.choice(self.id_num)
        pose_mesh_p = np.random.randint(400)

        identity_mesh = pymesh.load_mesh(self.path + 'toy_' + str(identity_mesh_i) + '_' + str(identity_mesh_p) + '.ply')
        pose_mesh = pymesh.load_mesh(self.path + 'toy_' + str(pose_mesh_i) + '_' + str(pose_mesh_p) + '.ply')
        gt_mesh = pymesh.load_mesh(self.path + 'toy_' + str(identity_mesh_i) + '_' + str(pose_mesh_p) + '.ply')

        identity_points = identity_mesh.vertices
        identity_faces = identity_mesh.faces
        pose_points = pose_mesh.vertices
        pose_faces = pose_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points = identity_points - (identity_mesh.bbox[0] + identity_mesh.bbox[1]) / 2
        identity_points = torch.from_numpy(identity_points.astype(np.float32))

        gt_points = gt_points - (gt_mesh.bbox[0] + gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        random_sample = np.random.choice(self.npoints, size=self.npoints, replace=False)
        random_sample2 = np.random.choice(self.npoints, size=self.npoints, replace=False)

        new_face = identity_faces
        # shuffle orders
        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points = identity_points[random_sample]
            gt_points = gt_points[random_sample]

            face_dict = {}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]] = i
            new_f = []
            for i in range(len(identity_faces)):
                new_f.append(
                    [face_dict[identity_faces[i][0]], face_dict[identity_faces[i][1]], face_dict[identity_faces[i][2]]])
            new_face = np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face

    def __len__(self):
        return len(self.datapath)
