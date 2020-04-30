import open3d as o3d
from utils import *
import numpy as np
import time
import glob
import datetime
import os
import logging
import argparse
import traceback


def farthest_point_sample(point, npoint):
    """
    Args:
        point: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class RPCDPrepreocess():
    def __init__(self, folder_path, save_path):
        self.folder_path = folder_path
        self.save_path = save_path
        self.noiseless_point_cloud = None
        self.sample_point_clouds_from_mesh = []
        self.sample_point_clouds_tmp = []
        self.sample_point_clouds_from_point_cloud = []
        self.reconstructed_mesh = None
        self.segmentation_point_cloud = []
        self.annotation_name = []
        self.annotation_dict = {}
        self.translate_matrix = None
        logging.basicConfig(filename='./log.txt', datefmt='%Y-%m-%d %H:%M:%S %p', level=logging.DEBUG,
                            format='%(asctime)s-%(message)s')

    def load_point_cloud(self):
        all_ply = glob.glob(os.path.join(self.folder_path, f'**{os.path.sep}*.ply'), recursive=True)
        clean_txt_files = glob.glob(os.path.join(self.folder_path, '*_clean.txt'))
        clean_txt_files.extend(glob.glob(os.path.join(self.folder_path, '*_th.txt')))
        txt_files = glob.glob(os.path.join(self.folder_path, '*.txt'))
        for file_path in txt_files:
            if file_path not in clean_txt_files:
                # text file for part seg
                np_points = np.loadtxt(file_path)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np_points[:, 0:3])
                self.segmentation_point_cloud.append(pcd)
                filename = os.path.basename(file_path)
                part_name = '_'.join(filename.split('_')[2:]).split('.')[0]
                self.annotation_name.append(part_name)

        if len(clean_txt_files) > 0:
            file_path = clean_txt_files[0]
            if len(clean_txt_files) > 1:
                print(f"\tWARNING: multiple clean txt in {self.folder_path}")
            np_points = np.loadtxt(file_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_points[:, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(np_points[:, 3:6] / 255)
            pcd.normals = o3d.utility.Vector3dVector(np_points[:, 6:9])
            self.raw_point_cloud = pcd
        else:
            if len(all_ply) >= 1:
                ply = all_ply[0]
                if len(all_ply) > 1:
                    print(f"\tWARNING: multiple ply in {self.folder_path}")
                self.raw_point_cloud = o3d.io.read_point_cloud(ply)
            else:
                print(f"\tWARNING: no ply in {self.folder_path}")
                return
        assert self.raw_point_cloud.points.__len__() > 0
        self.translate_matrix = 0 - self.raw_point_cloud.get_center()

    def print_info(self):
        try:
            yellow_print('File:')
            blue_print(self.folder_path)
        except Exception as e:
            red_print('No input file')
            print(traceback.format_exc())
            print(e)
        try:
            yellow_print('Raw point cloud:')
            blue_print(self.raw_point_cloud)
        except Exception as e:
            red_print('No raw point cloud')
            print(traceback.format_exc())
            print(e)
        try:
            yellow_print('Noiseless point cloud:')
            blue_print(self.noiseless_point_cloud)
        except Exception as e:
            red_print('No raw point cloud')
            print(traceback.format_exc())
            print(e)
        try:
            yellow_print('Reconstructed mesh info:')
            blue_print(self.reconstructed_mesh)
        except Exception as e:
            red_print('No reconstructed mesh')
            print(traceback.format_exc())
            print(e)
        yellow_print('Down sample times:')
        blue_print(f'{len(self.sample_point_clouds_from_point_cloud)}')
        for idx, (pc) in enumerate(self.sample_point_clouds_from_point_cloud):
            yellow_print(f'Down sampled pc from pc {idx}:')
            blue_print(pc)

    def wirte_info(self, path):
        f = open(os.path.join(path, 'info.txt'), 'w', encoding='utf-8')  # 以'w'方式打开文件
        f.write(self.folder_path.split('/')[-1] + '\n')
        f.write('annotations:' + '\n')
        for index, (name) in enumerate(self.annotation_name):
            f.write(str(index) + ' ' + name + '\n')
        f.write(f'noiseless.ply:{self.noiseless_point_cloud}\n')
        f.write(f'clean_100k.ply:{self.sample_point_clouds_from_mesh[0]}\n')
        f.write(f'clean_10k.ply:{self.sample_point_clouds_from_mesh[1]}\n')
        f.write(f'clean_1k.ply:{self.sample_point_clouds_from_mesh[2]}\n')
        f.write(f'real_100k.ply:{self.sample_point_clouds_from_point_cloud[0]}\n')
        f.write(f'real_10k.ply:{self.sample_point_clouds_from_point_cloud[1]}\n')
        f.write(f'real_1k.ply:{self.sample_point_clouds_from_point_cloud[2]}\n')
        f.write(f'mesh.ply:{self.reconstructed_mesh}\n')
        f.close()

    def remove_noise(self):
        '''
        Function to remove points that have less than nb_points in a given sphere of a given radius
        
        The parameters of remove_radius_outlier:
        nb_points (int) – Number of points within the radius.
        radius (float) – Radius of the sphere.
        
        You can change the parameters of remove_radius_outlier
        '''
        self.noiseless_point_cloud, _ = self.raw_point_cloud.remove_radius_outlier(nb_points=5, radius=1.0)

    def reconstruct_mesh(self):
        '''
        Function that computes a triangle mesh from a oriented PointCloud pcd. 
        This implements the Screened Poisson Reconstruction proposed in Kazhdan and Hoppe, “Screened Poisson Surface Reconstruction”, 2013. 

        The parameters of o3d.geometry.TriangleMesh.create_from_point_cloud_poisson:
        pcd (open3d.geometry.PointCloud) – PointCloud from which the TriangleMesh surface is reconstructed. Has to contain normals.
        depth (int, optional, default=8) – Maximum depth of the tree that will be used for surface reconstruction. Running at depth d corresponds to solving on a grid whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the reconstructor adapts the octree to the sampling density, the specified reconstruction depth is only an upper bound.
        width (int, optional, default=0) – Specifies the target width of the finest level octree cells. This parameter is ignored if depth is specified
        scale (float, optional, default=1.1) – Specifies the ratio between the diameter of the cube used for reconstruction and the diameter of the samples’ bounding cube.
        linear_fit (bool, optional, default=False) – If true, the reconstructor use linear interpolation to estimate the positions of iso-vertices.

        You can change the parameters of creatre_from_point_cloud_poisson
        '''
        [self.reconstructed_mesh, _] = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.noiseless_point_cloud)
        self.reconstructed_mesh = self.reconstructed_mesh.remove_degenerate_triangles()
        self.reconstructed_mesh = self.reconstructed_mesh.remove_duplicated_triangles()
        self.reconstructed_mesh = self.reconstructed_mesh.remove_duplicated_vertices()
        self.reconstructed_mesh = self.reconstructed_mesh.remove_non_manifold_edges()
        # self.reconstructed_mesh = self.reconstructed_mesh.merge_close_vertices(0.5)

    def down_sample_to_100k_10k_1k_from_mesh(self):
        down_sample_100k = self.reconstructed_mesh.sample_points_poisson_disk(100000)
        down_sample_10k = self.reconstructed_mesh.sample_points_poisson_disk(10000)
        down_sample_1k = self.reconstructed_mesh.sample_points_poisson_disk(1000)
        self.sample_point_clouds_from_mesh.append(down_sample_100k)
        self.sample_point_clouds_from_mesh.append(down_sample_10k)
        self.sample_point_clouds_from_mesh.append(down_sample_1k)

    def down_sample_to_100k_10k_1k_from_pointcloud(self):
        self.voxel_down_sample(1 / 10)
        self.voxel_down_sample(2 / 10)
        sample_100k = None
        sample_10k = None
        sample_1k = None
        voxel_size = 2.0
        while sample_1k is None:
            denser_sample = self.sample_point_clouds_tmp[-2]
            sparser_sample = self.sample_point_clouds_tmp[-1]
            denser_sample_cnt = np.asarray(denser_sample.points).shape[0]
            sparser_sample_cnt = np.asarray(sparser_sample.points).shape[0]
            if sparser_sample_cnt < 10:
                print('\tWarning: sample_1k is not sampled')
                break
            if denser_sample_cnt >= 100000 > sparser_sample_cnt:
                sample_100k = denser_sample
            if denser_sample_cnt >= 10000 > sparser_sample_cnt:
                sample_10k = denser_sample
            if denser_sample_cnt >= 1000 > sparser_sample_cnt:
                sample_1k = denser_sample
            voxel_size += 1
            self.voxel_down_sample(voxel_size / 10)
        self.sample_point_clouds_from_point_cloud = [sample_100k, sample_10k, sample_1k]

    def voxel_down_sample(self, voxel_size):
        '''
        voxel_size (float) – Voxel size to downsample into.

        Function to downsample input pointcloud into output pointcloud with a voxel
        '''
        self.sample_point_clouds_tmp.append(self.noiseless_point_cloud.voxel_down_sample(voxel_size))

    def make_annotation(self):
        for anno_id, (target) in enumerate(self.segmentation_point_cloud):
            vt = self.sample_point_clouds_from_mesh[0].compute_point_cloud_distance(target)
            indices = []
            for i, (ky) in enumerate(vt):
                if ky < 0.2:
                    indices.append(i)
                    self.annotation_dict[i] = anno_id
            pt = self.sample_point_clouds_from_mesh[0].select_down_sample(indices)
            # o3d.visualization.draw_geometries([pt])         

    def save_ply(self):
        green_print('Saving!')
        assert self.noiseless_point_cloud is not None
        assert self.reconstructed_mesh is not None
        assert len(self.sample_point_clouds_from_mesh) > 0
        assert len(self.sample_point_clouds_from_point_cloud) > 0
        save_floder_name = os.path.basename(self.folder_path)
        if os.path.exists(os.path.join(self.save_path, save_floder_name)):
            red_print(f'{os.path.join(self.save_path, save_floder_name)} already exists')
        os.mkdir(os.path.join(self.save_path, save_floder_name))
        o3d.io.write_point_cloud(os.path.join(self.save_path, save_floder_name, 'noiseless.ply'),
                                 self.noiseless_point_cloud)
        o3d.io.write_point_cloud(os.path.join(self.save_path, save_floder_name, 'clean_100k.ply'),
                                 self.sample_point_clouds_from_mesh[0])
        o3d.io.write_point_cloud(os.path.join(self.save_path, save_floder_name, 'clean_10k.ply'),
                                 self.sample_point_clouds_from_mesh[1])
        o3d.io.write_point_cloud(os.path.join(self.save_path, save_floder_name, 'clean_1k.ply'),
                                 self.sample_point_clouds_from_mesh[2])
        real_100k, real_10k, real_1k = self.sample_point_clouds_from_point_cloud
        if real_1k is None:
            raise ValueError("sampling points is None, cannot save")
        if real_100k is not None:
            o3d.io.write_point_cloud(os.path.join(self.save_path, save_floder_name, 'real_100k.ply'), real_100k)
        if real_10k is not None:
            o3d.io.write_point_cloud(os.path.join(self.save_path, save_floder_name, 'real_10k.ply'), real_10k)
        o3d.io.write_point_cloud(os.path.join(self.save_path, save_floder_name, 'real_1k.ply'), real_1k)
        o3d.io.write_triangle_mesh(os.path.join(self.save_path, save_floder_name, 'mesh.ply'), self.reconstructed_mesh)

        f = open(os.path.join(self.save_path, save_floder_name, 'annotation.txt'), 'w', encoding='utf-8')  # 以'w'方式打开文件
        for k, v in self.annotation_dict.items():  # 遍历字典中的键值
            f.write(f'{k} {v}\n')
        f.close()

        self.wirte_info(os.path.join(self.save_path, save_floder_name))
        green_print('Done!')

    def move_to_origin(self):
        self.noiseless_point_cloud = self.noiseless_point_cloud.translate(self.translate_matrix)
        self.sample_point_clouds_from_mesh[0] = self.sample_point_clouds_from_mesh[0].translate(self.translate_matrix)
        self.sample_point_clouds_from_mesh[1] = self.sample_point_clouds_from_mesh[1].translate(self.translate_matrix)
        self.sample_point_clouds_from_mesh[2] = self.sample_point_clouds_from_mesh[2].translate(self.translate_matrix)
        if self.sample_point_clouds_from_point_cloud[0] is not None:
            self.sample_point_clouds_from_point_cloud[0] = self.sample_point_clouds_from_point_cloud[0].translate(
                self.translate_matrix)
        self.sample_point_clouds_from_point_cloud[1] = self.sample_point_clouds_from_point_cloud[1].translate(
            self.translate_matrix)
        self.sample_point_clouds_from_point_cloud[2] = self.sample_point_clouds_from_point_cloud[2].translate(
            self.translate_matrix)
        self.reconstructed_mesh = self.reconstructed_mesh.translate(self.translate_matrix)

    def run(self):
        green_print('Processing:')
        green_print(self.folder_path)
        self.load_point_cloud()
        logging.info(self.folder_path)
        green_print('Remove noise')
        logging.info('Remove noise')
        try:
            start = datetime.datetime.now()
            self.remove_noise()
            cost_time = (datetime.datetime.now() - start).seconds
            green_print(f'Remove noise finished, cost {cost_time}s')
            logging.info(f'Remove noise finished, cost {cost_time}s')
        except Exception as e:
            red_print("Remove noise error")
            logging.warning("Remove noise error")
            print(traceback.format_exc())
            print(e)

        green_print('Down sampling from point cloud')
        try:
            start = datetime.datetime.now()
            self.down_sample_to_100k_10k_1k_from_pointcloud()
            green_print(f'Down sampling from point cloud finished, cost {(datetime.datetime.now() - start).seconds}s')
            logging.info(f'Down sampling from point cloud finished, cost {(datetime.datetime.now() - start).seconds}s')
        except Exception as e:
            red_print('Down sampling error')
            logging.warning("Down sampling error")
            print(traceback.format_exc())
            print(e)

        green_print('Reconstruct mesh')
        logging.info('Reconstruct mesh')
        try:
            start = datetime.datetime.now()
            self.reconstruct_mesh()
            cost_time = (datetime.datetime.now() - start).seconds
            green_print(f'Reconstruct mesh finished, cost {cost_time}s')
            logging.info(f'Reconstruct mesh finished, cost {cost_time}s')
            # o3d.visualization.draw_geometries([self.reconstructed_mesh])
        except Exception as e:
            red_print("Reconstruct mesh error")
            logging.warning("Reconstruct mesh error")
            print(traceback.format_exc())
            print(e)

        green_print('Down sampling from mesh')
        try:
            start = datetime.datetime.now()
            self.down_sample_to_100k_10k_1k_from_mesh()
            green_print(f'Down sampling from mesh finished, cost {(datetime.datetime.now() - start).seconds}s')
            logging.info(f'Down sampling from mesh finished, cost {(datetime.datetime.now() - start).seconds}s')
        except Exception as e:
            red_print('Down sampling from mesh error')
            logging.warning("Down sampling from mesh error")
            print(traceback.format_exc())
            print(e)

        try:
            self.make_annotation()
        except Exception as e:
            red_print('Make annotation error')
            logging.warning("Make annotation error")
            print(traceback.format_exc())
            print(e)

        try:
            self.move_to_origin()
        except Exception as e:
            red_print('Move to origin error')
            logging.warning("Move to origin error")
            print(traceback.format_exc())
            print(e)

        self.print_info()

        try:
            self.save_ply()
        except Exception as e:
            red_print('Save ply error')
            logging.warning("Save ply error")
            print(traceback.format_exc())
            print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, help='src', default='D:/code/data')
    parser.add_argument('-d', type=str, help='dst', default='res')
    args = parser.parse_args()
    dst_path = args.d
    src_path = args.s

    src_floder = []
    dst_floder = []
    count = 0
    for file in os.listdir(f'{dst_path}'):
        if file == '.DS_Store':
            continue
        dst_floder.append(file)

    for file in os.listdir(f'{src_path}'):
        if file == '.DS_Store':
            continue
        src_floder.append(file)

    blue_print(f'Existing files: {dst_floder}')
    blue_print(f'{len(dst_floder)} files already exist, skip!')

    blue_print(f'{len(src_floder) - len(dst_floder)} files are ready to go')

    start = datetime.datetime.now()
    for file in src_floder:
        if file in dst_floder:
            continue
        # for debug
        # rp = RPCDPrepreocess(r'C:\Users\86189\Documents\data-20200118\1084_fj', save_path='.')
        rp = RPCDPrepreocess(os.path.join(f'{src_path}', file), save_path=dst_path)
        rp.run()
        count += 1
        blue_print(f'Finished {count}/{len(src_floder) - len(dst_floder)}\n')
        blue_print(f'Totally cost {(datetime.datetime.now() - start).seconds}s')


if __name__ == "__main__":
    main()
