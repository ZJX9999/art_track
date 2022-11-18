from src.dataset.pose import PoseDataset


class MPII(PoseDataset):

    def mirror_joint_coords(self, joints, image_width):
        joints[:, 1] = image_width - joints[:, 1]
        return joints

    def get_pose_segments(self):
        return [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
