import argparse
import os
import os.path as osp

from easydict import EasyDict as edict

from vision3d.utils.io import ensure_dir

_C = edict()

# exp
_C.exp = edict()
_C.exp.name = osp.basename(osp.dirname(osp.realpath(__file__)))
_C.exp.working_dir = osp.dirname(osp.realpath(__file__))
_C.exp.output_dir = osp.join("/workspace/vision3d-output", _C.exp.name)
_C.exp.checkpoint_dir = osp.join(_C.exp.output_dir, "checkpoints")
_C.exp.log_dir = osp.join(_C.exp.output_dir, "logs")
_C.exp.event_dir = osp.join(_C.exp.output_dir, "events")
_C.exp.cache_dir = osp.join(_C.exp.output_dir, "cache")
_C.exp.result_dir = osp.join(_C.exp.output_dir, "results")
_C.exp.seed = 7351

ensure_dir(_C.exp.output_dir)
ensure_dir(_C.exp.checkpoint_dir)
ensure_dir(_C.exp.log_dir)
ensure_dir(_C.exp.event_dir)
ensure_dir(_C.exp.cache_dir)
ensure_dir(_C.exp.result_dir)

# data
_C.data = edict()
_C.data.dataset_dir = "../../data/r3live/sequences"
# Image resolution after resize. Must satisfy: H % 24 == 0 and W % 32 == 0.
# Adjust to match the actual camera resolution (or the nearest valid size).
_C.data.image_h = 480
_C.data.image_w = 640
# Outdoor LiDAR: use 2x the indoor matching radii.
_C.data.matching_radius_3d = 0.075
_C.data.matching_radius_2d = 8.0
_C.data.overlap_threshold = None

# Train/val/test split by sequence name (subdirectory names under dataset_dir).
# Set all to None to auto-detect sequences and apply a 70%/15%/15% split.
_C.data.train_sequences = ["01", "02", "03", "05", "06", "07", "09"]
_C.data.val_sequences   = ["00", "04", "08"]
_C.data.test_sequences  = ["10"]

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.max_points = 30000

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.max_points = 30000

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.3
_C.eval.acceptance_radius = 0.1    # 2x indoor value for outdoor scale
_C.eval.inlier_ratio_threshold = 0.1
_C.eval.rmse_threshold = 0.2       # 2x indoor value for outdoor scale

# ransac
_C.ransac = edict()
_C.ransac.distance_tolerance = 8.0
_C.ransac.num_iterations = 50000

# trainer
_C.trainer = edict()
_C.trainer.max_epoch = 20
_C.trainer.grad_acc_steps = 1

# optim
_C.optimizer = edict()
_C.optimizer.type = "Adam"
_C.optimizer.lr = 1e-4
_C.optimizer.weight_decay = 1e-6

# scheduler
_C.scheduler = edict()
_C.scheduler.type = "Step"
_C.scheduler.gamma = 0.95
_C.scheduler.step_size = 1

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius_2d = 8.0
_C.model.ground_truth_matching_radius_3d = 0.075
_C.model.pcd_num_points_in_patch = 32
# Depth limit for back-projection of the synthetic depth image.
# Indoor datasets use 6.0 m; outdoor LiDAR data needs a larger value.
_C.model.depth_limit = 50.0

# model - image backbone
_C.model.image_backbone = edict()
_C.model.image_backbone.input_dim = 1
_C.model.image_backbone.output_dim = 128
_C.model.image_backbone.init_dim = 128
_C.model.image_backbone.dilation = 1

# model - point backbone (20 cm voxels for outdoor frustum-cropped LiDAR).
# With ~43k in-frustum points and base=0.20m, the coarsest level (1.6m) has
# ~800 nodes, keeping the transformer O(N^2) attention well within GPU memory.
_C.model.point_backbone = edict()
_C.model.point_backbone.num_stages = 4
_C.model.point_backbone.base_voxel_size = 0.20
_C.model.point_backbone.kernel_size = 15
_C.model.point_backbone.kpconv_radius = 2.5
_C.model.point_backbone.kpconv_sigma = 2.0
_C.model.point_backbone.input_dim = 1
_C.model.point_backbone.init_dim = 64
_C.model.point_backbone.output_dim = 128

# model - Coarse Matching
_C.model.coarse_matching = edict()
_C.model.coarse_matching.num_targets = 128
_C.model.coarse_matching.overlap_threshold = 0.3
_C.model.coarse_matching.num_correspondences = 96
_C.model.coarse_matching.topk = 5
_C.model.coarse_matching.similarity_threshold = 0.85
_C.model.coarse_matching.dual_normalization = False

# model - GeoTransformer
_C.model.transformer = edict()
_C.model.transformer.img_input_dim = 512
_C.model.transformer.pcd_input_dim = 1024
_C.model.transformer.hidden_dim = 256
_C.model.transformer.output_dim = 256
_C.model.transformer.num_heads = 4
_C.model.transformer.blocks = ["self", "cross", "self", "cross", "self", "cross"]
_C.model.transformer.use_embedding = True

# model - Fine Matching
_C.model.fine_matching = edict()
_C.model.fine_matching.topk = 2
_C.model.fine_matching.mutual = True
_C.model.fine_matching.confidence_threshold = 0.05
_C.model.fine_matching.use_dustbin = False
_C.model.fine_matching.use_global_score = False

# loss
_C.loss = edict()

# loss - Coarse level
_C.loss.coarse_loss = edict()
_C.loss.coarse_loss.positive_margin = 0.1
_C.loss.coarse_loss.negative_margin = 1.4
_C.loss.coarse_loss.positive_optimal = 0.1
_C.loss.coarse_loss.negative_optimal = 1.4
_C.loss.coarse_loss.log_scale = 40
_C.loss.coarse_loss.positive_overlap = 0.3
_C.loss.coarse_loss.negative_overlap = 0.2
_C.loss.coarse_loss.weight = 1.0

# loss - Fine level (radii scaled 2x for outdoor)
_C.loss.fine_loss = edict()
_C.loss.fine_loss.positive_margin = 0.1
_C.loss.fine_loss.negative_margin = 1.4
_C.loss.fine_loss.positive_optimal = 0.1
_C.loss.fine_loss.negative_optimal = 1.4
_C.loss.fine_loss.log_scale = 24
_C.loss.fine_loss.positive_radius_3d = 0.075
_C.loss.fine_loss.negative_radius_3d = 0.2
_C.loss.fine_loss.positive_radius_2d = 8.0
_C.loss.fine_loss.negative_radius_2d = 12.0
_C.loss.fine_loss.max_correspondences = 256
_C.loss.fine_loss.weight = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true")
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
