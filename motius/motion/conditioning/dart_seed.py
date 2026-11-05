import time, random, pickle
from pathlib import Path
from os.path import join as pjoin
import numpy as np
import torch
from omegaconf import OmegaConf
from motius.motion.representation import dart_rotation as transforms
from motius.motion.representation.dart import PrimitiveUtility, tensor_dict_to_device
from motius.motion.conditioning.dart import encode_text, have_overlap

class DARTSeedData:

    def __init__(self, cfg_path=None, sequence_path=None, dataset_path=None, device='cuda', batch_size=16, texts=None, enforce_gender=None, enforce_zero_beta=None, clip_to_seq_length=True, body_type='smplx', **kwargs):
        self.batch_size = batch_size
        self.device = device
        self.prob_static = 0.0
        self.enforce_gender = enforce_gender
        self.enforce_zero_beta = enforce_zero_beta
        self.weight_scheme = 'uniform'
        self.enforce_gender = enforce_gender
        self.enforce_zero_beta = enforce_zero_beta
        self.clip_to_seq_length = clip_to_seq_length
        self.primitive_utility = PrimitiveUtility(device=self.device, body_type=body_type)
        self.motion_repr = self.primitive_utility.motion_repr
        with open(cfg_path, 'r') as f:
            self.cfg = OmegaConf.load(f)
        self.target_fps = self.cfg.fps
        self.history_length = self.cfg.history_length
        self.future_length = self.cfg.future_length
        self.primitive_length = self.history_length + self.future_length
        self.num_primitive = 1
        self.seq_length = self.history_length + self.future_length * self.num_primitive + 1
        self.tensor_mean_device_dict = {}
        if str(cfg_path).endswith('/motion_primitive/mp_2_8.yaml'):
            mean_std_path = pjoin(dataset_path, 'mean_std.pkl')
            with open(mean_std_path, 'rb') as f:
                self.mean_std = pickle.load(f)
            mean_dict = {}
            std_dict = {}
            for key in self.mean_std:
                mean_dict[key] = self.mean_std[key]['mean']
                std_dict[key] = self.mean_std[key]['std']
            self.tensor_mean = self.dict_to_tensor(mean_dict).reshape(1, 1, -1)
            self.tensor_std = self.dict_to_tensor(std_dict).reshape(1, 1, -1)
        else:
            mean_std_path = Path(dataset_path, f'mean_std_h{self.history_length}_f{self.future_length}.pkl')
            if mean_std_path.exists():
                print(f'loading mean and std from {mean_std_path}')
                with open(mean_std_path, 'rb') as f:
                    self.tensor_mean, self.tensor_std = pickle.load(f)
            else:
                raise FileNotFoundError(mean_std_path)
        self.clip_model = kwargs['clip_model']
        self.update_seq(sequence_path)

    def update_seq(self, sequence_path):
        with open(sequence_path, 'rb') as f:
            self.sequence = pickle.load(f)
        text_prompt = self.sequence['texts'][0] if 'texts' in self.sequence else ''
        clip_length = self.seq_length if self.clip_to_seq_length else len(self.sequence['transl'])
        body_pose = torch.tensor(self.sequence['body_pose'][:clip_length], dtype=torch.float32)
        if len(body_pose.shape) > 2 and body_pose.shape[-2:] == (3, 3):
            body_pose = transforms.matrix_to_axis_angle(body_pose).reshape(-1, 63)
        global_orient = torch.tensor(self.sequence['global_orient'][:clip_length], dtype=torch.float32)
        if len(global_orient.shape) > 2 and global_orient.shape[-2:] == (3, 3):
            global_orient = transforms.matrix_to_axis_angle(global_orient).reshape(-1, 3)
        poses = torch.cat([global_orient, body_pose], dim=1)
        transl = torch.tensor(self.sequence['transl'][:clip_length], dtype=torch.float32)
        if poses.shape[0] < self.seq_length:
            poses = torch.cat([poses, poses[-1].unsqueeze(0).expand(self.seq_length - poses.shape[0], -1)], dim=0)
            transl = torch.cat([transl, transl[-1].unsqueeze(0).expand(self.seq_length - transl.shape[0], -1)], dim=0)
        self.dataset = [{'motion': {'gender': self.sequence['gender'] if self.enforce_gender is None else self.enforce_gender, 'betas': torch.tensor(self.sequence['betas'], dtype=torch.float32) if not self.enforce_zero_beta else torch.zeros(10, dtype=torch.float32), 'trans': transl, 'poses': poses}, 'text': text_prompt}]
        self.seq_weights = np.array([1.0])
        text_embedding = encode_text(self.clip_model, [text_prompt])
        self.text_embedding_dict = {text_prompt: text_embedding[0]}

    def update_text_embedding_dict(self, new_texts):
        new_text_embeddings = encode_text(self.clip_model, new_texts)
        for idx, text in enumerate(new_texts):
            self.text_embedding_dict[text] = new_text_embeddings[idx]

    def dict_to_tensor(self, data_dict):
        return self.primitive_utility.dict_to_tensor(data_dict)

    def tensor_to_dict(self, tensor):
        return self.primitive_utility.tensor_to_dict(tensor)

    def get_mean_std_by_device(self, device):
        if device not in self.tensor_mean_device_dict:
            self.tensor_mean_device_dict[device] = (self.tensor_mean.to(device=device), self.tensor_std.to(device=device))
        return self.tensor_mean_device_dict[device]

    def normalize(self, tensor):
        tensor_mean, tensor_std = self.get_mean_std_by_device(tensor.device)
        return (tensor - tensor_mean) / tensor_std

    def denormalize(self, tensor):
        tensor_mean, tensor_std = self.get_mean_std_by_device(tensor.device)
        return tensor * tensor_std + tensor_mean

    def get_primitive(self, seq_data, start_frame, end_frame, skip_text=False):
        """end_frame included"""
        motion_data = seq_data['motion']
        primitive_dict = {'gender': motion_data['gender'], 'betas': motion_data['betas'].expand(1, self.primitive_length + 1, 10), 'transl': motion_data['trans'][start_frame:end_frame + 1].unsqueeze(0), 'global_orient': transforms.axis_angle_to_matrix(motion_data['poses'][start_frame:end_frame + 1, :3].unsqueeze(0)), 'body_pose': transforms.axis_angle_to_matrix(motion_data['poses'][start_frame:end_frame + 1, 3:66].unsqueeze(0).reshape(1, end_frame - start_frame + 1, 21, 3)), 'transf_rotmat': torch.eye(3).unsqueeze(0), 'transf_transl': torch.zeros(1, 1, 3)}
        texts = []
        if not skip_text and 'frame_labels' in seq_data:
            future_start = (start_frame + self.history_length) / self.target_fps
            future_end = (start_frame + self.history_length + self.future_length - 1) / self.target_fps
            for seg in seq_data['frame_labels']:
                if have_overlap([seg['start_t'], seg['end_t']], [future_start - self.text_tolerance, future_end + self.text_tolerance]):
                    texts.append(seg['proc_label'])
        output = {'text': random.choice(texts) if len(texts) > 0 else '', 'primitive_dict': primitive_dict}
        return output

    def get_batch_idx(self, batch_size=8):
        batch_idx = np.random.choice(len(self.dataset), size=batch_size, replace=True, p=self.seq_weights)
        return batch_idx

    def get_batch(self, batch_size=8):
        self.time = time.time()
        seq_list = []
        batch_idx = self.get_batch_idx(batch_size)
        for seq_idx in batch_idx:
            seq_data = self.dataset[seq_idx]
            num_frames = len(seq_data['motion']['trans'])
            if self.prob_static > 0 and random.random() < self.prob_static:
                static_frame = random.randint(0, num_frames - 1)
                motion_data = seq_data['motion']
                primitive_length = self.primitive_length
                primitive_dict = {'gender': motion_data['gender'], 'betas': motion_data['betas'].expand(1, primitive_length + 1, 10), 'transl': motion_data['trans'][[static_frame]].expand(primitive_length + 1, -1).unsqueeze(0), 'global_orient': transforms.axis_angle_to_matrix(motion_data['poses'][[static_frame], :3].expand(primitive_length + 1, -1).unsqueeze(0)), 'body_pose': transforms.axis_angle_to_matrix(motion_data['poses'][[static_frame], 3:66].expand(primitive_length + 1, -1).unsqueeze(0).reshape(1, primitive_length + 1, 21, 3)), 'transf_rotmat': torch.eye(3).unsqueeze(0), 'transf_transl': torch.zeros(1, 1, 3)}
                primitive_data = {'text': '', 'primitive_dict': primitive_dict}
                primitive_data_list = [primitive_data] * self.num_primitive
            else:
                if 'text' in self.weight_scheme:
                    start_frame = random.choices(range(num_frames - self.seq_length + 1), weights=seq_data['frame_weights'], k=1)[0]
                else:
                    start_frame = random.randint(0, num_frames - self.seq_length)
                primitive_data_list = []
                for frame_idx in range(start_frame, start_frame + self.seq_length - self.primitive_length, self.future_length):
                    primitive_data = self.get_primitive(seq_data, frame_idx, frame_idx + self.primitive_length)
                    primitive_data_list.append(primitive_data)
            seq_list.append(primitive_data_list)
        batch = None
        for gender in ['female', 'male']:
            gender_idx = [idx for idx in range(len(seq_list)) if seq_list[idx][0]['primitive_dict']['gender'] == gender]
            if len(gender_idx) == 0:
                continue
            gender_seq_list = [seq_list[i] for i in gender_idx]
            gender_batch_size = len(gender_idx)
            gender_batch = []
            gender_seq_texts = None
            gender_seq_dict = None
            for primitive_idx in range(self.num_primitive):
                primitive_texts = [mp_seq[primitive_idx]['text'] for mp_seq in gender_seq_list]
                primitive_dict = {'gender': gender}
                for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl']:
                    primitive_dict[key] = torch.cat([mp_seq[primitive_idx]['primitive_dict'][key] for mp_seq in gender_seq_list], dim=0)
                gender_seq_texts = primitive_texts if gender_seq_texts is None else gender_seq_texts + primitive_texts
                if gender_seq_dict is None:
                    gender_seq_dict = primitive_dict
                else:
                    for key in ['betas', 'transl', 'global_orient', 'body_pose', 'transf_rotmat', 'transf_transl']:
                        gender_seq_dict[key] = torch.cat([gender_seq_dict[key], primitive_dict[key]], dim=0)
            gender_seq_dict = tensor_dict_to_device(gender_seq_dict, self.device)
            _, _, canonicalized_primitive_dict = self.primitive_utility.canonicalize(gender_seq_dict)
            transf_rotmat, transf_transl = (canonicalized_primitive_dict['transf_rotmat'], canonicalized_primitive_dict['transf_transl'])
            feature_dict = self.primitive_utility.calc_features(canonicalized_primitive_dict)
            feature_dict['transl'] = feature_dict['transl'][:, :-1, :]
            feature_dict['poses_6d'] = feature_dict['poses_6d'][:, :-1, :]
            feature_dict['joints'] = feature_dict['joints'][:, :-1, :]
            motion_tensor_normalized = self.normalize(self.dict_to_tensor(feature_dict))
            motion_tensor_normalized = motion_tensor_normalized.permute(0, 2, 1).unsqueeze(2)
            history_mask = torch.zeros_like(motion_tensor_normalized, dtype=torch.bool, device=self.device)
            history_mask[..., :self.cfg.history_length] = True
            history_motion = torch.zeros_like(motion_tensor_normalized, dtype=torch.float32, device=self.device)
            history_motion[..., :self.cfg.history_length] = motion_tensor_normalized[..., :self.cfg.history_length]
            for primitive_idx in range(self.num_primitive):
                start_idx = primitive_idx * gender_batch_size
                end_idx = (primitive_idx + 1) * gender_batch_size
                primitive_texts = gender_seq_texts[start_idx:end_idx]
                unseen_texts = [text for text in primitive_texts if text not in self.text_embedding_dict]
                if len(unseen_texts) > 0:
                    self.update_text_embedding_dict(unseen_texts)
                text_embedding = torch.stack([self.text_embedding_dict[text] for text in primitive_texts], dim=0)
                gender_batch.append({'texts': primitive_texts, 'text_embedding': text_embedding, 'gender': [gender_seq_dict['gender']] * gender_batch_size, 'betas': gender_seq_dict['betas'][start_idx:end_idx, :-1, :10], 'motion_tensor_normalized': motion_tensor_normalized[start_idx:end_idx, ...], 'history_motion': history_motion[start_idx:end_idx, ...], 'history_mask': history_mask[start_idx:end_idx, ...], 'history_length': self.cfg.history_length, 'future_length': self.cfg.future_length, 'transf_rotmat': transf_rotmat, 'transf_transl': transf_transl})
            if batch is None:
                batch = gender_batch
            else:
                for primitive_idx in range(self.num_primitive):
                    for key in ['texts', 'gender']:
                        batch[primitive_idx][key] = batch[primitive_idx][key] + gender_batch[primitive_idx][key]
                    for key in ['betas', 'motion_tensor_normalized', 'history_motion', 'history_mask', 'text_embedding', 'transf_rotmat', 'transf_transl']:
                        batch[primitive_idx][key] = torch.cat([batch[primitive_idx][key], gender_batch[primitive_idx][key]], dim=0)
        return batch
