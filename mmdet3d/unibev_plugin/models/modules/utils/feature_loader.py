import torch
import os
import pickle
from typing import List, Dict, Tuple, Union

class UniBevFeatureLoader:
    def __init__(self, base_path: str = "/dataset/shuangzhi/mmdet3d/unibev_LC_features/frames", device: str = 'cuda', expected_shape = (1, 40000, 256)):
        """
        UniBev feature data loader with history frame support
        
        Args:
            base_path (str): Base path to feature data
        """
        self.base_path = base_path
        self.frame_meta = None  # Will load frame metadata for history lookup
        self.device = device
        self.expected_shape = expected_shape
    
    def load_frame_metadata(self, metadata_path: str):
        """
        Load frame metadata for history frame lookup
        
        Args:
            metadata_path (str): Path to frame metadata (pickle or json file)
        """
        if metadata_path.endswith('.pkl'):
            with open(metadata_path, 'rb') as f:
                self.frame_meta = pickle.load(f)
        elif metadata_path.endswith('.json'):
            import json
            with open(metadata_path, 'r') as f:
                self.frame_meta = json.load(f)
        else:
            raise ValueError("Metadata file must be .pkl or .json")
        
        print(f"Loaded metadata for {len(self.frame_meta)} frames")
    
    def load_single_frame(self, frame_name: str) -> Dict[str, torch.Tensor]:
        """
        Load feature data for a single frame
        
        Args:
            frame_name (str): Point cloud frame name
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing feature data
        """
        file_path = os.path.join(self.base_path, frame_name, "vis_data.pt")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        
        data = torch.load(file_path, map_location=self.device)
        return data
    
    def get_history_frames(self, current_frame_name: str, max_history: int = 6) -> List[Dict]:
        """
        Get history frames for a given current frame
        
        Args:
            current_frame_name (str): Current frame name
            max_history (int): Maximum number of history frames
        
        Returns:
            List[Dict]: List containing current frame and history frames with metadata
        """
        if self.frame_meta is None:
            raise ValueError("Frame metadata not loaded. Call load_frame_metadata() first.")
        
        if current_frame_name not in self.frame_meta:
            raise KeyError(f"Frame {current_frame_name} not found in metadata")
        
        frame_sequence = []
        current_frame_data = self.frame_meta[current_frame_name]
        
        # Add current frame (index 0)
        frame_sequence.append({
            'frame_name': current_frame_name,
            'timestamp': current_frame_data['current_frame']['timestamp'],
            'relative_time': 0.0,  # Current frame has 0 relative time
            'frame_index': 0
        })
        
        # Get available history frames
        available_history = current_frame_data.get('history_frames', [])
        history_count = min(len(available_history), max_history)
        
        # Add available history frames
        for i in range(history_count):
            hist_frame = available_history[i]
            relative_time = current_frame_data['current_frame']['timestamp'] - hist_frame['timestamp']
            
            frame_sequence.append({
                'frame_name': hist_frame['pointcloud_filename'],
                'timestamp': hist_frame['timestamp'],
                'relative_time': relative_time,
                'frame_index': i + 1
            })
        
        # If we need more frames, pad with the oldest available frame
        if len(frame_sequence) - 1 < max_history:  # -1 because we exclude current frame from history count
            assert len(frame_sequence) > 1, f'At least one history frame is required in frame {current_frame_name}'
            oldest_frame = frame_sequence[-1]
            needed_frames = max_history - (len(frame_sequence) - 1)
            
            for i in range(needed_frames):
                frame_sequence.append({
                    'frame_name': oldest_frame['frame_name'],
                    'timestamp': oldest_frame['timestamp'],
                    'relative_time': oldest_frame['relative_time'],
                    'frame_index': len(frame_sequence),
                    'is_padded': True
                })
        
        return frame_sequence[:max_history + 1]  # +1 for current frame
    
    def load_batch_features_with_history(self, frame_names: List[str], 
                                       feature_keys: List[str] = ['ori_img_bev_embed', 'ori_pts_bev_embed'],
                                       max_history: int = 6, verbose: bool = False,
                                       device: str = 'cuda') -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """
        Load feature data in batch with history frames
        
        Args:
            frame_names (List[str]): List of current frame names with length B
            feature_keys (List[str]): List of feature keys to extract
            max_history (int): Maximum number of history frames
        
        Returns:
            Dict containing:
                - feature_sequence: List of [current_batch, hist1_batch, hist2_batch, ...]
                - relative_times: Tensor of shape (B, max_history+1) with relative time differences
                - frame_info: List of frame sequence info for each batch item
        """
        batch_size = len(frame_names)
        
        # Initialize output structure
        result = {
            'feature_sequence': {},  # Will contain lists of tensors for each feature key
            'relative_times': [],    # Relative time differences
            'frame_info': []         # Frame metadata for each batch item
        }
        
        # Initialize feature sequence lists
        for key in feature_keys:
            result['feature_sequence'][key] = [[] for _ in range(max_history + 1)]  # +1 for current frame
        
        # Process each current frame
        for batch_idx, current_frame_name in enumerate(frame_names):
            # Get frame sequence (current + history)
            frame_sequence = self.get_history_frames(current_frame_name, max_history)
            result['frame_info'].append(frame_sequence)
            
            # Extract relative times
            relative_times = [frame['relative_time'] for frame in frame_sequence]
            result['relative_times'].append(relative_times)
            
            # Load features for each frame in sequence
            for seq_idx, frame_info in enumerate(frame_sequence):
                frame_name = frame_info['frame_name']
                # Load frame features
                frame_data = self.load_single_frame(frame_name)
                
                for key in feature_keys:
                    # Check feature key
                    assert key in frame_data, f"Error: {frame_name} - {key} not found in frame data"
                    feature = frame_data[key]  # Shape: (1, 40000, 256)
                    
                    # Check feature dimensions
                    assert feature.shape == self.expected_shape, f"Error: {frame_name} - {key} shape {feature.shape} != expected {self.expected_shape}"
                    
                    result['feature_sequence'][key][seq_idx].append(feature)
                
        
        # Convert lists to tensors
        for key in feature_keys:
            for seq_idx in range(max_history + 1):
                # Concatenate batch dimension: (1, 40000, 256) * B -> (B, 40000, 256)
                result['feature_sequence'][key][seq_idx] = torch.cat(
                    result['feature_sequence'][key][seq_idx], dim=0
                ).to(device)
        
        # Convert relative times to tensor
        result['relative_times'] = torch.tensor(result['relative_times'], dtype=torch.float32, device=device)  # (B, max_history+1)
        
        # Print summary
        if verbose:
            print(f"Loaded feature sequences for {batch_size} frames:")
            for key in feature_keys:
                print(f"  {key}: {len(result['feature_sequence'][key])} time steps")
                for seq_idx, tensor in enumerate(result['feature_sequence'][key]):
                    step_name = "current" if seq_idx == 0 else f"history-{seq_idx}"
                    print(f"    {step_name}: {tensor.shape}")
            print(f"  relative_times: {result['relative_times']}")
        
        return result
    
    def check_data_availability(self, frame_names: List[str]) -> Dict[str, bool]:
        """
        Check if data files exist
        
        Args:
            frame_names (List[str]): List of frame names
        
        Returns:
            Dict[str, bool]: Data availability for each frame
        """
        availability = {}
        for frame_name in frame_names:
            file_path = os.path.join(self.base_path, frame_name, "vis_data.pt")
            availability[frame_name] = os.path.exists(file_path)
        
        return availability
    
    def prepare_temporal_data(self, loader_result: Dict, 
                         feature_key: str = 'ori_img_bev_embed',
                         bev_h: int = 200, 
                            bev_w: int = 200) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Convert loader result to LongTermPredictor input format
        
        Args:
            loader_result: Output from UniBevFeatureLoader
            feature_key: Which feature to use
            bev_h, bev_w: BEV grid dimensions
        
        Returns:
            historical_feats: List of historical frame tensors
            current_feat: Current frame tensor
        """
        feature_sequence = loader_result['feature_sequence'][feature_key]
        
        # Current frame (index 0)
        current_feat = feature_sequence[0]  # (B, 40000, 256)
        
        # Reshape to BEV format: (B, 40000, 256) -> (B, 256, H, W)
        B = current_feat.shape[0]
        current_feat = current_feat.view(B, bev_h, bev_w, -1).permute(0, 3, 1, 2)
        
        # Historical frames (index 1+)
        historical_feats = []
        for i in range(1, len(feature_sequence)):
            hist_feat = feature_sequence[i]  # (B, 40000, 256)
            hist_feat = hist_feat.view(B, bev_h, bev_w, -1).permute(0, 3, 1, 2)
            historical_feats.append(hist_feat)
        
        return historical_feats, current_feat

# Example usage and test function
def test_loader_with_history():
    """Test loader functionality with history frames"""
    loader = UniBevFeatureLoader()
    
    # Load frame metadata first
    metadata_path = "/path/to/your/pointcloud_sequences.pkl"  # Update this path
    try:
        loader.load_frame_metadata(metadata_path)
    except:
        print("Note: Frame metadata not loaded. History functionality requires metadata.")
        return
    
    # Example frame names
    frame_names = [
        "n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392",
        "n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243547396",
    ]
    
    # Load batch features with history
    print(f"Loading batch features with history for {len(frame_names)} frames...")
    try:
        result = loader.load_batch_features_with_history(
            frame_names, 
            feature_keys=['ori_img_bev_embed', 'ori_pts_bev_embed'],
            max_history=6
        )
        
        # Access results
        feature_sequence = result['feature_sequence']
        relative_times = result['relative_times']
        frame_info = result['frame_info']
        
        print(f"\nResults:")
        print(f"Relative times shape: {relative_times.shape}")
        print(f"Relative times sample:\n{relative_times}")
        
        # Example: Access current frame features
        current_img_features = feature_sequence['ori_img_bev_embed'][0]  # (B, 40000, 256)
        history1_img_features = feature_sequence['ori_img_bev_embed'][1]  # (B, 40000, 256)
        
        print(f"\nFeature access example:")
        print(f"Current frame img features: {current_img_features.shape}")
        print(f"History 1 img features: {history1_img_features.shape}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_loader_with_history()