import torch
import time
from typing import List, Dict, Tuple, Union, Optional
from collections import deque

class FeatureMemoryReader:
    def __init__(self, 
                 device: str = 'cuda',
                 history_max: int = 10,
                 time_threshold: float = 1.0,
                 verbose: bool = False):
        """
        Feature Memory Reader for temporal feature management
        
        Args:
            device (str): Device to store tensors on
            history_max (int): Maximum number of history frames to keep in memory
            time_threshold (float): Time threshold in seconds for scene continuity detection
            verbose (bool): Whether to print debug information
        """
        self.device = device
        self.history_max = history_max
        self.time_threshold = time_threshold
        self.verbose = verbose
        
        # Memory storage: deque for efficient append/pop operations
        # Each element: {'file_name': str, 'timestamp': float, 'feature': torch.Tensor}
        self.memory = deque(maxlen=history_max)
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'scene_resets': 0,
            'memory_hits': 0,
            'memory_misses': 0
        }
        
        if self.verbose:
            print(f"FeatureMemoryReader initialized:")
            print(f"  Device: {device}")
            print(f"  History max: {history_max}")
            print(f"  Time threshold: {time_threshold}s")
            print(f"  Verbose: {verbose}")
    
    def _log(self, message: str, level: str = 'INFO'):
        """Print message if verbose is enabled"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}][{level}] {message}")
    
    def get_latest_timestamp(self) -> Optional[float]:
        """Get the timestamp of the latest (most recent) frame in memory"""
        if len(self.memory) == 0:
            return None
        return self.memory[0]['timestamp']  # First element is latest
    
    def is_scene_continuous(self, input_timestamp: float) -> bool:
        """
        Check if input timestamp is continuous with current scene
        
        Args:
            input_timestamp (float): Input timestamp to check
            
        Returns:
            bool: True if continuous (time diff <= threshold), False if new scene
        """
        latest_timestamp = self.get_latest_timestamp()
        if latest_timestamp is None:
            return True  # Empty memory, consider as continuous
        
        time_diff = abs(input_timestamp - latest_timestamp)
        is_continuous = time_diff <= self.time_threshold
        
        self._log(f"Scene continuity check: latest={latest_timestamp:.3f}, "
                 f"input={input_timestamp:.3f}, diff={time_diff:.3f}s, "
                 f"continuous={is_continuous}")
        
        return is_continuous
    
    def clear_memory(self):
        """Clear all memory and reset statistics for new scene"""
        old_size = len(self.memory)
        
        # Explicitly delete tensor references to free CUDA memory
        for entry in self.memory:
            if 'feature' in entry and isinstance(entry['feature'], torch.Tensor):
                del entry['feature']
        
        self.memory.clear()
        self.stats['scene_resets'] += 1
        
        self._log(f"Memory cleared: removed {old_size} frames (scene reset #{self.stats['scene_resets']})")
    
    def get_history_features(self, input_timestamp: float, output_length: int) -> Tuple[List[torch.Tensor], List[str], List[float]]:
        """
        Get history features based on input timestamp
        
        Args:
            input_timestamp (float): Current timestamp 
            output_length (int): Desired number of history features to output
            
        Returns:
            Tuple[List[torch.Tensor], List[str], List[float]]: 
                - feature_list: List of feature tensors [newest, older, oldest, ...]
                - file_name_list: List of file names corresponding to features
                - timestamp_list: List of timestamps corresponding to features
                Returns ([], [], []) if scene discontinuous
        """
        # Check scene continuity
        if not self.is_scene_continuous(input_timestamp):
            self._log("Scene discontinuity detected, clearing memory and returning empty lists")
            self.clear_memory()
            self.stats['memory_misses'] += 1
            return [], [], []
        
        # If memory is empty, return empty lists
        if len(self.memory) == 0:
            self._log(f"Empty memory, returning empty lists")
            self.stats['memory_misses'] += 1
            return [], [], []
        
        self.stats['memory_hits'] += 1
        
        # Prepare output lists
        feature_list = []
        file_name_list = []
        timestamp_list = []
        memory_size = len(self.memory)
        
        if memory_size >= output_length:
            # Sufficient memory: take the most recent output_length frames
            for i in range(output_length):
                entry = self.memory[i]
                feature_list.append(entry['feature'])
                file_name_list.append(entry['file_name'])
                timestamp_list.append(entry['timestamp'])
            
            self._log(f"Sufficient memory: returning {output_length} recent features")
        else:
            # Insufficient memory: use all available + pad with oldest
            # First, add all available frames
            for entry in self.memory:
                feature_list.append(entry['feature'])
                file_name_list.append(entry['file_name'])
                timestamp_list.append(entry['timestamp'])
            
            if memory_size > 0:
                # Pad with oldest frame
                oldest_entry = self.memory[-1]  # Last element is oldest
                padding_needed = output_length - memory_size
                
                for i in range(padding_needed):
                    # Clone feature to avoid reference issues
                    padded_feature = oldest_entry['feature'].clone() if isinstance(oldest_entry['feature'], torch.Tensor) else oldest_entry['feature']
                    feature_list.append(padded_feature)
                    file_name_list.append(f"{oldest_entry['file_name']}_pad{i+1}")
                    timestamp_list.append(oldest_entry['timestamp'])
                
                self._log(f"Insufficient memory: returning {memory_size} real + {padding_needed} padded features")
            else:
                self._log("No frames in memory for padding")
        
        return feature_list, file_name_list, timestamp_list
    

    
    def update_memory(self, file_name: str, timestamp: float, feature: torch.Tensor):
        """
        Update memory with new frame data
        
        Args:
            file_name (str): Name/identifier of the frame
            timestamp (float): Timestamp of the frame
            feature (torch.Tensor): Feature tensor to store
        """
        # Purify input feature: detach from computation graph and clone
        if isinstance(feature, torch.Tensor):
            # detach(): break gradient connection, avoid memory leaks
            # clone(): create independent copy, avoid external modifications
            purified_feature = feature.detach().clone().to(self.device)
        else:
            purified_feature = feature
        
        # Create new memory entry
        new_entry = {
            'file_name': file_name,
            'timestamp': timestamp,
            'feature': purified_feature
        }
        
        # Add to front of deque (most recent at the beginning)
        # Note: if deque is at maxlen, oldest entry (at end) will be automatically removed
        old_size = len(self.memory)
        self.memory.appendleft(new_entry)
        
        self.stats['total_updates'] += 1
        
        # Log if an old entry was automatically removed due to maxlen
        if old_size == self.history_max:
            self._log(f"Memory updated: added '{file_name}' (timestamp={timestamp:.3f}), "
                     f"oldest entry automatically removed, memory size={len(self.memory)}/{self.history_max}")
        else:
            self._log(f"Memory updated: added '{file_name}' (timestamp={timestamp:.3f}), "
                     f"memory size={len(self.memory)}/{self.history_max}")
        
        # Log memory state
        if self.verbose and len(self.memory) > 1:
            timestamps = [entry['timestamp'] for entry in self.memory]
            self._log(f"Memory timestamps: {timestamps}")
    
    def get_memory_info(self) -> Dict:
        """Get current memory state information"""
        if len(self.memory) == 0:
            return {
                'size': 0,
                'max_size': self.history_max,
                'latest_timestamp': None,
                'oldest_timestamp': None,
                'time_span': 0.0,
                'file_names': []
            }
        
        timestamps = [entry['timestamp'] for entry in self.memory]
        file_names = [entry['file_name'] for entry in self.memory]
        
        return {
            'size': len(self.memory),
            'max_size': self.history_max,
            'latest_timestamp': timestamps[0],   # First is latest
            'oldest_timestamp': timestamps[-1],  # Last is oldest
            'time_span': timestamps[0] - timestamps[-1] if len(timestamps) > 1 else 0.0,
            'file_names': file_names
        }
    
    def get_stats(self) -> Dict:
        """Get memory usage statistics"""
        total_requests = self.stats['memory_hits'] + self.stats['memory_misses']
        hit_rate = self.stats['memory_hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'total_updates': self.stats['total_updates'],
            'scene_resets': self.stats['scene_resets'],
            'memory_hits': self.stats['memory_hits'],
            'memory_misses': self.stats['memory_misses'],
            'hit_rate': hit_rate,
            'current_memory_size': len(self.memory)
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'total_updates': 0,
            'scene_resets': 0,
            'memory_hits': 0,
            'memory_misses': 0
        }
        self._log("Statistics reset")
    
    def process_frame(self, 
                     input_timestamp: float,
                     file_name: Optional[str] = None,
                     input_feature: Optional[torch.Tensor] = None,
                     output_length: int = 6,
                     update_memory: bool = True) -> Tuple[Tuple[List[torch.Tensor], List[str], List[float]], Dict]:
        """
        Complete frame processing: get history features and optionally update memory
        
        Args:
            input_timestamp (float): Current frame timestamp
            file_name (str, optional): Frame identifier for memory update
            input_feature (torch.Tensor, optional): Feature to store in memory
            output_length (int): Number of history features to return
            update_memory (bool): Whether to update memory with input feature
            
        Returns:
            Tuple[Tuple[List[torch.Tensor], List[str], List[float]], Dict]: 
                ((feature_list, file_name_list, timestamp_list), memory_info)
        """
        self._log(f"Processing frame: timestamp={input_timestamp:.3f}, "
                 f"file_name={file_name}, update_memory={update_memory}")
        
        # Step 1: Get history features based on timestamp
        history_features = self.get_history_features(input_timestamp, output_length)
        
        # Step 2: Update memory if requested and inputs are provided
        if update_memory and file_name is not None and input_feature is not None:
            self.update_memory(file_name, input_timestamp, input_feature)
        elif update_memory:
            self._log("Warning: update_memory=True but file_name or input_feature is None")
        
        # Step 3: Get current memory info
        memory_info = self.get_memory_info()
        
        return history_features, memory_info


def test_feature_memory_reader():
    """Test the FeatureMemoryReader functionality"""
    print("="*60)
    print("Testing FeatureMemoryReader")
    print("="*60)
    
    # Initialize reader
    reader = FeatureMemoryReader(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        history_max=4,
        time_threshold=0.6,
        verbose=True
    )
    
    # Test data
    test_frames = [
        {'file_name': 'frame_001', 'timestamp': 1.0, 'feature_shape': (1, 256, 200, 200)},
        {'file_name': 'frame_002', 'timestamp': 1.1, 'feature_shape': (1, 256, 200, 200)},
        {'file_name': 'frame_003', 'timestamp': 1.2, 'feature_shape': (1, 256, 200, 200)},
        {'file_name': 'frame_004', 'timestamp': 1.3, 'feature_shape': (1, 256, 200, 200)},
        {'file_name': 'frame_005', 'timestamp': 1.4, 'feature_shape': (1, 256, 200, 200)},
        # Scene break
        {'file_name': 'frame_101', 'timestamp': 10.0, 'feature_shape': (1, 256, 200, 200)},
        {'file_name': 'frame_102', 'timestamp': 10.1, 'feature_shape': (1, 256, 200, 200)},
    ]
    
    print("\n1. Testing continuous scene processing:")
    print("-" * 40)
    
    for i, frame_data in enumerate(test_frames[:5]):
        # Create dummy feature
        feature = torch.randn(*frame_data['feature_shape'], device=reader.device)
        
        # Process frame
        (feature_list, file_name_list, timestamp_list), memory_info = reader.process_frame(
            input_timestamp=frame_data['timestamp'],
            file_name=frame_data['file_name'],
            input_feature=feature,
            output_length=3,
            update_memory=True
        )
        
        print(f"\nFrame {i+1} processed:")
        print(f"  Input: {frame_data['file_name']} @ {frame_data['timestamp']}")
        print(f"  History features returned: {len(feature_list)}")
        print(f"  Memory size: {memory_info['size']}/{memory_info['max_size']}")
        
        if len(timestamp_list) > 0:
            print(f"  History timestamps: {[f'{ts:.1f}' for ts in timestamp_list]}")
    
    print("\n2. Testing scene discontinuity:")
    print("-" * 40)
    
    # Process frame with large time gap (should trigger scene reset)
    frame_data = test_frames[5]  # timestamp 10.0
    feature = torch.randn(*frame_data['feature_shape'], device=reader.device)
    
    (feature_list, file_name_list, timestamp_list), memory_info = reader.process_frame(
        input_timestamp=frame_data['timestamp'],
        file_name=frame_data['file_name'],
        input_feature=feature,
        output_length=3,
        update_memory=True
    )
    
    print(f"New scene frame processed:")
    print(f"  Input: {frame_data['file_name']} @ {frame_data['timestamp']}")
    print(f"  History features returned: {len(feature_list)} (should be 0 for new scene)")
    print(f"  Memory size after reset: {memory_info['size']}")
    
    print("\n3. Testing get history features decomposed output:")
    print("-" * 40)
    
    # Add one more frame to have some history
    frame_data = test_frames[6]
    feature = torch.randn(*frame_data['feature_shape'], device=reader.device)
    reader.process_frame(
        input_timestamp=frame_data['timestamp'],
        file_name=frame_data['file_name'],
        input_feature=feature,
        output_length=3,
        update_memory=True
    )
    
    # Test decomposed output
    query_timestamp = 10.2
    feature_list, file_name_list, timestamp_list = reader.get_history_features(query_timestamp, output_length=4)
    memory_info = reader.get_memory_info()
    
    print(f"Decomposed history features query:")
    print(f"  Query timestamp: {query_timestamp}")
    print(f"  Feature list length: {len(feature_list)}")
    print(f"  File names: {file_name_list}")
    print(f"  Timestamps: {timestamp_list}")
    print(f"  Memory size unchanged: {memory_info['size']}")
    
    if len(feature_list) > 0:
        padded_count = sum(1 for name in file_name_list if '_pad' in name)
        print(f"  Real features: {len(feature_list) - padded_count}, Padded: {padded_count}")
        print(f"  Feature shapes: {[f.shape if hasattr(f, 'shape') else type(f) for f in feature_list]}")
        print(f"  History timestamps: {[f'{ts:.1f}' for ts in timestamp_list]}")
    
    print("\n4. Final statistics:")
    print("-" * 40)
    stats = reader.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_feature_memory_reader()