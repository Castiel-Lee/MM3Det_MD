import torch
import os
import pickle
import time
from typing import List, Dict, Tuple, Union
from collections import OrderedDict

class UniBevFeatureLoader:
    def __init__(self, base_path: str = "/dataset/shuangzhi/mmdet3d/unibev_LC_features/frames", 
                 device: str = 'cuda', 
                 expected_shape = (1, 40000, 256),
                 cache_size: int = 100,
                 enable_cache: bool = True,
                 extract_keys: List[str] = ['ori_pts_bev_embed', 'ori_img_bev_embed']):
        """
        UniBev feature data loader with history frame support and caching
        
        Args:
            base_path (str): Base path to feature data
            device (str): Device to load tensors to
            expected_shape (tuple): Expected shape of feature tensors
            cache_size (int): Maximum number of frames to cache in memory (default: 20)
            enable_cache (bool): Whether to enable caching
            extract_keys (List[str]): Only extract and cache these specific keys (default: ['ori_pts_bev_embed', 'ori_img_bev_embed'])
        """
        self.base_path = base_path
        self.frame_meta = None  # Will load frame metadata for history lookup
        self.device = device
        self.expected_shape = expected_shape
        
        # Caching parameters
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._cache = OrderedDict()  # LRU cache using OrderedDict
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        # Key extraction filter
        self.extract_keys = extract_keys
        if self.extract_keys:
            print(f"Key extraction enabled: {self.extract_keys}")
        else:
            print("Warning: No extract_keys specified, will cache all keys from files")
        
        print(f"UniBevFeatureLoader initialized:")
        print(f"  Device: {device}")
        print(f"  Cache enabled: {enable_cache}")
        print(f"  Cache size: {cache_size}")
        print(f"  Extract keys: {self.extract_keys}")
        print(f"  Expected shape: {expected_shape}")
    
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
    
    def _get_cache_key(self, frame_name: str) -> str:
        """Generate cache key for a frame"""
        return frame_name
    
    def _update_cache_stats(self, hit: bool):
        """Update cache statistics"""
        if hit:
            self._cache_stats['hits'] += 1
        else:
            self._cache_stats['misses'] += 1
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = self._cache_stats['hits'] / total if total > 0 else 0
        return {
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
            'max_cache_size': self.cache_size
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        self._cache_stats = {'hits': 0, 'misses': 0}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cache cleared")
    
    def load_single_frame_uncached(self, frame_name: str) -> Dict[str, torch.Tensor]:
        """
        Load feature data for a single frame without caching
        
        Args:
            frame_name (str): Point cloud frame name
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing feature data (filtered by extract_keys if specified)
        """
        file_path = os.path.join(self.base_path, frame_name, "vis_data.pt")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        
        # Load raw data
        raw_data = torch.load(file_path, map_location=self.device)
        
        # Apply key filtering if extract_keys is specified
        if self.extract_keys is not None:
            filtered_data = {}
            for key in self.extract_keys:
                if key in raw_data:
                    filtered_data[key] = raw_data[key]
                else:
                    # Create zero placeholder for missing required keys
                    print(f"Warning: {frame_name} missing required key '{key}', creating zero placeholder")
                    filtered_data[key] = torch.zeros(self.expected_shape, device=self.device)
            
            # Validate that we have all required keys
            missing_keys = set(self.extract_keys) - set(filtered_data.keys())
            if missing_keys:
                raise KeyError(f"Frame {frame_name} missing required keys: {missing_keys}")
            
            return filtered_data
        else:
            return raw_data
    
    def load_single_frame(self, frame_name: str) -> Dict[str, torch.Tensor]:
        """
        Load feature data for a single frame with caching
        
        Args:
            frame_name (str): Point cloud frame name
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing feature data
        """
        if not self.enable_cache:
            return self.load_single_frame_uncached(frame_name)
        
        cache_key = self._get_cache_key(frame_name)
        
        # Check cache first
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self._update_cache_stats(hit=True)
            return self._cache[cache_key]
        
        # Cache miss - load from disk
        self._update_cache_stats(hit=False)
        data = self.load_single_frame_uncached(frame_name)
        
        # Add to cache
        self._add_to_cache(cache_key, data)
        
        return data
    
    def _add_to_cache(self, cache_key: str, data: Dict[str, torch.Tensor]):
        """Add data to cache with LRU eviction and key filtering"""
        # Remove oldest items if cache is full
        while len(self._cache) >= self.cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        # Apply key filtering if extract_keys is specified
        if self.extract_keys is not None:
            # Only cache the specified keys
            filtered_data = {}
            for key in self.extract_keys:
                if key in data:
                    if isinstance(data[key], torch.Tensor):
                        filtered_data[key] = data[key].clone().detach()
                    else:
                        filtered_data[key] = data[key]
                else:
                    # This should not happen if load_single_frame_uncached works correctly
                    print(f"Warning: Key '{key}' not found during caching, creating placeholder")
                    filtered_data[key] = torch.zeros(self.expected_shape, device=self.device)
            
            cached_data = filtered_data
        else:
            # Cache all keys (original behavior)
            cached_data = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    cached_data[key] = value.clone().detach()
                else:
                    cached_data[key] = value
        
        self._cache[cache_key] = cached_data
    
    def preload_frames(self, frame_names: List[str], feature_keys: List[str] = None, verbose: bool = False):
        """
        Preload specific frames into cache
        
        Args:
            frame_names (List[str]): List of frame names to preload
            feature_keys (List[str]): Specific feature keys to preload (deprecated, use extract_keys in __init__)
        """
        if not self.enable_cache and verbose:
            print("Caching is disabled, skipping preload")
            return
        
        # If extract_keys is set, feature_keys parameter is ignored
        if self.extract_keys is not None and feature_keys is not None and verbose:
            print(f"Note: feature_keys parameter ignored, using extract_keys: {self.extract_keys}")
        
        if verbose:
            print(f"Preloading {len(frame_names)} frames into cache...")
        start_time = time.time()
        
        loaded_count = 0
        for i, frame_name in enumerate(frame_names):
            try:
                # Check if already in cache
                cache_key = self._get_cache_key(frame_name)
                if cache_key not in self._cache:
                    # Load with key filtering applied
                    data = self.load_single_frame_uncached(frame_name)
                    
                    # Validate that required keys are present (if extract_keys is set)
                    if self.extract_keys is not None:
                        missing_keys = set(self.extract_keys) - set(data.keys())
                        if missing_keys:
                            print(f"Warning: Frame {frame_name} missing keys: {missing_keys}")
                    
                    self._add_to_cache(cache_key, data)
                    loaded_count += 1
                
                # if verbose:
                #     print(f"  Preloaded {i + 1}/{len(frame_names)} frames")
                    
            except Exception as e:
                print(f"Error: failed to preload {frame_name}: {e}")
                raise e
        
        end_time = time.time()
        if verbose:
            print(f"Preloading completed: {loaded_count} new frames loaded in {end_time - start_time:.2f}s")
            print(f"Cache status: {len(self._cache)}/{self.cache_size} frames")
        
        # Print memory usage if extract_keys is used
        if self.extract_keys is not None and verbose:
            total_keys = len(self._cache) * len(self.extract_keys)
            print(f"Cached keys: {total_keys} tensors ({len(self.extract_keys)} keys per frame)")
    
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
                                       device: str = None,
                                       preload_batch: bool = True, validate_frame:  bool = False,
                                       strict_mode: bool = True) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """
        Load feature data in batch with history frames
        
        Args:
            frame_names (List[str]): List of current frame names with length B
            feature_keys (List[str]): List of feature keys to extract
            max_history (int): Maximum number of history frames
            verbose (bool): Whether to print detailed info
            device (str): Device to load tensors to (overrides self.device)
            preload_batch (bool): Whether to preload all required frames before processing
            strict_mode (bool): If True, raise errors on missing features. If False, use placeholders.
        
        Returns:
            Dict containing:
                - feature_sequence: List of [current_batch, hist1_batch, hist2_batch, ...]
                - relative_times: Tensor of shape (B, max_history+1) with relative time differences
                - frame_info: List of frame sequence info for each batch item
        """
        if device is None:
            device = self.device
            
        start_time = time.time()
        
        # Collect all unique frame names needed for this batch
        if preload_batch and self.enable_cache:
            all_required_frames = set()
            for current_frame_name in frame_names:
                try:
                    frame_sequence = self.get_history_frames(current_frame_name, max_history)
                    for frame_info in frame_sequence:
                        all_required_frames.add(frame_info['frame_name'])
                except Exception as e:
                    if strict_mode:
                        raise e
                    else:
                        print(f"Warning: Failed to get history for {current_frame_name}: {e}")
                        continue
            
            # Preload frames that are not in cache
            frames_to_preload = []
            for frame_name in all_required_frames:
                cache_key = self._get_cache_key(frame_name)
                if cache_key not in self._cache:
                    frames_to_preload.append(frame_name)
            
            if frames_to_preload:
                if verbose:
                    print(f"Preloading {len(frames_to_preload)} frames for this batch...")
                
                # Validate frames before preloading
                if validate_frame:
                    self._validate_frames_batch(frames_to_preload, feature_keys)
                
                self.preload_frames(frames_to_preload, feature_keys)
        
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
        successful_batches = 0
        for batch_idx, current_frame_name in enumerate(frame_names):
            try:
                # Get frame sequence (current + history)
                frame_sequence = self.get_history_frames(current_frame_name, max_history)
                result['frame_info'].append(frame_sequence)
                
                # Extract relative times
                relative_times = [frame['relative_time'] for frame in frame_sequence]
                result['relative_times'].append(relative_times)
                
                # Load features for each frame in sequence
                for seq_idx, frame_info in enumerate(frame_sequence):
                    frame_name = frame_info['frame_name']
                    
                    try:
                        # Load frame features (with caching)
                        frame_data = self.load_single_frame(frame_name)
                        
                        for key in feature_keys:
                            if strict_mode:
                                # Strong validation with detailed error info
                                if key not in frame_data:
                                    available_keys = list(frame_data.keys())
                                    raise KeyError(f"Error: {frame_name} - {key} not found in frame data. Available keys: {available_keys}")
                                
                                feature = frame_data[key]  # Shape: (1, 40000, 256)
                                
                                # Strong validation for tensor type
                                if not isinstance(feature, torch.Tensor):
                                    raise TypeError(f"Error: {frame_name} - {key} is not a tensor, got {type(feature)}")
                                
                                # Strong validation for shape
                                if feature.shape != self.expected_shape:
                                    raise ValueError(f"Error: {frame_name} - {key} shape {feature.shape} != expected {self.expected_shape}")
                                
                                result['feature_sequence'][key][seq_idx].append(feature)
                            else:
                                # Graceful handling mode
                                if key in frame_data:
                                    feature = frame_data[key]
                                    if isinstance(feature, torch.Tensor) and feature.shape == self.expected_shape:
                                        result['feature_sequence'][key][seq_idx].append(feature)
                                    else:
                                        print(f"Warning: {frame_name} - {key} invalid shape/type, using placeholder")
                                        placeholder = torch.zeros(self.expected_shape, device=device)
                                        result['feature_sequence'][key][seq_idx].append(placeholder)
                                else:
                                    print(f"Warning: {frame_name} - {key} missing, using placeholder")
                                    placeholder = torch.zeros(self.expected_shape, device=device)
                                    result['feature_sequence'][key][seq_idx].append(placeholder)
                    
                    except Exception as frame_error:
                        if strict_mode:
                            raise frame_error
                        else:
                            print(f"Warning: Failed to load {frame_name}: {frame_error}")
                            # Add placeholders for all features
                            for key in feature_keys:
                                placeholder = torch.zeros(self.expected_shape, device=device)
                                result['feature_sequence'][key][seq_idx].append(placeholder)
                
                successful_batches += 1
                
            except Exception as batch_error:
                if strict_mode:
                    raise batch_error
                else:
                    print(f"Warning: Failed to process batch {batch_idx} ({current_frame_name}): {batch_error}")
                    # Skip this batch item or add placeholder batch
                    continue
        
        if successful_batches == 0:
            raise ValueError("No successful batches processed!")
        
        # Convert lists to tensors
        for key in feature_keys:
            for seq_idx in range(max_history + 1):
                if len(result['feature_sequence'][key][seq_idx]) > 0:
                    # Concatenate batch dimension: (1, 40000, 256) * B -> (B, 40000, 256)
                    result['feature_sequence'][key][seq_idx] = torch.cat(
                        result['feature_sequence'][key][seq_idx], dim=0
                    ).to(device)
                else:
                    # Empty sequence, create placeholder
                    placeholder = torch.zeros((successful_batches, *self.expected_shape[1:]), device=device)
                    result['feature_sequence'][key][seq_idx] = placeholder
        
        # Convert relative times to tensor
        if len(result['relative_times']) > 0:
            result['relative_times'] = torch.tensor(result['relative_times'], dtype=torch.float32, device=device)
        else:
            result['relative_times'] = torch.zeros((successful_batches, max_history + 1), device=device)
        
        end_time = time.time()
        
        # Print summary
        if verbose:
            cache_stats = self.get_cache_stats()
            print(f"Loaded feature sequences for {successful_batches}/{batch_size} frames in {end_time - start_time:.3f}s:")
            print(f"  Cache stats - Hit rate: {cache_stats['hit_rate']:.2%}, Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
            for key in feature_keys:
                print(f"  {key}: {len(result['feature_sequence'][key])} time steps")
                for seq_idx, tensor in enumerate(result['feature_sequence'][key]):
                    step_name = "current" if seq_idx == 0 else f"history-{seq_idx}"
                    print(f"    {step_name}: {tensor.shape}")
            print(f"  relative_times: {result['relative_times'].shape}")
        
        return result
    
    def _validate_frames_batch(self, frame_names: List[str], feature_keys: List[str], verbose: bool = True):
        """
        Validate a batch of frames before processing
        
        Args:
            frame_names (List[str]): Frame names to validate
            feature_keys (List[str]): Required feature keys
        """
        if verbose:
            print(f"Validating {len(frame_names)} frames...")
        
        invalid_frames = []
        for frame_name in frame_names:
            try:
                # Load without cache for validation
                data = self.load_single_frame_uncached(frame_name)
                
                # Check required keys
                missing_keys = [key for key in feature_keys if key not in data]
                if missing_keys:
                    invalid_frames.append((frame_name, f"Missing keys: {missing_keys}"))
                    continue
                
                # Check tensor properties
                for key in feature_keys:
                    feature = data[key]
                    if not isinstance(feature, torch.Tensor):
                        invalid_frames.append((frame_name, f"{key} is not tensor: {type(feature)}"))
                        break
                    
                    if feature.shape != self.expected_shape:
                        invalid_frames.append((frame_name, f"{key} wrong shape: {feature.shape}"))
                        break
                        
            except Exception as e:
                invalid_frames.append((frame_name, f"Load error: {e}"))
        
        if invalid_frames and verbose:
            print("Invalid frames found:")
            for frame_name, error in invalid_frames[:5]:  # Show first 5 errors
                print(f"  {frame_name}: {error}")
            
            if len(invalid_frames) > 5:
                print(f"  ... and {len(invalid_frames) - 5} more")
            
            raise ValueError(f"Found {len(invalid_frames)} invalid frames out of {len(frame_names)}")
        if verbose:
            print("All frames validated successfully!")
    
    def check_frame_features(self, frame_names: List[str], 
                           feature_keys: List[str] = ['ori_img_bev_embed', 'ori_pts_bev_embed']) -> Dict[str, Dict]:
        """
        Check which features are available for given frames
        
        Args:
            frame_names (List[str]): List of frame names to check
            feature_keys (List[str]): List of feature keys to check
        
        Returns:
            Dict[str, Dict]: Frame availability and feature status
        """
        results = {}
        
        for frame_name in frame_names:
            try:
                frame_data = self.load_single_frame(frame_name)
                available_keys = list(frame_data.keys())
                
                feature_status = {}
                for key in feature_keys:
                    if key in frame_data:
                        tensor = frame_data[key]
                        feature_status[key] = {
                            'available': True,
                            'shape': tensor.shape if isinstance(tensor, torch.Tensor) else 'not_tensor',
                            'dtype': str(tensor.dtype) if isinstance(tensor, torch.Tensor) else 'not_tensor'
                        }
                    else:
                        feature_status[key] = {'available': False}
                
                results[frame_name] = {
                    'file_exists': True,
                    'all_keys': available_keys,
                    'features': feature_status
                }
                
            except FileNotFoundError:
                results[frame_name] = {
                    'file_exists': False,
                    'features': {key: {'available': False} for key in feature_keys}
                }
            except Exception as e:
                results[frame_name] = {
                    'file_exists': True,
                    'error': str(e),
                    'features': {key: {'available': False} for key in feature_keys}
                }
        
        return results
    
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
def test_cached_loader():
    """Test cached loader functionality with default parameters"""
    # Test with default parameters
    loader = UniBevFeatureLoader(
        device='cuda'  # Using all defaults: cache_size=20, extract_keys=['ori_pts_bev_embed', 'ori_img_bev_embed']
    )
    
    # Load frame metadata first
    metadata_path = "/dataset/shuangzhi/mmdet3d/unibev_LC_features/pointcloud_sequences.json"
    try:
        loader.load_frame_metadata(metadata_path)
    except Exception as e:
        print(f"Note: Frame metadata not loaded: {e}")
        return
    
    # Example frame names
    frame_names = [
        "n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392",
    ]
    
    # Test single frame loading first
    print("Testing single frame loading with default extract_keys:")
    try:
        single_data = loader.load_single_frame(frame_names[0])
        print(f"Loaded keys: {list(single_data.keys())}")
        for key, value in single_data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
    except Exception as e:
        print(f"Single frame test failed: {e}")
        return
    
    # First load (should be cache misses)
    print("\nFirst load (cache misses expected):")
    start_time = time.time()
    result1 = loader.load_batch_features_with_history(
        frame_names, 
        feature_keys=['ori_img_bev_embed', 'ori_pts_bev_embed'],  # Should match default extract_keys
        max_history=6,
        verbose=True
    )
    print(f"First load time: {time.time() - start_time:.3f}s")
    
    # Second load (should be cache hits)
    print("\nSecond load (cache hits expected):")
    start_time = time.time()
    result2 = loader.load_batch_features_with_history(
        frame_names, 
        feature_keys=['ori_img_bev_embed', 'ori_pts_bev_embed'],
        max_history=6,
        verbose=True
    )
    print(f"Second load time: {time.time() - start_time:.3f}s")
    
    # Print final cache statistics
    stats = loader.get_cache_stats()
    print(f"\nFinal cache statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show memory usage with small cache
    print(f"\nMemory efficiency with cache_size=20:")
    print(f"  Cached frames: {len(loader._cache)}")
    print(f"  Max cache frames: {loader.cache_size}")
    print(f"  Extract keys: {loader.extract_keys}")
    
    estimated_memory = len(loader._cache) * len(loader.extract_keys) * 80  # MB per tensor
    print(f"  Estimated cache memory: ~{estimated_memory}MB")

if __name__ == "__main__":
    test_cached_loader()