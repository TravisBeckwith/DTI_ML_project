#!/usr/bin/env python3
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras

# Disable TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_tensorflow_version():
    """Check TensorFlow compatibility and apply fixes"""
    tf_version = tf.__version__
    major, minor = map(int, tf_version.split('.')[:2])
    
    if major >= 2:
        # Handle newer TensorFlow versions
        tf.config.run_functions_eagerly(False)
        
        # Fix for Keras 3 compatibility
        def _patch_keras_tensor():
            """Add get_shape() method to KerasTensor for VoxelMorph compatibility"""
            try:
                # Try different import paths for KerasTensor
                KerasTensor = None
                try:
                    from keras.src.engine.keras_tensor import KerasTensor
                except ImportError:
                    try:
                        from keras.engine.keras_tensor import KerasTensor
                    except ImportError:
                        try:
                            from tensorflow.python.keras.engine.keras_tensor import KerasTensor
                        except ImportError:
                            pass
                
                if KerasTensor and not hasattr(KerasTensor, 'get_shape'):
                    # Add get_shape method
                    def get_shape(self):
                        return self.shape
                    KerasTensor.get_shape = get_shape
                    
                    # Also add _keras_shape property for older code
                    if not hasattr(KerasTensor, '_keras_shape'):
                        KerasTensor._keras_shape = property(lambda self: tuple(self.shape))
                    
                    print("Applied KerasTensor compatibility patch")
                    
            except Exception as e:
                print(f"Warning: Could not patch KerasTensor: {e}")
        
        _patch_keras_tensor()
    
    return f"{major}.{minor}"

# Apply TensorFlow compatibility fixes
tf_version = check_tensorflow_version()
print(f"Using TensorFlow version: {tf_version}")

class VoxelMorphDWIRegistration:
    def __init__(self, reference_vol, use_gpu=False):
        self.reference = reference_vol
        self.use_gpu = use_gpu
        self.model = None
        
        # Configure TensorFlow
        if not use_gpu:
            tf.config.set_visible_devices([], 'GPU')
        else:
            # Configure GPU memory growth
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                except:
                    pass  # Memory growth already configured
        
        self.setup_registration_network()

    def setup_registration_network(self):
        """Setup lightweight registration network for DWI"""
        try:
            import voxelmorph as vxm
            
            # Reduce model size for memory efficiency and compatibility
            nb_features = [[8, 16, 16], [16, 16, 8, 8]]  # Reduced from [16, 32, 32]
            
            # Create VoxelMorph model with compatibility fixes
            try:
                self.model = vxm.networks.VxmDense(
                    inshape=self.reference.shape,
                    nb_unet_features=nb_features,
                    int_steps=5,  # Reduced from 7
                    int_downsize=2
                )
                
                # Compile with appropriate loss
                self.model.compile(
                    optimizer=keras.optimizers.Adam(1e-4),
                    loss=[vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss],
                    loss_weights=[1.0, 0.01]
                )
                
                print("VoxelMorph model loaded successfully")
                return True
                
            except Exception as e:
                print(f"VoxelMorph model creation failed: {e}")
                self.setup_simple_registration()
                return False
        
        except ImportError:
            print("VoxelMorph not available, using simple registration")
            self.setup_simple_registration()
            return False

    def setup_simple_registration(self):
        """Fallback simple registration network"""
        try:
            # Use functional API for better compatibility
            inputs = keras.Input(shape=self.reference.shape + (2,))
            
            # Simple encoder with proper layer naming
            x = keras.layers.Conv3D(8, 3, activation='relu', padding='same', name='conv1')(inputs)
            x = keras.layers.MaxPooling3D(2, name='pool1')(x)
            x = keras.layers.Conv3D(16, 3, activation='relu', padding='same', name='conv2')(x)
            x = keras.layers.MaxPooling3D(2, name='pool2')(x)
            
            # Simple decoder
            x = keras.layers.UpSampling3D(2, name='upsample1')(x)
            x = keras.layers.Conv3D(8, 3, activation='relu', padding='same', name='conv3')(x)
            x = keras.layers.UpSampling3D(2, name='upsample2')(x)
            
            # Output displacement field
            outputs = keras.layers.Conv3D(3, 1, activation='tanh', padding='same', name='displacement')(x)
            outputs = keras.layers.Lambda(lambda x: x * 10, name='scale_displacement')(outputs)
            
            self.model = keras.Model(inputs, outputs, name='SimpleRegistration')
            self.model.compile(optimizer='adam', loss='mse')
            
            print("Simple registration model created")
            
        except Exception as e:
            print(f"Failed to create simple registration model: {e}")
            self.model = None

    def register_volume(self, moving_vol, quick_mode=True):
        """Register moving volume to reference"""
        try:
            # Normalize volumes
            ref_norm = self.normalize_volume(self.reference)
            mov_norm = self.normalize_volume(moving_vol)
            
            # Check data validity
            if np.any(np.isnan(mov_norm)) or np.any(np.isinf(mov_norm)):
                mov_norm = np.nan_to_num(mov_norm, nan=0.0, posinf=0.0, neginf=0.0)
            
            if quick_mode or self.model is None:
                # Use simple cross-correlation for quick registration
                return self.simple_registration(moving_vol, ref_norm, mov_norm)
            else:
                # Use full ML registration
                return self.ml_registration(moving_vol, ref_norm, mov_norm)
                
        except Exception as e:
            print(f"Registration failed: {e}")
            return moving_vol, np.zeros(moving_vol.shape + (3,))

    def normalize_volume(self, volume):
        """Normalize volume intensity"""
        volume = volume.astype(np.float32)
        
        # Mask out background
        mask = volume > np.percentile(volume, 5)  # More robust than > 0
        
        if np.sum(mask) > 0:
            mean_val = np.mean(volume[mask])
            std_val = np.std(volume[mask])
            if std_val > 0:
                normalized = np.zeros_like(volume)
                normalized[mask] = (volume[mask] - mean_val) / std_val
                return normalized
        
        return volume

    def simple_registration(self, moving_vol, ref_norm, mov_norm):
        """Simple registration using correlation"""
        try:
            from scipy import ndimage
            from scipy.optimize import minimize
            
            def correlation_metric(params):
                # Simple translation parameters
                shift = params[:3]
                
                # Apply shift
                shifted = ndimage.shift(mov_norm, shift, order=1, cval=0)
                
                # Calculate normalized cross-correlation
                mask = (ref_norm > 0.1) & (shifted > 0.1)  # More restrictive mask
                if np.sum(mask) < 100:  # Ensure sufficient overlap
                    return 1.0
                
                try:
                    corr = np.corrcoef(ref_norm[mask], shifted[mask])[0, 1]
                    return -corr if not np.isnan(corr) else 1.0
                except:
                    return 1.0
            
            # Optimize translation with bounds
            bounds = [(-10, 10), (-10, 10), (-5, 5)]  # Reasonable movement bounds
            result = minimize(correlation_metric, [0, 0, 0], method='L-BFGS-B', bounds=bounds)
            
            if result.success and result.fun < -0.3:  # Minimum correlation threshold
                # Apply optimal shift
                optimal_shift = result.x
                registered = ndimage.shift(moving_vol, optimal_shift, order=1, cval=0)
                
                # Create displacement field
                displacement = np.zeros(moving_vol.shape + (3,))
                for i in range(3):
                    displacement[..., i] = optimal_shift[i]
                
                return registered, displacement
            else:
                return moving_vol, np.zeros(moving_vol.shape + (3,))
                
        except ImportError:
            print("SciPy not available for registration")
            return moving_vol, np.zeros(moving_vol.shape + (3,))
        except Exception as e:
            print(f"Simple registration failed: {e}")
            return moving_vol, np.zeros(moving_vol.shape + (3,))

    def ml_registration(self, moving_vol, ref_norm, mov_norm):
        """Full ML-based registration with improved error handling"""
        if self.model is None:
            return self.simple_registration(moving_vol, ref_norm, mov_norm)
        
        try:
            # Check if it's a VoxelMorph model
            is_vxm = hasattr(self.model, 'register') or 'vxm' in str(type(self.model)).lower()
            
            if is_vxm:
                # VoxelMorph expects two separate inputs
                moving = mov_norm[np.newaxis, ..., np.newaxis]  # (1, D, H, W, 1)
                fixed = ref_norm[np.newaxis, ..., np.newaxis]   # (1, D, H, W, 1)
                
                # Ensure inputs are valid
                moving = np.nan_to_num(moving, nan=0.0, posinf=0.0, neginf=0.0)
                fixed = np.nan_to_num(fixed, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Predict
                with tf.device('/CPU:0' if not self.use_gpu else '/GPU:0'):
                    if hasattr(self.model, 'register'):
                        # Use register method if available
                        moved, displacement = self.model.register(moving, fixed)
                        registered = moved[0, ..., 0]
                        displacement = displacement[0]
                    else:
                        # Use predict method
                        outputs = self.model.predict([moving, fixed], verbose=0)
                        if isinstance(outputs, list) and len(outputs) == 2:
                            moved, displacement = outputs
                            registered = moved[0, ..., 0]
                            displacement = displacement[0]
                        else:
                            # Single output - assume it's displacement
                            displacement = outputs[0]
                            registered = self.apply_displacement(moving_vol, displacement)
                
                return registered, displacement
                
            else:
                # Simple model - uses stacked input
                input_vol = np.stack([ref_norm, mov_norm], axis=-1)
                input_vol = input_vol[np.newaxis, ...]  # Add batch dimension
                
                # Ensure input is valid
                input_vol = np.nan_to_num(input_vol, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Predict displacement field
                with tf.device('/CPU:0' if not self.use_gpu else '/GPU:0'):
                    displacement = self.model.predict(input_vol, verbose=0)[0]
                    registered = self.apply_displacement(moving_vol, displacement)
                
                return registered, displacement
                
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return self.simple_registration(moving_vol, ref_norm, mov_norm)

    def apply_displacement(self, volume, displacement):
        """Apply displacement field to volume"""
        try:
            from scipy import ndimage
            
            # Create coordinate grids
            coords = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
            
            # Apply displacement with bounds checking
            new_coords = []
            for i in range(3):
                displaced = coords[i] + displacement[..., i]
                # Clamp to valid range
                displaced = np.clip(displaced, 0, volume.shape[i] - 1)
                new_coords.append(displaced)
            
            # Interpolate
            registered = ndimage.map_coordinates(
                volume, new_coords, order=1, cval=0, prefilter=False
            )
            
            return registered
            
        except ImportError:
            print("SciPy not available for displacement application")
            return volume
        except Exception as e:
            print(f"Displacement application failed: {e}")
            return volume

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            
            # Clear TensorFlow session
            if hasattr(tf.keras.backend, 'clear_session'):
                tf.keras.backend.clear_session()
        except:
            pass  # Ignore cleanup errors

def run_voxelmorph_dwi_registration(dwi_file, output_file, sub, use_gpu=False, quick_mode=True):
    """Main DWI registration function with comprehensive error handling"""
    print(f"[{sub}] Starting {'quick' if quick_mode else 'full'} ML DWI registration")
    
    try:
        # Check file exists
        if not os.path.exists(dwi_file):
            raise FileNotFoundError(f"DWI file not found: {dwi_file}")
        
        # Load DWI with better error handling
        try:
            dwi_img = nib.load(dwi_file)
            dwi_data = dwi_img.get_fdata()
        except Exception as e:
            raise ValueError(f"Failed to load DWI data: {str(e)}")
        
        if dwi_data.ndim != 4:
            raise ValueError(f"Expected 4D DWI data, got {dwi_data.ndim}D")
        
        # Check data validity
        if np.any(np.isnan(dwi_data)) or np.any(np.isinf(dwi_data)):
            print(f"[{sub}] Warning: Invalid values detected in DWI data, cleaning...")
            dwi_data = np.nan_to_num(dwi_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check data range
        if np.max(dwi_data) == 0:
            raise ValueError("DWI data appears to be empty (all zeros)")
        
        # Use first volume as reference (b0)
        reference_vol = dwi_data[..., 0]
        
        # Initialize registration
        registrator = VoxelMorphDWIRegistration(reference_vol, use_gpu)
        
        registered_volumes = [reference_vol]  # Reference doesn't need registration
        
        print(f"[{sub}] Registering {dwi_data.shape[-1]-1} volumes to reference")
        
        # Process volumes with progress reporting
        success_count = 0
        for i in range(1, dwi_data.shape[-1]):
            try:
                moving_vol = dwi_data[..., i]
                
                # Skip empty volumes
                if np.max(moving_vol) == 0:
                    print(f"[{sub}] Skipping empty volume {i}")
                    registered_volumes.append(moving_vol)
                    continue
                
                registered_vol, displacement = registrator.register_volume(moving_vol, quick_mode)
                registered_volumes.append(registered_vol)
                success_count += 1
                
                if i % 20 == 0 or i == dwi_data.shape[-1] - 1:
                    print(f"[{sub}] Processed {i}/{dwi_data.shape[-1]-1} volumes ({success_count} successful)")
                    
            except Exception as e:
                print(f"[{sub}] Failed to register volume {i}: {e}")
                registered_volumes.append(dwi_data[..., i])  # Keep original
        
        # Stack registered volumes
        registered_dwi = np.stack(registered_volumes, axis=-1)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save with proper data type
        registered_img = nib.Nifti1Image(
            registered_dwi.astype(np.float32),
            dwi_img.affine,
            dwi_img.header
        )
        nib.save(registered_img, output_file)
        
        print(f"[{sub}] ML DWI registration completed successfully ({success_count}/{dwi_data.shape[-1]-1} volumes registered)")
        return True
        
    except Exception as e:
        print(f"[{sub}] ML DWI registration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if 'registrator' in locals():
            del registrator

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <dwi_file> <output_file> <subject_id> [use_gpu] [quick_mode]")
        sys.exit(1)
    
    dwi_file = sys.argv[1]
    output_file = sys.argv[2] 
    subject_id = sys.argv[3]
    use_gpu = len(sys.argv) > 4 and sys.argv[4].lower() == 'true'
    quick_mode = len(sys.argv) <= 5 or sys.argv[5].lower() != 'false'
    
    success = run_voxelmorph_dwi_registration(dwi_file, output_file, subject_id, use_gpu, quick_mode)
    sys.exit(0 if success else 1)
