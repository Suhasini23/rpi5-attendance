import cv2
import os
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any

class CameraManager:
    """
    Camera Manager for Raspberry Pi 5 Industrial Shield and other platforms.
    Handles Picamera2 initialization, CSI port selection, and platform-specific optimizations.
    """
    
    def __init__(self, width: int = 1280, height: int = 720, framerate: int = 20, 
                 camera_port: str = 'CSI0', autofocus: bool = False, 
                 awb_mode: str = 'auto', exposure_mode: str = 'auto', 
                 horizontal_flip: bool = True, vertical_flip: bool = False, debug: bool = False):
        """
        Initialize CameraManager with configuration parameters.
        
        Args:
            width: Frame width (16:9 aspect ratio recommended)
            height: Frame height (16:9 aspect ratio recommended)
            framerate: Target FPS
            camera_port: CSI port ('CSI0' or 'CSI1')
            autofocus: Enable autofocus (if supported)
            awb_mode: Auto white balance mode
            exposure_mode: Exposure mode
            horizontal_flip: Enable horizontal flip (mirror effect)
            vertical_flip: Enable vertical flip
            debug: Enable debug output for color conversion
        """
        self.width = width
        self.height = height
        self.framerate = framerate
        self.camera_port = camera_port
        self.autofocus = autofocus
        self.awb_mode = awb_mode
        self.exposure_mode = exposure_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.debug = debug
        
        # Camera objects
        self.cap = None
        self.picam2 = None
        
        # Status tracking
        self.is_initialized = False
        self.platform = self._detect_platform()
        
        print(f"🔍 Platform detected: {self.platform}")
        print(f"📷 Camera config: {width}x{height} @ {framerate}fps, Port: {camera_port}")
        print(f"🔄 Transform: H-flip={'ON' if horizontal_flip else 'OFF'}, V-flip={'ON' if vertical_flip else 'OFF'}")

    def _detect_platform(self) -> str:
        """Detect the current platform and hardware."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo and 'BCM2712' in cpuinfo:
                    return 'raspberry_pi_5_industrial'
                elif 'Raspberry Pi' in cpuinfo:
                    return 'raspberry_pi'
        except:
            pass
        
        # Fallback platform detection
        import platform
        system = platform.system().lower()
        if system == 'darwin':
            return 'macos'
        elif system == 'linux':
            return 'linux'
        elif system == 'windows':
            return 'windows'
        else:
            return 'unknown'

    def _get_sensor_id(self) -> int:
        """Get the correct sensor ID based on CSI port."""
        if self.camera_port == 'CSI1':
            return 1
        elif self.camera_port == 'CSI0':
            return 0
        else:
            # Default to CSI0 if unknown port
            print(f"⚠️  Unknown camera port '{self.camera_port}', defaulting to CSI0")
            return 0

    def _check_available_controls(self):
        """Check what camera controls are available on this system."""
        if not self.picam2:
            return
        
        try:
            # Get available controls
            controls = self.picam2.camera_controls
            print("📋 Available camera controls:")
            for control in controls:
                print(f"   - {control}")
        except Exception as e:
            print(f"⚠️  Could not get camera controls: {e}")

    def _detect_camera_module(self) -> str:
        """Detect the type of camera module attached."""
        try:
            # Check for Camera Module 3 specific characteristics
            if hasattr(self, 'picam2') and self.picam2:
                # Try to get camera info
                try:
                    camera_info = self.picam2.camera_config
                    if camera_info:
                        print(f"📷 Camera info: {camera_info}")
                except:
                    pass
                
                # Check available formats
                try:
                    formats = self.picam2.sensor_modes
                    print(f"📋 Available sensor modes: {len(formats)}")
                    for i, mode in enumerate(formats):
                        print(f"   Mode {i}: {mode}")
                except:
                    pass
            
            # For now, assume Camera Module 3 if we're on RPi5
            if self.platform in ['raspberry_pi_5_industrial', 'raspberry_pi']:
                return 'camera_module_3'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"⚠️  Could not detect camera module: {e}")
            return 'unknown'

    def _check_supported_formats(self):
        """Check what formats are supported by the camera."""
        if not self.picam2:
            return []
        
        try:
            # Get available formats from sensor modes
            formats = []
            try:
                sensor_modes = self.picam2.sensor_modes
                print(f"📋 Available sensor modes: {len(sensor_modes)}")
                for i, mode in enumerate(sensor_modes):
                    if 'format' in mode:
                        formats.append(mode['format'])
                        print(f"   Mode {i}: {mode['format']} - {mode.get('size', 'unknown size')}")
            except Exception as e:
                print(f"⚠️  Could not get sensor modes: {e}")
            
            # Also try to get formats from camera config
            try:
                camera_config = self.picam2.camera_config
                if camera_config and 'format' in camera_config:
                    formats.append(camera_config['format'])
                    print(f"📷 Camera config format: {camera_config['format']}")
            except Exception as e:
                print(f"⚠️  Could not get camera config: {e}")
            
            return list(set(formats))  # Remove duplicates
            
        except Exception as e:
            print(f"⚠️  Could not check supported formats: {e}")
            return []

    def _configure_picamera2(self) -> bool:
        """Configure Picamera2 with industrial RPi5 optimizations."""
        try:
            from picamera2 import Picamera2
            
            sensor_id = self._get_sensor_id()
            print(f"🔧 Configuring Picamera2 with sensor ID: {sensor_id}")
            
            # Create initial Picamera2 instance to check supported formats
            self.picam2 = Picamera2(camera_num=sensor_id)
            
            # Check what formats are actually supported
            supported_formats = self._check_supported_formats()
            print(f"📋 Supported formats: {supported_formats}")
            
            # Choose formats to try based on what's supported
            if supported_formats:
                # Try supported formats first, then fallbacks
                formats_to_try = supported_formats + ['BGR888', 'RGB888', 'XBGR8888']
                # Remove duplicates while preserving order
                seen = set()
                formats_to_try = [f for f in formats_to_try if not (f in seen or seen.add(f))]
            else:
                # Fallback to common formats
                formats_to_try = ['BGR888', 'RGB888', 'XBGR8888']
            
            print(f"🔄 Will try formats: {formats_to_try}")
            
            for format_type in formats_to_try:
                try:
                    print(f"🔄 Trying format: {format_type}")
                    
                    # Create fresh Picamera2 instance for each attempt
                    if hasattr(self, 'picam2') and self.picam2:
                        try:
                            self.picam2.close()
                        except:
                            pass
                    
                    self.picam2 = Picamera2(camera_num=sensor_id)
                    
                    # Detect camera module type
                    camera_module = self._detect_camera_module()
                    print(f"📷 Detected camera module: {camera_module}")
                    
                    # Check available controls
                    self._check_available_controls()
                    
                    # Create configuration with current format
                    config = self.picam2.create_preview_configuration(
                        main={
                            "format": format_type, 
                            "size": (self.width, self.height)
                        }
                    )
                    
                    # Add transform controls for horizontal/vertical flip
                    if self.horizontal_flip or self.vertical_flip:
                        try:
                            # Set transform controls
                            if self.horizontal_flip:
                                config["transform"] = config.get("transform", {})
                                config["transform"]["hflip"] = True
                                print("🔄 Horizontal flip enabled")
                            
                            if self.vertical_flip:
                                config["transform"] = config.get("transform", {})
                                config["transform"]["vflip"] = True
                                print("🔄 Vertical flip enabled")
                                
                        except Exception as e:
                            print(f"⚠️  Transform controls not supported: {e}")
                    
                    # Try to add framerate control if supported
                    try:
                        config["controls"]["FrameDurationLimits"] = (int(1e6 / self.framerate), int(1e6 / self.framerate))
                        print(f"✅ Framerate set to: {self.framerate} FPS")
                    except Exception as e:
                        print(f"⚠️  Framerate control not supported: {e}")
                    
                    # Configure and start camera
                    self.picam2.configure(config)
                    self.picam2.start()
                    
                    # Test frame capture
                    test_frame = self.picam2.capture_array()
                    if test_frame is not None and test_frame.size > 0:
                        print(f"✅ Picamera2 configured successfully with format: {format_type}")
                        print(f"   Frame shape: {test_frame.shape}")
                        print(f"   Frame dtype: {test_frame.dtype}")
                        
                        # Set the capture object
                        self.cap = self.picam2
                        self.camera_format = format_type
                        return True
                    else:
                        print(f"❌ Format {format_type} produced invalid frame")
                        continue
                        
                except Exception as e:
                    print(f"❌ Format {format_type} failed: {e}")
                    # Clean up current picam2 instance
                    if hasattr(self, 'picam2') and self.picam2:
                        try:
                            self.picam2.close()
                        except:
                            pass
                        self.picam2 = None
                    continue
            
            print("❌ All Picamera2 formats failed")
            return False
            
        except ImportError:
            print("❌ Picamera2 not available - falling back to OpenCV")
            return False
        except Exception as e:
            print(f"❌ Picamera2 configuration failed: {e}")
            # Clean up picam2 object if it was created
            if hasattr(self, 'picam2') and self.picam2:
                try:
                    self.picam2.close()
                except:
                    pass
                self.picam2 = None
            return False

    def _configure_opencv(self) -> bool:
        """Configure OpenCV camera for non-Raspberry Pi platforms."""
        try:
            # For Raspberry Pi, try different camera devices
            if self.platform in ['raspberry_pi_5_industrial', 'raspberry_pi']:
                # Try common RPi camera devices
                camera_devices = [0, '/dev/video0', '/dev/video1']
            else:
                # For other platforms, try standard indices
                camera_devices = [0, 1, 2, 3]
            
            for device in camera_devices:
                try:
                    self.cap = cv2.VideoCapture(device)
                    if self.cap.isOpened():
                        print(f"✅ OpenCV camera opened on device: {device}")
                        break
                except Exception as e:
                    print(f"⚠️  Failed to open camera on {device}: {e}")
                    continue
            
            if self.cap and self.cap.isOpened():
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
                
                # Additional optimizations
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                
                # Verify settings were applied
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✅ OpenCV camera configured successfully")
                print(f"   Resolution: {actual_width}x{actual_height}")
                print(f"   FPS: {actual_fps}")
                return True
            else:
                print("❌ Failed to open any OpenCV camera")
                return False
                
        except Exception as e:
            print(f"❌ OpenCV configuration failed: {e}")
            return False

    def initialize(self) -> bool:
        """
        Initialize the camera based on platform detection.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.is_initialized:
            print("ℹ️  Camera already initialized")
            return True
        
        print(f"🚀 Initializing camera for platform: {self.platform}")
        
        success = False
        
        if self.platform in ['raspberry_pi_5_industrial', 'raspberry_pi']:
            # Try Picamera2 first for Raspberry Pi
            success = self._configure_picamera2()
            if not success:
                print("⚠️  Picamera2 failed, trying OpenCV fallback")
                success = self._configure_opencv()
        else:
            # Use OpenCV for other platforms
            success = self._configure_opencv()
        
        if success:
            self.is_initialized = True
            print(f"✅ Camera initialized successfully on {self.platform}")
        else:
            print(f"❌ Camera initialization failed on {self.platform}")
        
        return success

    def _detect_color_space(self, frame):
        """Detect the color space of the frame and convert to BGR if needed."""
        if frame is None or frame.size == 0:
            return frame
        
        # Check frame shape and properties
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1
        
        if self.debug:
            print(f"🔍 Frame analysis: {width}x{height}, {channels} channels, dtype: {frame.dtype}")
        
        if channels == 1:
            # Grayscale - convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if self.debug:
                print("🔄 Converted grayscale to BGR")
        elif channels == 3:
            # Check if it's already BGR by testing color distribution
            # BGR typically has blue channel with higher values in blue areas
            # RGB typically has red channel with higher values in red areas
            
            # Simple heuristic: check if blue channel has higher values than red channel
            blue_mean = np.mean(frame[:, :, 0])
            green_mean = np.mean(frame[:, :, 1])
            red_mean = np.mean(frame[:, :, 2])
            
            if self.debug:
                print(f"🔍 Channel means - B:{blue_mean:.1f}, G:{green_mean:.1f}, R:{red_mean:.1f}")
            
            # If red channel has higher values, it might be RGB
            if red_mean > blue_mean and red_mean > green_mean:
                if self.debug:
                    print("🔄 Detected RGB format, converting to BGR")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                if self.debug:
                    print("🔄 Assuming BGR format (no conversion needed)")
        elif channels == 4:
            # RGBA or BGRA - remove alpha channel
            frame = frame[:, :, :3]
            if self.debug:
                print("🔄 Removed alpha channel")
        
        return frame

    def _apply_transforms(self, frame):
        """Apply horizontal and vertical flips to the frame."""
        if frame is None or frame.size == 0:
            return frame
        
        # Apply horizontal flip if enabled
        if self.horizontal_flip:
            frame = cv2.flip(frame, 1)  # 1 for horizontal flip
            if self.debug:
                print("🔄 Applied horizontal flip")
        
        # Apply vertical flip if enabled
        if self.vertical_flip:
            frame = cv2.flip(frame, 0)  # 0 for vertical flip
            if self.debug:
                print("🔄 Applied vertical flip")
        
        return frame

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the camera.
        
        Returns:
            np.ndarray: Frame data or None if failed
        """
        if not self.is_initialized:
            if not self.initialize():
                return None
        
        if not self.cap:
            return None
        
        try:
            if hasattr(self.cap, 'capture_array'):
                # Picamera2 frame capture
                frame = self.cap.capture_array()
                if frame is not None and frame.size > 0:
                    # Handle different formats
                    if hasattr(self, 'camera_format'):
                        if self.camera_format == 'PC1B':
                            # PC1B is already in BGR format, no conversion needed
                            if self.debug:
                                print(f"🔄 Using PC1B format directly: {frame.shape}")
                        elif self.camera_format == 'BGR888':
                            # Camera Module 3 outputs RGB frames, convert to BGR for OpenCV
                            if self.debug:
                                print(f"🔄 Converting RGB to BGR for Camera Module 3: {frame.shape}")
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        elif self.camera_format == 'NV12':
                            # Convert NV12 to BGR for OpenCV compatibility
                            height, width = frame.shape[:2]
                            y_size = height * width
                            y = frame[:height, :width]
                            uv = frame[height:, :width//2]
                            
                            # Convert YUV to BGR
                            yuv = np.zeros((height, width, 3), dtype=np.uint8)
                            yuv[:, :, 0] = y
                            yuv[:, :, 1] = uv[:, :, 0].repeat(2, axis=1)
                            yuv[:, :, 2] = uv[:, :, 1].repeat(2, axis=1)
                            
                            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                            if self.debug:
                                print(f"🔄 Converted NV12 frame to BGR: {frame.shape}")
                        elif self.camera_format == 'RGB888':
                            # Convert RGB to BGR
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        elif self.camera_format == 'XBGR8888':
                            # Remove alpha channel if present
                            if frame.shape[2] == 4:
                                frame = frame[:, :, :3]
                        else:
                            # For other formats, detect color space automatically
                            frame = self._detect_color_space(frame)
                    
                    # Apply transforms (flips) if needed
                    frame = self._apply_transforms(frame)
                    
                    return frame
            else:
                # OpenCV frame capture
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Apply transforms (flips) if needed
                    frame = self._apply_transforms(frame)
                    return frame
                    
        except Exception as e:
            print(f"⚠️  Frame capture error: {e}")
        
        return None

    def get_status(self) -> Dict[str, Any]:
        """
        Get current camera status and configuration.
        
        Returns:
            dict: Status information
        """
        camera_working = False
        if self.is_initialized and self.cap:
            try:
                frame = self.read_frame()
                camera_working = frame is not None and frame.shape[0] > 0
            except:
                camera_working = False
        
        status = {
            'platform': self.platform,
            'initialized': self.is_initialized,
            'working': camera_working,
            'camera_type': 'Picamera2' if self.picam2 else 'OpenCV',
            'resolution': f"{self.width}x{self.height}",
            'aspect_ratio': f"{self.width/self.height:.2f}:1",
            'framerate': self.framerate,
            'camera_port': self.camera_port,
            'autofocus': self.autofocus,
            'awb_mode': self.awb_mode,
            'exposure_mode': self.exposure_mode,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip
        }
        
        # Add camera format if available
        if hasattr(self, 'camera_format'):
            status['camera_format'] = self.camera_format
        
        return status

    def close(self):
        """Clean up camera resources."""
        print("🧹 Cleaning up camera resources...")
        
        if self.picam2:
            try:
                self.picam2.close()
                print("✅ Picamera2 closed")
            except Exception as e:
                print(f"⚠️  Error closing Picamera2: {e}")
        elif self.cap:
            try:
                self.cap.release()
                print("✅ OpenCV camera released")
            except Exception as e:
                print(f"⚠️  Error releasing OpenCV camera: {e}")
        
        self.cap = None
        self.picam2 = None
        self.is_initialized = False
        print("✅ Camera cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
