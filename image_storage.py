# image_storage.py
# Image Storage Manager with Azure Blob Storage support

# image_storage.py - Image Storage Manager with original filename preservation

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import shutil
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Azure Storage SDK
try:
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    from azure.core.exceptions import ResourceNotFoundError
    AZURE_STORAGE_AVAILABLE = True
except ImportError:
    AZURE_STORAGE_AVAILABLE = False
    logger.warning("Azure Storage SDK not installed. Azure Blob Storage will not be available.")

class BaseImageStorage(ABC):
    """Abstract base class for image storage."""
    
    @abstractmethod
    def store_image(self, file_path: str, extraction_id: int, original_filename: str = None) -> str:
        """Store an image and associate it with an extraction ID."""
        pass
    
    @abstractmethod
    def get_image_path(self, extraction_id: int) -> Optional[str]:
        """Get the stored image path/URL for an extraction ID."""
        pass
    
    @abstractmethod
    def get_image_data(self, extraction_id: int) -> Optional[bytes]:
        """Get the image data for an extraction ID."""
        pass
    
    @abstractmethod
    def get_image_metadata(self, extraction_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata about the stored image including original filename."""
        pass
    
    @abstractmethod
    def cleanup_old_images(self, days: int = 7) -> int:
        """Remove images older than specified days."""
        pass
    
    @abstractmethod
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass

class LocalImageStorage(BaseImageStorage):
    """Local file system storage for images with original filename preservation."""
    
    def __init__(self, storage_dir: str = "radar_images"):
        """Initialize local storage."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.mapping_file = self.storage_dir / "image_mappings.json"
        self._ensure_mapping_file()
        self._lock = threading.Lock()  # Thread safety for file operations
        logger.info(f"Local image storage initialized at: {self.storage_dir}")
    
    def _ensure_mapping_file(self):
        """Ensure the mapping file exists."""
        if not self.mapping_file.exists():
            with open(self.mapping_file, 'w') as f:
                json.dump({}, f)
    
    def _load_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load the current mappings with thread safety."""
        with self._lock:
            try:
                with open(self.mapping_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading mappings: {e}")
                return {}
    
    def _save_mappings(self, mappings: Dict[str, Dict[str, Any]]):
        """Save mappings to file with atomic write."""
        with self._lock:
            try:
                # Atomic write using temp file
                temp_file = self.mapping_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(mappings, f, indent=2)
                temp_file.replace(self.mapping_file)
            except Exception as e:
                logger.error(f"Error saving mappings: {e}")
    
    def store_image(self, file_path: str, extraction_id: int, original_filename: str = None) -> str:
        """
        Store an image preserving original filename.
        
        Args:
            file_path: Path to the image file to store
            extraction_id: ID of the extraction
            original_filename: Original filename to preserve
        """
        try:
            # Use original filename if provided, otherwise use basename
            if original_filename is None:
                original_filename = Path(file_path).name
            
            # Create extraction-specific directory
            extraction_dir = self.storage_dir / f"extraction_{extraction_id}"
            extraction_dir.mkdir(exist_ok=True)
            
            # Preserve original filename but make it unique within the extraction
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(original_filename).suffix
            file_stem = Path(original_filename).stem
            
            # Store with original name but in unique directory
            stored_filename = original_filename
            stored_path = extraction_dir / stored_filename
            
            # If file already exists, add timestamp
            if stored_path.exists():
                stored_filename = f"{file_stem}_{timestamp}{file_extension}"
                stored_path = extraction_dir / stored_filename
            
            # Copy file to storage
            shutil.copy2(file_path, stored_path)
            
            # Update mappings with metadata
            mappings = self._load_mappings()
            mappings[str(extraction_id)] = {
                'path': str(stored_path),
                'original_filename': original_filename,
                'stored_filename': stored_filename,
                'timestamp': datetime.now().isoformat(),
                'size': os.path.getsize(stored_path)
            }
            self._save_mappings(mappings)
            
            logger.info(f"Image stored locally for extraction {extraction_id}: {original_filename}")
            return str(stored_path)
            
        except Exception as e:
            logger.error(f"Error storing image: {e}")
            raise
    
    def get_image_path(self, extraction_id: int) -> Optional[str]:
        """Get the stored image path for an extraction ID."""
        mappings = self._load_mappings()
        mapping = mappings.get(str(extraction_id))
        
        if mapping and 'path' in mapping:
            path = mapping['path']
            if Path(path).exists():
                return path
        return None
    
    def get_image_data(self, extraction_id: int) -> Optional[bytes]:
        """Get the image data for an extraction ID."""
        path = self.get_image_path(extraction_id)
        if path:
            try:
                with open(path, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading image data: {e}")
        return None
    
    def get_image_metadata(self, extraction_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata about the stored image."""
        mappings = self._load_mappings()
        return mappings.get(str(extraction_id))
    
    def cleanup_old_images(self, days: int = 7) -> int:
        """Remove images older than specified days."""
        removed = 0
        mappings = self._load_mappings()
        updated_mappings = {}
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            for extraction_id, mapping in mappings.items():
                if 'timestamp' in mapping:
                    stored_date = datetime.fromisoformat(mapping['timestamp'])
                    if stored_date < cutoff_date:
                        # Remove entire extraction directory
                        extraction_dir = self.storage_dir / f"extraction_{extraction_id}"
                        if extraction_dir.exists():
                            shutil.rmtree(extraction_dir)
                            removed += 1
                            logger.info(f"Removed old extraction directory: {extraction_dir}")
                    else:
                        updated_mappings[extraction_id] = mapping
                else:
                    # Keep if no timestamp (backward compatibility)
                    updated_mappings[extraction_id] = mapping
            
            self._save_mappings(updated_mappings)
            logger.info(f"Cleanup complete: {removed} extraction directories removed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return removed
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        mappings = self._load_mappings()
        total_size = 0
        valid_images = 0
        
        for mapping in mappings.values():
            if 'path' in mapping:
                path_obj = Path(mapping['path'])
                if path_obj.exists():
                    total_size += mapping.get('size', path_obj.stat().st_size)
                    valid_images += 1
        
        return {
            'storage_type': 'local',
            'total_images': len(mappings),
            'valid_images': valid_images,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'storage_dir': str(self.storage_dir)
        }

class AzureBlobImageStorage(BaseImageStorage):
    """Azure Blob Storage for images with original filename preservation."""
    
    def __init__(self, connection_string: str = None, container_name: str = "radar-images"):
        """Initialize Azure Blob Storage."""
        if not AZURE_STORAGE_AVAILABLE:
            raise ImportError("Azure Storage SDK is not installed. Install with: pip install azure-storage-blob")
        
        # Get connection string from environment if not provided
        if connection_string is None:
            connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
            if not connection_string:
                raise ValueError("Azure Storage connection string not provided")
        
        self.connection_string = connection_string
        self.container_name = container_name
        
        # Initialize blob service client
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Ensure container exists
        self._ensure_container()
        
        # Metadata blob for mappings
        self.metadata_blob_name = "image_mappings.json"
        
        logger.info(f"Azure Blob Storage initialized with container: {container_name}")
    
    def _ensure_container(self):
        """Ensure the container exists."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client = self.blob_service_client.create_container(self.container_name)
                logger.info(f"Created container: {self.container_name}")
        except Exception as e:
            logger.error(f"Error ensuring container: {e}")
            raise
    
    def _load_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load mappings from Azure Blob Storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.metadata_blob_name
            )
            
            if blob_client.exists():
                blob_data = blob_client.download_blob().readall()
                return json.loads(blob_data)
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error loading mappings from Azure: {e}")
            return {}
    
    def _save_mappings(self, mappings: Dict[str, Dict[str, Any]]):
        """Save mappings to Azure Blob Storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.metadata_blob_name
            )
            
            blob_client.upload_blob(
                json.dumps(mappings, indent=2),
                overwrite=True
            )
            
        except Exception as e:
            logger.error(f"Error saving mappings to Azure: {e}")
    
    def store_image(self, file_path: str, extraction_id: int, original_filename: str = None) -> str:
        """Store an image in Azure Blob Storage preserving original filename."""
        try:
            # Use original filename if provided
            if original_filename is None:
                original_filename = Path(file_path).name
            
            # Create blob path with extraction ID as folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"extractions/{extraction_id}/{original_filename}"
            
            # Upload to Azure
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Get blob URL
            blob_url = blob_client.url
            
            # Update mappings with metadata
            mappings = self._load_mappings()
            mappings[str(extraction_id)] = {
                'blob_name': blob_name,
                'original_filename': original_filename,
                'url': blob_url,
                'timestamp': datetime.now().isoformat(),
                'size': os.path.getsize(file_path)
            }
            self._save_mappings(mappings)
            
            logger.info(f"Image stored in Azure for extraction {extraction_id}: {original_filename}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Error storing image in Azure: {e}")
            raise
    
    def get_image_path(self, extraction_id: int) -> Optional[str]:
        """Get the blob URL for an extraction ID."""
        mappings = self._load_mappings()
        mapping = mappings.get(str(extraction_id))
        
        if mapping and 'url' in mapping:
            return mapping['url']
        elif mapping and 'blob_name' in mapping:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=mapping['blob_name']
            )
            return blob_client.url
        return None
    
    def get_image_data(self, extraction_id: int) -> Optional[bytes]:
        """Get the image data from Azure Blob Storage."""
        mappings = self._load_mappings()
        mapping = mappings.get(str(extraction_id))
        
        if mapping and 'blob_name' in mapping:
            try:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=mapping['blob_name']
                )
                
                if blob_client.exists():
                    return blob_client.download_blob().readall()
                    
            except Exception as e:
                logger.error(f"Error getting image from Azure: {e}")
        
        return None
    
    def get_image_metadata(self, extraction_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata about the stored image."""
        mappings = self._load_mappings()
        return mappings.get(str(extraction_id))
    
    def cleanup_old_images(self, days: int = 7) -> int:
        """Remove images older than specified days from Azure."""
        removed = 0
        mappings = self._load_mappings()
        updated_mappings = {}
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            for extraction_id, mapping in mappings.items():
                if 'timestamp' in mapping:
                    stored_date = datetime.fromisoformat(mapping['timestamp'])
                    if stored_date < cutoff_date and 'blob_name' in mapping:
                        try:
                            # Delete all blobs in the extraction folder
                            prefix = f"extractions/{extraction_id}/"
                            for blob in container_client.list_blobs(name_starts_with=prefix):
                                container_client.delete_blob(blob.name)
                            removed += 1
                            logger.info(f"Removed old extraction blobs: {prefix}")
                        except Exception as e:
                            logger.warning(f"Error deleting blob: {e}")
                    else:
                        updated_mappings[extraction_id] = mapping
                else:
                    updated_mappings[extraction_id] = mapping
            
            self._save_mappings(updated_mappings)
            logger.info(f"Azure cleanup complete: {removed} extractions removed")
            
        except Exception as e:
            logger.error(f"Error during Azure cleanup: {e}")
        
        return removed
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get Azure storage statistics."""
        mappings = self._load_mappings()
        total_size = sum(m.get('size', 0) for m in mappings.values())
        
        return {
            'storage_type': 'azure_blob',
            'total_images': len(mappings),
            'valid_images': len(mappings),  # Assume all are valid in Azure
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'container_name': self.container_name
        }

class ImageStorageManager:
    """Unified image storage manager with original filename preservation."""
    
    def __init__(self, storage_type: str = 'local', **kwargs):
        """Initialize the appropriate storage backend."""
        self.storage_type = storage_type
        
        if storage_type == 'azure':
            if not AZURE_STORAGE_AVAILABLE:
                logger.warning("Azure Storage not available, falling back to local storage")
                self.storage = LocalImageStorage(kwargs.get('storage_dir', 'radar_images'))
            else:
                try:
                    self.storage = AzureBlobImageStorage(
                        connection_string=kwargs.get('connection_string'),
                        container_name=kwargs.get('container_name', 'radar-images')
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize Azure storage: {e}")
                    logger.info("Falling back to local storage")
                    self.storage = LocalImageStorage(kwargs.get('storage_dir', 'radar_images'))
        else:
            self.storage = LocalImageStorage(kwargs.get('storage_dir', 'radar_images'))
        
        logger.info(f"ImageStorageManager initialized with {self.storage.__class__.__name__}")
    
    def store_image(self, file_path: str, extraction_id: int, original_filename: str = None) -> str:
        """Store an image preserving original filename."""
        return self.storage.store_image(file_path, extraction_id, original_filename)
    
    def get_image_path(self, extraction_id: int) -> Optional[str]:
        """Get the stored image path/URL for an extraction ID."""
        return self.storage.get_image_path(extraction_id)
    
    def get_image_data(self, extraction_id: int) -> Optional[bytes]:
        """Get the image data for an extraction ID."""
        return self.storage.get_image_data(extraction_id)
    
    def get_image_metadata(self, extraction_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata about the stored image."""
        return self.storage.get_image_metadata(extraction_id)
    
    def cleanup_old_images(self, days: int = 7) -> int:
        """Remove images older than specified days."""
        return self.storage.cleanup_old_images(days)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.storage.get_storage_stats()

# Factory function for easy initialization
def create_image_storage(use_azure: bool = None) -> ImageStorageManager:
    """
    Create an image storage manager based on environment configuration.
    
    Args:
        use_azure: Force Azure usage (if None, checks environment)
        
    Returns:
        Configured ImageStorageManager instance
    """
    if use_azure is None:
        # Check environment for storage type
        storage_type = os.environ.get('IMAGE_STORAGE_TYPE', 'local')
        use_azure = storage_type == 'azure'
    
    if use_azure:
        return ImageStorageManager(
            storage_type='azure',
            connection_string=os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        )
    else:
        storage_dir = os.environ.get('IMAGE_STORAGE_DIR', 'radar_images')
        return ImageStorageManager(
            storage_type='local',
            storage_dir=storage_dir
        )

if __name__ == "__main__":
    # Example usage
    print("Image Storage Manager Test")
    print("-" * 40)
    
    # Create storage manager based on environment
    storage = create_image_storage()
    
    # Get storage stats
    stats = storage.get_storage_stats()
    print(f"Storage Type: {stats['storage_type']}")
    print(f"Total Images: {stats['total_images']}")
    print(f"Valid Images: {stats['valid_images']}")
    print(f"Total Size: {stats['total_size_mb']} MB")