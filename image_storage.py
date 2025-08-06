# image_storage.py
# Image Storage Manager with Azure Blob Storage support

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import shutil
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

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
    def store_image(self, file_path: str, extraction_id: int) -> str:
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
    def cleanup_old_images(self, days: int = 7) -> int:
        """Remove images older than specified days."""
        pass
    
    @abstractmethod
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass

class LocalImageStorage(BaseImageStorage):
    """Local file system storage for images."""
    
    def __init__(self, storage_dir: str = "radar_images"):
        """Initialize local storage."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.mapping_file = self.storage_dir / "image_mappings.json"
        self._ensure_mapping_file()
        logger.info(f"Local image storage initialized at: {self.storage_dir}")
    
    def _ensure_mapping_file(self):
        """Ensure the mapping file exists."""
        if not self.mapping_file.exists():
            with open(self.mapping_file, 'w') as f:
                json.dump({}, f)
    
    def _load_mappings(self) -> Dict[str, str]:
        """Load the current mappings."""
        try:
            with open(self.mapping_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
            return {}
    
    def _save_mappings(self, mappings: Dict[str, str]):
        """Save mappings to file."""
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(mappings, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving mappings: {e}")
    
    def store_image(self, file_path: str, extraction_id: int) -> str:
        """Store an image and associate it with an extraction ID."""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(file_path).suffix
            stored_filename = f"extraction_{extraction_id}_{timestamp}{file_extension}"
            stored_path = self.storage_dir / stored_filename
            
            # Copy file to storage
            shutil.copy2(file_path, stored_path)
            
            # Update mappings
            mappings = self._load_mappings()
            mappings[str(extraction_id)] = str(stored_path)
            self._save_mappings(mappings)
            
            logger.info(f"Image stored locally for extraction {extraction_id}")
            return str(stored_path)
            
        except Exception as e:
            logger.error(f"Error storing image: {e}")
            raise
    
    def get_image_path(self, extraction_id: int) -> Optional[str]:
        """Get the stored image path for an extraction ID."""
        mappings = self._load_mappings()
        path = mappings.get(str(extraction_id))
        
        if path and Path(path).exists():
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
    
    def cleanup_old_images(self, days: int = 7) -> int:
        """Remove images older than specified days."""
        removed = 0
        mappings = self._load_mappings()
        updated_mappings = {}
        
        try:
            for extraction_id, path in mappings.items():
                path_obj = Path(path)
                if path_obj.exists():
                    # Check age
                    age_days = (datetime.now() - datetime.fromtimestamp(path_obj.stat().st_mtime)).days
                    if age_days > days:
                        path_obj.unlink()
                        removed += 1
                        logger.info(f"Removed old image: {path}")
                    else:
                        updated_mappings[extraction_id] = path
                else:
                    # Path doesn't exist, remove from mappings
                    removed += 1
            
            self._save_mappings(updated_mappings)
            logger.info(f"Cleanup complete: {removed} images removed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return removed
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        mappings = self._load_mappings()
        total_size = 0
        valid_images = 0
        
        for path in mappings.values():
            path_obj = Path(path)
            if path_obj.exists():
                total_size += path_obj.stat().st_size
                valid_images += 1
        
        return {
            'storage_type': 'local',
            'total_images': len(mappings),
            'valid_images': valid_images,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'storage_dir': str(self.storage_dir)
        }

class AzureBlobImageStorage(BaseImageStorage):
    """Azure Blob Storage for images."""
    
    def __init__(self, connection_string: str = None, container_name: str = "radar-images"):
        """
        Initialize Azure Blob Storage.
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the blob container
        """
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
    
    def _load_mappings(self) -> Dict[str, str]:
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
    
    def _save_mappings(self, mappings: Dict[str, str]):
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
    
    def store_image(self, file_path: str, extraction_id: int) -> str:
        """Store an image in Azure Blob Storage."""
        try:
            # Generate unique blob name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = Path(file_path).suffix
            blob_name = f"extraction_{extraction_id}_{timestamp}{file_extension}"
            
            # Upload to Azure
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Get blob URL
            blob_url = blob_client.url
            
            # Update mappings
            mappings = self._load_mappings()
            mappings[str(extraction_id)] = blob_name
            self._save_mappings(mappings)
            
            logger.info(f"Image stored in Azure for extraction {extraction_id}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Error storing image in Azure: {e}")
            raise
    
    def get_image_path(self, extraction_id: int) -> Optional[str]:
        """Get the blob URL for an extraction ID."""
        mappings = self._load_mappings()
        blob_name = mappings.get(str(extraction_id))
        
        if blob_name:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            return blob_client.url
        return None
    
    def get_image_data(self, extraction_id: int) -> Optional[bytes]:
        """Get the image data from Azure Blob Storage."""
        mappings = self._load_mappings()
        blob_name = mappings.get(str(extraction_id))
        
        if blob_name:
            try:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_name
                )
                
                if blob_client.exists():
                    return blob_client.download_blob().readall()
                    
            except Exception as e:
                logger.error(f"Error getting image from Azure: {e}")
        
        return None
    
    def cleanup_old_images(self, days: int = 7) -> int:
        """Remove images older than specified days from Azure."""
        removed = 0
        mappings = self._load_mappings()
        updated_mappings = {}
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            for extraction_id, blob_name in mappings.items():
                try:
                    blob_client = container_client.get_blob_client(blob_name)
                    
                    if blob_client.exists():
                        properties = blob_client.get_blob_properties()
                        
                        if properties.last_modified.replace(tzinfo=None) < cutoff_date:
                            blob_client.delete_blob()
                            removed += 1
                            logger.info(f"Removed old blob: {blob_name}")
                        else:
                            updated_mappings[extraction_id] = blob_name
                    else:
                        removed += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing blob {blob_name}: {e}")
            
            self._save_mappings(updated_mappings)
            logger.info(f"Azure cleanup complete: {removed} blobs removed")
            
        except Exception as e:
            logger.error(f"Error during Azure cleanup: {e}")
        
        return removed
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get Azure storage statistics."""
        mappings = self._load_mappings()
        total_size = 0
        valid_images = 0
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            for blob_name in mappings.values():
                try:
                    blob_client = container_client.get_blob_client(blob_name)
                    if blob_client.exists():
                        properties = blob_client.get_blob_properties()
                        total_size += properties.size
                        valid_images += 1
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error getting Azure storage stats: {e}")
        
        return {
            'storage_type': 'azure_blob',
            'total_images': len(mappings),
            'valid_images': valid_images,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'container_name': self.container_name
        }

class ImageStorageManager:
    """
    Unified image storage manager that can use either local or Azure storage.
    """
    
    def __init__(self, storage_type: str = 'local', **kwargs):
        """
        Initialize the appropriate storage backend.
        
        Args:
            storage_type: 'local' or 'azure'
            **kwargs: Additional arguments for the storage backend
        """
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
    
    def store_image(self, file_path: str, extraction_id: int) -> str:
        """Store an image and associate it with an extraction ID."""
        return self.storage.store_image(file_path, extraction_id)
    
    def get_image_path(self, extraction_id: int) -> Optional[str]:
        """Get the stored image path/URL for an extraction ID."""
        return self.storage.get_image_path(extraction_id)
    
    def get_image_data(self, extraction_id: int) -> Optional[bytes]:
        """Get the image data for an extraction ID."""
        return self.storage.get_image_data(extraction_id)
    
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