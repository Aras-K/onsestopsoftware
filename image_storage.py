import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import shutil
from datetime import datetime
import json

class ImageStorageManager:
    """Manages persistent storage of radar images for review purposes."""
    
    def __init__(self, storage_dir: str = "radar_images"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.mapping_file = self.storage_dir / "image_mappings.json"
        self._ensure_mapping_file()
    
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
        except:
            return {}
    
    def _save_mappings(self, mappings: Dict[str, str]):
        """Save mappings to file."""
        with open(self.mapping_file, 'w') as f:
            json.dump(mappings, f, indent=2)
    
    def store_image(self, file_path: str, extraction_id: int) -> str:
        """
        Store an image and associate it with an extraction ID.
        
        Args:
            file_path: Path to the image file
            extraction_id: ID from the extraction process
            
        Returns:
            Path to the stored image
        """
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
        
        return str(stored_path)
    
    def get_image_path(self, extraction_id: int) -> Optional[str]:
        """
        Get the stored image path for an extraction ID.
        
        Args:
            extraction_id: The extraction ID
            
        Returns:
            Path to the image file or None if not found
        """
        mappings = self._load_mappings()
        path = mappings.get(str(extraction_id))
        
        if path and Path(path).exists():
            return path
        return None
    
    def get_image_data(self, extraction_id: int) -> Optional[bytes]:
        """
        Get the image data for an extraction ID.
        
        Args:
            extraction_id: The extraction ID
            
        Returns:
            Image data as bytes or None if not found
        """
        path = self.get_image_path(extraction_id)
        if path:
            try:
                with open(path, 'rb') as f:
                    return f.read()
            except:
                pass
        return None
    
    def cleanup_old_images(self, days: int = 7) -> int:
        """
        Remove images older than specified days.
        
        Args:
            days: Number of days to keep images
            
        Returns:
            Number of images removed
        """
        removed = 0
        mappings = self._load_mappings()
        updated_mappings = {}
        
        for extraction_id, path in mappings.items():
            path_obj = Path(path)
            if path_obj.exists():
                # Check age
                age_days = (datetime.now() - datetime.fromtimestamp(path_obj.stat().st_mtime)).days
                if age_days > days:
                    path_obj.unlink()
                    removed += 1
                else:
                    updated_mappings[extraction_id] = path
            else:
                # Path doesn't exist, remove from mappings
                removed += 1
        
        self._save_mappings(updated_mappings)
        return removed
    
    def get_storage_stats(self) -> Dict[str, any]:
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
            'total_images': len(mappings),
            'valid_images': valid_images,
            'total_size_mb': total_size / (1024 * 1024),
            'storage_dir': str(self.storage_dir)
        }